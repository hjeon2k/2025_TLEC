# bqer.py
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict, List


# --------------------- Layer definition --------------------------------------

class BQERDecoderLayer(nn.Module):
    """
    TCEC-like residual correction with channel-wise (group_size=-1) or group-wise (group_size>0) K.
    """
    def __init__(
        self,
        inner_layer: nn.Module,
        hidden_size: int,
        K_current: Optional[torch.Tensor] = None,   # (C,) if channel-wise, (G,) if group-wise
        K_prev: Optional[torch.Tensor] = None,      # unused for m=0
        window_m: int = 0,
        alpha: float = 0.5,
        place: str = "post",
        group_size: int = -1,                       # -1=channel-wise, >0=group-wise
    ):
        super().__init__()
        assert place in ("post",)
        assert window_m in (0, 1)
        self.inner = inner_layer
        self.hidden_size = hidden_size
        self.window_m = int(window_m)
        self.alpha = float(alpha)
        self.place = place
        self.group_size = int(group_size)

        # Store K in compact form; expand lazily in forward.
        if self.group_size is None or self.group_size <= 0:
            # channel-wise: expect (C,)
            C = hidden_size
            Kc = torch.zeros(C, dtype=torch.float32)
            if K_current is not None:
                assert K_current.shape == (C,), f"Expected K_current shape {(C,)}, got {tuple(K_current.shape)}"
                Kc = K_current.detach().clone().to(torch.float32)
            self.register_buffer("K_vec", Kc, persistent=True)      # (C,)
            self.register_buffer("K_groups", torch.empty(0), persistent=False)
        else:
            # group-wise: expect (G,)
            G = (hidden_size + self.group_size - 1) // self.group_size
            Kg = torch.zeros(G, dtype=torch.float32)
            if K_current is not None:
                assert K_current.dim() == 1, f"Expected 1D K_current with length G={G}"
                Kg = K_current.detach().clone().to(torch.float32)
            self.register_buffer("K_groups", Kg, persistent=True)   # (G,)
            self.register_buffer("K_vec", torch.empty(0), persistent=False)

        self.register_buffer("_prev_y", None, persistent=False)

    # ---- public helpers -------------------------------------------------

    def set_K(self, K_current: Optional[torch.Tensor] = None, K_prev: Optional[torch.Tensor] = None):
        """Set/replace K buffers. Tensors should be shape (hidden_size,)."""
        if K_current is not None:
            assert K_current.shape == (self.hidden_size,)
            self.K_current = K_current.detach().to(self.K_current.device, dtype=self.K_current.dtype)
        if K_prev is not None:
            assert K_prev.shape == (self.hidden_size,)
            self.K_prev = K_prev.detach().to(self.K_prev.device, dtype=self.K_prev.dtype)

    def reset_cache(self):
        """Reset stored y_{l-1} (e.g., between prompts or at generation start)."""
        self._prev_y = None

    # ---- core -----------------------------------------------------------

    def _expanded_K(self, like: torch.Tensor) -> torch.Tensor:
        """Return (1,1,C) K expanded to hidden_size, matching `like` dtype/device."""
        C = self.hidden_size
        if self.group_size is None or self.group_size <= 0:
            k = self.K_vec  # (C,)
        else:
            g = self.group_size
            k = torch.repeat_interleave(self.K_groups, g)[:C]  # (C,)
        return k.view(1,1,-1).to(device=like.device, dtype=like.dtype)

    @torch.no_grad()
    def _compute_delta(self, y_cur: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0.0:
            return torch.zeros_like(y_cur)
        K = self._expanded_K(y_cur)       # (1,1,C)
        return self.alpha * (K * y_cur)   # add estimated error

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        x_in = hidden_states
        out = self.inner(hidden_states, *args, **kwargs)
        x_out, tail = (out, ()) if isinstance(out, torch.Tensor) else (out[0], out[1:])
        y_cur = x_out - x_in
        x_out = x_out + self._compute_delta(y_cur)
        self._prev_y = y_cur.detach()
        return x_out if isinstance(out, torch.Tensor) else (x_out, *tail)

# --------------------- Collect residuals and compute Ks --------------------------------------

@torch.no_grad()
def collect_layer_residuals(model: nn.Module, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """
    Run a single forward pass and collect per-layer residuals y_l = out - in.
    Returns: list length L (num layers), each item shape (B, T, C).
    """
    # where are the decoder layers?
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Could not locate Llama decoder layers.")

    y_buffers: List[List[torch.Tensor]] = [[] for _ in range(len(layers))]
    handles = []

    def hook_factory(slot_idx):
        def hook(mod, inputs, outputs):
            # inputs[0]: hidden_states in, outputs[0 or tensor]: hidden_states out
            in_states = inputs[0]
            out_states = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            y = (out_states - in_states).detach()
            y_buffers[slot_idx].append(y)
            return outputs
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook_factory(i)))

    _ = model(**batch)

    for h in handles:
        h.remove()

    # stack per layer -> (N=1, B, T, C) cat later across many batches
    return [torch.cat(buf, dim=0) if len(buf) > 0 else None for buf in y_buffers]


@torch.no_grad()
def collect_residuals_over_loader(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    max_batches: int = 64,
) -> List[torch.Tensor]:
    """
    Iterate over a loader, gather residuals y per layer.
    Returns: list length L, each item shape (N_total, T, C) on CPU.
    """
    model.eval().to(device)
    # initialize lists
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        L = len(model.model.layers)
    else:
        L = len(model.layers)

    accum: List[List[torch.Tensor]] = [[] for _ in range(L)]

    for bi, batch in tqdm(enumerate(dataloader), total=max_batches, desc="Collecting residuals over loader"):
        if bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        ys = collect_layer_residuals(model, batch)  # list of (B, T, C)
        for i, y in enumerate(ys):
            if y is not None:
                accum[i].append(y.to("cpu"))
    # stack per layer
    out: List[torch.Tensor] = []
    for i in range(L):
        if len(accum[i]) == 0:
            out.append(None)
        else:
            out.append(torch.cat(accum[i], dim=0))  # (N_total, T, C)
    return out


@torch.no_grad()
def bqer_K_closed_form(
    y_q_list: torch.Tensor,      # (N,T,C)
    y_fp_list: torch.Tensor,     # (N,T,C)
    lambda1: float = 1e-6,
    clip: float = 0.5,
    group_size: int = -1,        # -1 channel-wise, >0 group-wise
) -> torch.Tensor:
    """
    Returns K with shape:
      - (C,) if group_size<=0 (channel-wise)
      - (G,) if group_size>0  where G=ceil(C/group_size) (group-wise)
    Uses ridge closed-form in fp32:
      per-channel: K = (⟨yq,yf⟩ - ⟨yq,yq⟩) / (⟨yq,yq⟩ + λ)
      per-group  : same but sums within groups.
    """
    assert y_q_list.dim() == 3 and y_fp_list.dim() == 3
    N, T, C = y_q_list.shape
    yq = y_q_list.reshape(-1, C).to(torch.float32)  # (M,C)
    yf = y_fp_list.reshape(-1, C).to(torch.float32)

    if group_size is None or group_size <= 0:
        # channel-wise
        Sqq = (yq * yq).sum(dim=0)      # (C,)
        Sfq = (yq * yf).sum(dim=0)      # (C,)
        K = (Sfq - Sqq) / (Sqq + lambda1)
        if clip is not None:
            K = K.clamp(-clip, clip)
        return K  # (C,)

    # group-wise
    g = int(group_size)
    G = (C + g - 1) // g
    pad = G * g - C
    yq_pad = torch.nn.functional.pad(yq, (0, pad))  # (M,G*g)
    yf_pad = torch.nn.functional.pad(yf, (0, pad))  # (M,G*g)
    yqv = yq_pad.view(-1, G, g)                     # (M,G,g)
    yfv = yf_pad.view(-1, G, g)

    Sqq_g = (yqv * yqv).sum(dim=(0,2))             # (G,)
    Sfq_g = (yqv * yfv).sum(dim=(0,2))             # (G,)
    Kg    = (Sfq_g - Sqq_g) / (Sqq_g + lambda1)    # (G,)
    if clip is not None:
        Kg = Kg.clamp(-clip, clip)
    return Kg  # (G,)


@torch.no_grad()
def calibrate_bqer_Ks_from_loader(
    fp_model: nn.Module,
    q_model: nn.Module,
    dataloader,
    device: str = "cuda",
    max_batches: int = 64,
    lambda1: float = 1e-6,
    clip: float = 0.5,
    group_size: int = -1,                 # NEW
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    y_q_layers = collect_residuals_over_loader(q_model, dataloader, device=device, max_batches=max_batches)
    y_fp_layers = collect_residuals_over_loader(fp_model, dataloader, device=device, max_batches=max_batches)

    if hasattr(q_model, "model") and hasattr(q_model.model, "layers"):
        L = len(q_model.model.layers)
        C = q_model.config.hidden_size
    else:
        L = len(q_model.layers)
        C = q_model.config.hidden_size

    Ks_cur, Ks_prev = {}, {}
    for i in tqdm(range(L), total=L, desc=f"Calibrating BQER Ks"):
        yq = y_q_layers[i]
        yf = y_fp_layers[i]
        if (yq is None) or (yf is None):
            if group_size is None or group_size <= 0:
                K = torch.zeros(C, dtype=torch.float32)
            else:
                G = (C + group_size - 1) // group_size
                K = torch.zeros(G, dtype=torch.float32)
        else:
            K = bqer_K_closed_form(yq, yf, lambda1=lambda1, clip=clip, group_size=group_size).to("cpu")
        Ks_cur[i] = K
        Ks_prev[i] = K.clone()

        del yq, yf
        torch.cuda.empty_cache()
        
    return Ks_cur, Ks_prev

# --------------------- Apply BQER wrapper --------------------------------------

def wrap_llama_with_bqer(
    model: nn.Module,
    Ks_current: Dict[int, torch.Tensor],   # (C,) if channel-wise, (G,) if group-wise
    Ks_prev: Optional[Dict[int, torch.Tensor]] = None,
    window_m: int = 0,
    alpha: float = 0.5,
    group_size: int = -1,                  # -1 channel-wise, >0 group-wise
):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        hidden_size = model.config.hidden_size
    elif hasattr(model, "layers"):
        layers = model.layers
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Could not locate Llama decoder layers on the given model.")

    for idx, layer in enumerate(layers):
        Kc = Ks_current.get(idx, None)
        wrapper = BQERDecoderLayer(
            inner_layer=layer,
            hidden_size=hidden_size,
            K_current=Kc,
            window_m=0,
            alpha=alpha,
            place="post",
            group_size=group_size,
        )
        layers[idx] = wrapper


def apply_bqer_wrapper(
    q_model: nn.Module,
    Ks_cur: Dict[int, torch.Tensor],
    Ks_prev: Dict[int, torch.Tensor],
    window_m: int = 0,
    alpha: float = 0.5,
    group_size: int = -1,      # NEW
):
    wrap_llama_with_bqer(
        q_model,
        Ks_current=Ks_cur,
        Ks_prev=Ks_prev,
        window_m=window_m,
        alpha=alpha,
        group_size=group_size,
    )
    for layer in (q_model.model.layers if hasattr(q_model, "model") else q_model.layers):
        if isinstance(layer, BQERDecoderLayer):
            layer.reset_cache()


# --------------------- quick usage example -----------------------------------
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
# # Suppose you’ve estimated per-layer K’s (dicts of tensors with shape (hidden_size,))
# Ks_cur = {i: torch.zeros(model.config.hidden_size) for i in range(len(model.model.layers))}
# Ks_prev = {i: torch.zeros(model.config.hidden_size) for i in range(len(model.model.layers))}
# wrap_llama_with_bqer(model, Ks_cur, Ks_prev, window_m=1)
# # At generation time, remember to reset y-cache between prompts:
# for layer in model.model.layers:
#     if isinstance(layer, BQERDecoderLayer):
#         layer.reset_cache()