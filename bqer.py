# bqer.py
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict, List


# --------------------- Layer definition --------------------------------------

class BQERDecoderLayer(nn.Module):
    """
    TCEC-like residual correction.
    """
    def __init__(
        self,
        inner_layer: nn.Module,
        hidden_size: int,
        K_current: Optional[torch.Tensor] = None,   # (G,)
        K_prev: Optional[torch.Tensor] = None,      # (G,) for window=2
        window_m: int = 0,
        alpha: float = 0.75,
        place: str = "post",
        group_size: int = 512,                        # >=1; 1 == channel-wise
    ):
        super().__init__()
        assert place in ("post",)
        assert window_m in (0, 1)
        assert hidden_size % group_size == 0, f"hidden_size {hidden_size} must be divisible by group_size {group_size}"

        self.inner = inner_layer
        self.hidden_size = hidden_size
        self.window_m = int(window_m)   # 0: window=1, 1: window=2
        self.alpha = float(alpha)
        self.place = place
        self.group_size = int(group_size)

        C = hidden_size
        G = C // self.group_size

        # K_l (current layer)
        Kg = torch.zeros(G, dtype=torch.float32)
        if K_current is not None:
            assert K_current.dim() == 1 and K_current.numel() == G, f"Expected K_current len={G}"
            Kg = K_current.detach().to(torch.float32)
        self.register_buffer("K_groups", Kg, persistent=True)  # (G,)

        # K_{l-1} (previous layer) for window=2
        Kpg = torch.zeros(G, dtype=torch.float32)
        if K_prev is not None:
            assert K_prev.dim() == 1 and K_prev.numel() == G, f"Expected K_prev len={G}"
            Kpg = K_prev.detach().to(torch.float32)
        self.register_buffer("K_prev_groups", Kpg, persistent=True)  # (G,)

        # runtime에서 이전 레이어의 y_l-1를 가져오기 위한 포인터/캐시
        self.prev_layer: Optional["BQERDecoderLayer"] = None
        self.register_buffer("_last_y", None, persistent=False)

    def reset_cache(self):
        self._last_y = None

    def _expanded_K(self, like: torch.Tensor) -> torch.Tensor:
        C = self.hidden_size
        k = torch.repeat_interleave(self.K_groups, self.group_size)[:C]  # (C,)
        return k.view(1, 1, -1).to(device=like.device, dtype=like.dtype)

    def _expanded_K_prev(self, like: torch.Tensor) -> torch.Tensor:
        C = self.hidden_size
        kp = torch.repeat_interleave(self.K_prev_groups, self.group_size)[:C]  # (C,)
        return kp.view(1, 1, -1).to(device=like.device, dtype=like.dtype)

    @torch.no_grad()
    def _compute_delta(self, y_cur: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0.0:
            return torch.zeros_like(y_cur)

        K = self._expanded_K(y_cur)
        delta = K * y_cur

        if self.window_m == 0:
            return self.alpha * delta

        if self.prev_layer is not None and getattr(self.prev_layer, "_last_y", None) is not None:
            y_prev = self.prev_layer._last_y.to(device=y_cur.device, dtype=y_cur.dtype)
            Kp = self._expanded_K_prev(y_cur)
            delta = delta + Kp * y_prev

        return self.alpha * delta

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        x_in = hidden_states
        out = self.inner(hidden_states, *args, **kwargs)
        x_out, tail = (out, ()) if isinstance(out, torch.Tensor) else (out[0], out[1:])

        y_cur = x_out - x_in

        x_out = x_out + self._compute_delta(y_cur)

        self._last_y = y_cur.detach()

        return x_out if isinstance(out, torch.Tensor) else (x_out, *tail)

# --------------------- Accumulate stats and calibrate Ks --------------------------------------

@torch.no_grad()
def accumulate_stats(fp_model: nn.Module,
                                 q_model: nn.Module,
                                 dataloader,
                                 device: str = "cuda"):
    # resolve L, C and layer lists
    if hasattr(q_model, "model") and hasattr(q_model.model, "layers"):
        L = len(q_model.model.layers)
        C = q_model.config.hidden_size
        q_layers = q_model.model.layers
        fp_layers = fp_model.model.layers
    else:
        L = len(q_model.layers)
        C = q_model.config.hidden_size
        q_layers = q_model.layers
        fp_layers = fp_model.layers

    # per-layer CPU accumulators (fp32)
    yq2  = {l: torch.zeros(C, dtype=torch.float32) for l in range(L)}  # Σ y_l^Q^2
    yqyf = {l: torch.zeros(C, dtype=torch.float32) for l in range(L)}  # Σ y_l^Q y_l^FP
    yqyq = {l: torch.zeros(C, dtype=torch.float32) for l in range(L)}  # Σ y_l^Q y_{l-1}^Q (l>=1만 사용)

    q_model.eval().to(device)
    fp_model.eval().to(device)

    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Streaming stats"):
        batch = {k: v.to(device) for k, v in batch.items()}

        # ---- pass 1: Q model -> yq2 and cache yq (compact bf16) ----
        yq_bt = [None] * L

        q_handles = []
        def q_hook_factory(i):
            def hook(mod, inputs, outputs):
                x_in  = inputs[0]
                x_out = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                # y_l^Q = x_out - x_in
                yq = (x_out - x_in).reshape(-1, C)           # (B*T, C) model dtype
                yq32 = yq.to(torch.float32)
                yq2[i].add_((yq32 * yq32).sum(dim=0).to("cpu"))  # Σ y_l^Q^2
                yq_bt[i] = yq.to(torch.bfloat16)             # compact GPU cache for cross terms
                return outputs
            return hook

        for i, layer in enumerate(q_layers):
            q_handles.append(layer.register_forward_hook(q_hook_factory(i)))
        _ = q_model(**batch)
        for h in q_handles: 
            h.remove()

        # compute Σ y_l^Q y_{l-1}^Q using cached yq_bt
        # yqyq[l] = Σ_{tokens} (y_l^Q * y_{l-1}^Q)
        for l in range(1, L):
            if yq_bt[l] is None or yq_bt[l-1] is None:
                continue
            y_cur  = yq_bt[l].to(torch.float32)      # (B*T, C)
            y_prev = yq_bt[l-1].to(torch.float32)    # (B*T, C)
            yqyq[l].add_((y_cur * y_prev).sum(dim=0).to("cpu"))

        # ---- pass 2: FP model -> yqyf using cached yq ----
        fp_handles = []
        def fp_hook_factory(i):
            def hook(mod, inputs, outputs):
                x_in  = inputs[0]
                x_out = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                yf32 = (x_out - x_in).reshape(-1, C).to(torch.float32)  # (B*T,C) = y_l^FP
                if yq_bt[i] is not None:
                    yq32 = yq_bt[i].to(torch.float32)
                    # Σ y_l^Q y_l^FP
                    yqyf[i].add_((yq32 * yf32).sum(dim=0).to("cpu"))
                return outputs
            return hook

        for i, layer in enumerate(fp_layers):
            fp_handles.append(layer.register_forward_hook(fp_hook_factory(i)))
        _ = fp_model(**batch)
        for h in fp_handles: 
            h.remove()

        del yq_bt
        torch.cuda.empty_cache()

    return yq2, yqyf, yqyq

@torch.no_grad()
def calibrate_bqer(fp_model: nn.Module,
                   q_model: nn.Module,
                   dataloader,
                   device: str = "cuda",
                   window_m: int = 0,
                   lambda1: float = 1e-5,
                   clip: float = 0.25,
                   group_size: int = 512):
    assert group_size >= 1
    assert window_m in (0, 1)
    yq2, yqyf, yqyq = accumulate_stats(fp_model, q_model, dataloader, device=device)

    if hasattr(q_model, "model") and hasattr(q_model.model, "layers"):
        L = len(q_model.model.layers)
        C = q_model.config.hidden_size
    else:
        L = len(q_model.layers)
        C = q_model.config.hidden_size

    g = int(group_size)
    G = C // g

    Ks_cur, Ks_prev = {}, {}

    for l in range(L):
        sqq = yq2[l]   # (C,) on CPU  = Σ ||y_l^Q||^2 (per channel)
        sfq = yqyf[l]  # (C,) on CPU  = Σ <y_l^Q, y_l^FP>

        # group-wise sum
        A_g = sqq.view(G, g).sum(dim=1)   # A = Σ ||f_l^Q||^2 per group
        D_g = sfq.view(G, g).sum(dim=1)   # D = Σ <f_l^Q, f_l^FP> per group

        if window_m == 0 or l == 0:
            # ----- window = 1 (기존) 또는 l=0 (이전 레이어 없음) -----
            # K_l = (D - A) / (A + λ1)
            K = (D_g - A_g) / (A_g + lambda1)
        else:
            C_g = yqyq[l].view(G, g).sum(dim=1)  # (G,)
            K_prev = Ks_cur[l - 1]  # (G,)
            K = (A_g - C_g * K_prev + D_g) / (A_g + lambda1)

        if clip is not None:
            K = K.clamp(-clip, clip)

        Ks_cur[l]  = K.to(torch.float32)
        Ks_prev[l] = Ks_cur[l - 1] if l > 0 else K.clone()

    return Ks_cur, Ks_prev

# --------------------- Apply BQER to model --------------------------------------

def wrap_model_with_bqer(model: nn.Module,
                         Ks_current: Dict[int, torch.Tensor],  # (G,)
                         Ks_prev: Optional[Dict[int, torch.Tensor]] = None,
                         window_m: int = 0,
                         alpha: float = 0.75,
                         group_size: int = 512):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        hidden_size = model.config.hidden_size
    else:
        layers = model.layers
        hidden_size = model.config.hidden_size

    prev_wrapper: Optional[BQERDecoderLayer] = None

    for idx, layer in enumerate(layers):
        Kc = Ks_current.get(idx, None)   # (G,)
        if Kc is None:
            prev_wrapper = None
            continue

        Kp = None
        if Ks_prev is not None:
            Kp = Ks_prev.get(idx, None)

        wrapper = BQERDecoderLayer(
            inner_layer=layer,
            hidden_size=hidden_size,
            K_current=Kc,        # expected length G
            K_prev=Kp,           # for window=2 term
            window_m=window_m,   # 0: only K_l*y_l, 1: K_l*y_l + K_{l-1}*y_{l-1}
            alpha=alpha,
            place="post",
            group_size=group_size,
        )

        wrapper.prev_layer = prev_wrapper
        layers[idx] = wrapper
        prev_wrapper = wrapper

def apply_bqer(
    q_model: nn.Module,
    Ks_cur: Dict[int, torch.Tensor],
    Ks_prev: Dict[int, torch.Tensor],
    window_m: int = 0,
    alpha: float = 0.75,
    group_size: int = 512,
):
    wrap_model_with_bqer(
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

