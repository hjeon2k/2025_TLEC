QBITS=(3 4)
CLIPS=(0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90)
ALPHAS=(0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90)
GROUP_SIZE=(1 32 64 128 256 512 1024 3072)

for QBIT in ${QBITS[@]}; do
    for CLIP in ${CLIPS[@]}; do
        for ALPHA in ${ALPHAS[@]}; do
            for GROUP_SIZE in ${GROUP_SIZE[@]}; do
                CUDA_VISIBLE_DEVICES=$1 python3 main.py --model meta-llama/Llama-3.1-8B --qbits $QBIT -c $CLIP -a $ALPHA -g $GROUP_SIZE
            done
        done
    done
done
