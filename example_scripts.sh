export CUDA_VISIBLE_DEVICES=0


# RUN NeRF Synthetic
# OBJS=("lego" "ship" "chair"  "drums"  "ficus"  "hotdog"  "materials"  "mic")
# for OBJ in ${OBJS[@]}
# do

# python run_nerf.py --config configs/hotdog_active.txt \
#         -t ckpt/${OBJ}_hessian_s1_K10 --active_method hessian_diagonal \
#         --expname ckpt/${OBJ} --datadir /root/Dataset/nerf_synthetic/${OBJ}

# done

python -u run_nerf.py --config configs/hotdog_active.txt \
        --expname ckpt/test --datadir /root/Dataset/nerf_synthetic/lego > nerf.txt