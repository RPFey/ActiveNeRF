export CUDA_VISIBLE_DEVICES=0


# RUN NeRF Synthetic
OBJS=("lego" "ship" "chair"  "drums"  "ficus"  "hotdog"  "materials"  "mic")
for OBJ in ${OBJS[@]}
do

python run_nerf.py --config configs/hotdog_active.txt \
        --expname ${OBJ}_s4 --datadir /root/Dataset/nerf_synthetic/${OBJ}

done
