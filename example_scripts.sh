export CUDA_VISIBLE_DEVICES=$1


# RUN NeRF Synthetic
# OBJS=("lego" "ship" "chair"  "drums"  "ficus"  "hotdog"  "materials"  "mic")
# for OBJ in ${OBJS[@]}
# do

# python run_nerf.py --config configs/hotdog_active.txt \
#         --expname ${OBJ}_s4 --datadir /root/Dataset/nerf_synthetic/${OBJ}

# done

# DATADIR=/mnt/kostas-graid/datasets/boshu_nerf/nerf_synthetic
# OBJ=lego
# python run_nerf.py --config configs/hotdog_active.txt \
#         --expname ${OBJ}_s4_test --datadir ${DATADIR}/${OBJ}

DATADIR=/mnt/kostas-graid/datasets/boshu_nerf/360_v2
OBJ=bonsai
python run_nerf.py --config configs/llff_active.txt \
        --expname ${OBJ}_s4_fvs --datadir ${DATADIR}/${OBJ}
