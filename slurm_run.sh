#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
##SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40:1
#SBATCH --nodes=1
#SBATCH --array=0-3
#SBATCH --partition=batch
#SBATCH --qos=normal
##SBATCH -w=kd-a40-0.grasp.maas
#SBATCH --time=4:00:00
##SBATCH --exclude=kd-2080ti-1.grasp.maas, kd-2080ti-2.grasp.maas, kd-2080ti-3.grasp.maas, kd-2080ti-4.grasp.maas
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./output/cluster/%x-%j.out

hostname
# echo SLURM_NTASKS: $SLURM_NTASKS
source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate active
cd ~/ActiveNeRF
pwd

# DATADIR=/mnt/kostas-graid/datasets/boshu_nerf/nerf_synthetic
# OBJS=("lego"  "mic" "drums" "materials") # "ficus" "hotdog"   "materials" "chair"  "drums" )

DATADIR=/mnt/kostas-graid/datasets/360_v2
OBJS=("treehill" "flowers"  "garden" "room") # "treehill" "bicycle" "counter"  "kitchen" "stump")

# DATADIR=/mnt/kostas-graid/datasets/boshu_nerf/LF
# OBJS=("africa"  "basket"  "statue"  "torch")

OBJ=${OBJS[${SLURM_ARRAY_TASK_ID}]}
# OBJ="ship"

SEED=123

echo ${DATADIR}/${OBJ}
srun python run_nerf.py --config configs/llff_active.txt \
        --expname ${OBJ}_s4_fvs --datadir ${DATADIR}/${OBJ}