#!/bin/bash

#SBATCH --ppartition=standard-g
rem #SBATCH --partition=eap
#SBATCH --account=project_462000183
#SBATCH --time=0:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=10G

rem #SBATCH --array=0-7

rem case $SLURM_ARRAY_TASK_ID in
rem     0) ENV="acrobot_swingup";;
rem     1) ENV="cheetah_run" ;;
rem     2) ENV="fish_swim";;
rem     3) ENV="dog_run" ;;
rem     4) ENV="quadruped_walk" ;;
rem     5) ENV="walker_walk" ;;
rem     6) ENV="humanoid_walk" ;;
rem     7) ENV="dog_walk" ;;
rem esac

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK #
export MPICH_GPU_SUPPORT_ENABLED=1 #


module load LUMI/22.08
module load lumi-container-wrapper


MINLPS_CONTAINER_DIR=~/minlps
export PATH="$NN2SVGP_CONTAINER_DIR/bin:$PATH"

python rl/cluster_train.py +experiment=nn2svgp-sample
rem echo Starting lumi_test.py
rem date
rem python lumi_test.py
rem echo Finished lumi_test.py
rem date

rem echo Starting lumi_test_render.py
rem date

rem export MUJOCO_GL=osmesa
rem python lumi_test_render.py

rem echo Finished lumi_test_render.py
rem date

rem echo "Starting lumi_test.py (osmesa)"
rem date

rem python lumi_test.py

rem echo "Finished lumi_test.py (osmesa)"
rem date
