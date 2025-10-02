#!/bin/bash

#SBATCH --partition=gpu        # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=1                # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)

#SBATCH --gres=gpu:2

#SBATCH --mem=128000                # Real memory (RAM) required (MB)

#SBATCH --time=3-00:00:00           # Total run time limit (D-HH:MM:SS)

#SBATCH --output=slurm_chip/generic/%j.out  # STDOUT output file

#SBATCH --error=slurm_chip/generic/%j.err   # STDERR output file (optional)

#SBATCH --gres-flags=enforce-binding

day=$(date +'%m/%d/%Y %R')
echo "gpu"  ${day} $SLURM_JOBID "node_list" $SLURM_NODELIST $@  "\n" >> jobs.txt
module purge
module load slurm/chip-gpu/23.11.4
#module load cudnn/7.0.3
module load   Autoconf/2.72-GCCcore-13.3.0 
module load  CMake/3.29.3-GCCcore-13.3.0
module load  CUDA/12.8.0  

gcc --version
source myenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export TORCH_USE_CUDA_DSA="1"
export CUDA_LAUNCH_BLOCKING="1"
export TRANSFORMERS_CACHE="/umbc/ada/donengel/common/trans_cache"
export HF_HOME="/umbc/ada/donengel/common/trans_cache"
export HF_HUB_CACHE="/umbc/ada/donengel/common/trans_cache"
export TORCH_CACHE="/umbc/ada/donengel/common/torch_hub_cache"
export TORCH_HOME="/umbc/ada/donengel/common/torch_home"
export WANDB_DIR="/umbc/ada/donengel/common/wandb"
export WANDB_CACHE_DIR="/umbc/ada/donengel/common/wandb_cache"
export HPS_ROOT="/umbc/ada/donengel/common/hps-cache"
export IMAGE_REWARD_PATH="/umbc/ada/donengel/common/reward-blob"
export IMAGE_REWARD_CONFIG="/umbc/ada/donengel/common/ImageReward/med_config.json"
export BRAIN_DATA_DIR='/umbc/ada/donengel/common/brain-diffuser/data'
export CUDA_LAUNCH_BLOCKING="1"
export SCIKIT_LEARN_DATA="/umbc/ada/donengel/common/scikit-learn-data"
export BRAIN_DATA_DIR="/umbc/ada/donengel/common/brain/data"
export TORCH_LOCAL_DIR="/umbc/ada/donengel/common/local_torch"
export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export KAGGLEHUB_CACHE="/umbc/ada/donengel/common/kaggle_cache"
export KAGGLE_CONFIG_DIR="/umbc/ada/donengel/common/kaggle_config"
#export NCCL_DEBUG="INFO"
#export NCCL_DEBUG_SUBSYS="ALL"
echo "Running on: $(hostname)"
echo "Allocated GPUs:"
nvidia-smi
echo "version"
nvcc --version
srun accelerate launch   $@
echo "Running on: $(hostname)"
echo "Allocated GPUs:"
nvidia-smi