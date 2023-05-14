#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=contr2128

module purge

singularity exec --nv \
	    --overlay /scratch/bka2022/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif\
	    /bin/bash -c 'source /ext3/env.sh; python3 "/scratch/bka2022/pytorch-example/multimodal-MEMES-master/main.py" --experiment "MMCONTR_f2_2aGPU_128" -data "/scratch/bka2022/pytorch-example/hateful_memes_data" --epochs 100 -b 128 --mmcontr'
