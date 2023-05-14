#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=supervised

module purge

singularity exec --nv \
	    --overlay /scratch/bka2022/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif\
	    /bin/bash -c 'source /ext3/env.sh; python3 main.py --ckpt "/scratch/bka2022/pytorch-example/multimodal-MEMES-master/runs/MMCONTR_f_1aGPU_32/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --experiment "spervised_2_MMcontr_1_GPU_2" -data "/scratch/bka2022/pytorch-example/hateful_memes_data" --supervised --epochs 100 -b 32 --bn --lr 0.0005 --mmcontr'
