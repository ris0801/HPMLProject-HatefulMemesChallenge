# HPMLProject-HatefulMemesChallenge
Utilizing Domain-aware Self-supervised Pre-training for Label-Efficient Meme Analysis.

- [Introduction](#introduction)
- [Project Description](#project-description)
- [Repository Overview](#repository-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Training](#training)
- [Datasets](#datasets)
- [Results](#results)

## Introduction
This project is based on the self-supervised pre-training method MM-SimCLR introduced in the paper "Domain-aware Self-supervised Pre-training for Label-Efficient Meme Analysis" (arXiv:2209.14667). 

## Project Description
The aim of this project is to design and evaluate efficient multimodal frameworks for meme analysis, which do not depend on large-scale dataset curation and annotation. The frameworks can be pre-trained using datasets obtained from various sources.

## Repository Overview
This repository contains all the code related to the project. To get started, create a Conda environment using the packages listed in `requirements.txt`.

## Getting Started

### Prerequisites
Before running the code, make sure you have Conda installed on your machine. If not, you can install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Training:
Unsupervised: Please run the following script file making appropriate file path and environment changes

**Train_Unsupervised.sh**

Supervised: Please run the following script file making appropriate file path and environment changes

**Train_Supervised.sh**

the output will be available in the "runs" folder with the training log and checkpoints in the corresponding "experiment_name" folder 

### Evaluation:
Please run the following commands making appropriate file path for data and checkpoints obtained after training and environment changes
With Profiling:
```
python3 "/scratch/bka2022/pytorch-example/multimodal-MEMES-master/main.py" --experiment "Eval_Profile" --ckpt "/scratch/bka2022/pytorch-example/multimodal-MEMES-master/runs/spervised_2_MMcontr_1_GPU_2/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std--c--classifier--2x512--bn--selfattend--tanh--2.pth.tar" -data "/scratch/bka2022/pytorch-example/hateful_memes_data" -b 256 --supervised --mmcontr --evaluate_only --bn --profile
```

Without profiling:
```
python3 "/scratch/bka2022/pytorch-example/multimodal-MEMES-master/main.py" --experiment "Eval_Profile" --ckpt "/scratch/bka2022/pytorch-example/multimodal-MEMES-master/runs/spervised_2_MMcontr_1_GPU_2/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std--c--classifier--2x512--bn--selfattend--tanh--2.pth.tar" -data "/scratch/bka2022/pytorch-example/hateful_memes_data" -b 256 --supervised --mmcontr --evaluate_only --bn
```

## Datasets:
- Hateful Memes: https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset
- Multimodal Hate Speech: https://www.kaggle.com/datasets/victorcallejasf/multimodal-hate-speech

## Results: 
![image](https://github.com/ris0801/HPMLProject-HatefulMemesChallenge/assets/131811678/59b5b783-4779-4d32-8302-55246d3cceae)

![image](https://github.com/ris0801/HPMLProject-HatefulMemesChallenge/assets/131811678/bc5519de-551e-49d4-9c60-a073a490f22b)


