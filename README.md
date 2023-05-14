# HPMLProject-HatefulMemesChallenge

Based on arXiv:2209.14667 (Domain-aware Self-supervised Pre-training for Label-Efficient Meme Analysis). 


Description of project: 
We are specifically working on self supervised pre-training method MM-SimCLR introduced in above paper. This paper presents the design and evaluation of efficient multimodal frameworks that do not rely upon large scale dataset curation and annotation and can be pretrained using the datasets from the wild.

Description of repository:

Commands to run the code: 

Results: 

Proposal: It highlights the requirement of better multi-modal self-supervision meth- ods involving specialized pretext tasks for efficient finetuning and generalizable performance.

What's new: 
This paper presents the design and evaluation of efficient multimodal frameworks that do not rely upon large scale dataset curation and annotation and can be pretrained using the datasets from the wild.

Dataset:
Pre-training: MMHS150K + Hateful Memes dataset
Fine tuning and Evaluation: Hateful Memes dataset

Fine-tuning: 
To evaluate the representations learned through pre-training, we employ the linear evaluation strategy (Oord et al., 2018), which trains a linear classi- fier with frozen base network parameters. This is a popular strategy for assessing the quality of the representations learned with a minimal predictive modeling setup that facilitates a fair assessment of the resulting inductive bias.

Although the performances of the proposed models fall behind that of their fullysupervised counterparts, they perform reasonably better than the strong self-supervised methods.

Result in paper: 
Whereas, MM-SimCLR is observed to show stable, yet non-incremental growth in performance reporting the best overall F1 score of 0.3318 (c.f. Table 2).
