import argparse
import torch
import torch.backends.cudnn as cudnn
from data_aug.contrastive_learning_dataset import get_supervision_dataset_hateful, get_unsupervision_dataset, get_supervision_dataset_harmeme, get_supervision_dataset_memotion
from models.unsupervised import UnsupervisedModel 
from models.classifier import MultiModalClassifier
from train_unsupervised import UnsupervisedLearner
from classification import SupervisedLearner

parser = argparse.ArgumentParser(description='PyTorch Multimodal Contrastive Learning with SimCLR')

parser.add_argument('--experiment', default='', type=str,
                     help="Optional Name of Experiment (used by tensorboard)")
parser.add_argument('--no-tqdm', action='store_true', help="Disable tqdm and not pollute nohup out")

# ================================= DataSet =====================================

parser.add_argument('-data', metavar='DIR', default='/home/khizirs/contr/hatefulmemes/dataset/data',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='hatefulmemes',
                    help='dataset name', choices=['hatefulmemes', 'harmeme', 'memotion'])
parser.add_argument('--task', default='a', help='task a or b for memotion')


# ================================= ARCHITECTURE ================================

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='Image Model Architecture: ')
parser.add_argument('--txtmodel', default='distilbert-base-uncased', type=str,
                    choices=['distilbert-base-uncased', 'roberta'],
                    help="Text Model used for encoding text")
parser.add_argument('--out-dim', default=512, type=int,
                    help='Embedding dimension for modalities (default: 512)')
parser.add_argument('--projector', default='std', type=str,
                    choices=['std', 'clip', 'pie'], help="Projection used for Unsupervised Training")
parser.add_argument('--ckpt', default='', type=str,
                    help='Path to load for checkpoint')
parser.add_argument('--dropout', default=0.2, type=float,
                    help="Dropout probability in classification layer of model")
parser.add_argument('--fuse_type', default='selfattend', type=str,
                    choices=['selfattend', 'concat', 'pie', 'mlb'],
                    help="How to combine embeddings in supervised learning")
parser.add_argument('--nl', default='tanh', type=str,
                    choices=['tanh', 'relu', 'sigmoid'],
                    help="Non Linearity to use between layers of projection heads")
parser.add_argument('--cl-ckpt', default='', type=str,
                    help="Resume classifier from this checkpoint location")
parser.add_argument('--num-classes', default=2, type=int,
                    help="Number of Classes in Supervised Setting")
parser.add_argument('--bn', action='store_true', default=False,
                    help="Use Batch Norm in Classifier")

# ================================= TRAINING ================================

parser.add_argument('--dryrun', action='store_true', default=False,
                    help='Use for initial testing purposes only. Runs train for 4 iterations')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--gpu-index', default=0, type=int,
                    help='Gpu index.')

parser.add_argument('--supervised', action='store_true', default=False,
                    help='Train Supervised')
parser.add_argument('--n-samples', default=0, type=int,
                    help="Number of samples used in training")

parser.add_argument('--vis-embed', action='store_true', default=False,
                    help='Visualise Embeddings in Tensorboard, can not be used with supervised learning.')

# ================================= EVALUATE ===================================

parser.add_argument('--evaluate_only', action='store_true', default=False,
                    help="Only evaluate the given model at checkpoints")


# ================================= LOSSES =====================================

# SimCLR
parser.add_argument('--simclr', action='store_true', help="Use SimCLR for Unsupervised Training on Image Views")
parser.add_argument('--n-views', default=1, type=int, metavar='N',
                    help='Number of views for contrastive learning training. 1 means no views generated Setting')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--moco_size', default=0, type=int,
                    help="Size of Memory Bank (MoCo, 2020), size=0 is set for SimCLR")

# Supervised Contrastive Learning
parser.add_argument('--supcontr', action='store_true', help="Use SupContr for Unsupervised Training")

# Multimodal Contrastive Learning
parser.add_argument('--mmcontr', action='store_true', help="Use MMContrLoss for Unsupervised Training")
parser.add_argument('--measure', default='cosine', type=str,
                    choices=['cosine', 'order'], help="Similarity measure to be used in MMContrLoss")
parser.add_argument('--margin', default=0, type=float,
                     help="Margin to be used in MMContrLoss")
parser.add_argument('--max_violation', action='store_true', default=False,
                     help="Consider only the max violation in MMContrLoss")

# CLIP
parser.add_argument('--cliploss', action='store_true', help="Use CLIP Loss for Unsupervised Training")


# ConVIRT
parser.add_argument('--convirt', action='store_true', help="Use ConVIRT Loss for Unsupervised Training")


# MemeMultimodal Loss
parser.add_argument('--memeloss', action='store_true', help="Use Meme Multimodal Loss for Unsupervised Training")
parser.add_argument('--w-f2i', type=float, default=0.2, help="Fuse2Image Loss Weight")
parser.add_argument('--w-f2t', type=float, default=0.2, help="Fuse2Text Loss Weight")
parser.add_argument('--w-f2f', type=float, default=0.6, help="Fuse2Fuse Loss Weight")


# LossV0 Loss
parser.add_argument('--lossv0', action='store_true', help="Use LossV0 for unsupervised training")


def main():
    args = parser.parse_args()
    if args.dryrun:
        args.experiment = 'dryrun'

    assert (not args.simclr) or (args.n_views > 1), "SimCLR requires at least 2 image views"

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    print(args)

    ckpt_use = args.ckpt != ''
    model = UnsupervisedModel(args.arch, args.txtmodel, args.out_dim, args.dropout, args.projector, not ckpt_use, not ckpt_use)
    model.to(args.device)
    print(f"Unsupervised Model Name: {model.name}")

    if ckpt_use:
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device)['state_dict'])
        print(f"Model Loaded from {args.ckpt}")

    if not args.supervised:
        train_dataset = get_unsupervision_dataset(args)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Parameters: ", pytorch_total_params)
        with torch.cuda.device(args.gpu_index):
            simclr = UnsupervisedLearner(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            if args.vis_embed:
                simclr.save_embed(train_loader)
            else:
                simclr.train(train_loader)
    else:
        if args.dataset_name == 'hatefulmemes':
            train_dataset, val_dataset = get_supervision_dataset_hateful(args)
        elif args.dataset_name == 'harmeme':
            train_dataset, val_dataset = get_supervision_dataset_harmeme(args)
        elif args.dataset_name == 'memotion':
            train_dataset, val_dataset = get_supervision_dataset_memotion(args)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)

        classifier = MultiModalClassifier(
                fuse_type=args.fuse_type,
                model=model,
                num_classes=args.num_classes,
                nl=args.nl,
                bn=args.bn
        ).to(args.device)
        print(f"Classification Head Name: {classifier.name}")

        if args.cl_ckpt != '':
            classifier.load_state_dict(torch.load(args.cl_ckpt, map_location=args.device)['state_dict'])
            print(f"Classifier Loaded from {args.cl_ckpt}")

        optimizer = torch.optim.Adam(classifier.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 1e-3,
                steps_per_epoch=len(train_loader),
                epochs=args.epochs
        )
        pytorch_total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        print("Total Parameters: ", pytorch_total_params)

        with torch.cuda.device(args.gpu_index):
            simclr = SupervisedLearner(model=model, optimizer=optimizer, scheduler=scheduler, classifier=classifier, args=args)
            if args.evaluate_only:
                simclr.evaluate(val_loader)
            else:
                simclr.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
