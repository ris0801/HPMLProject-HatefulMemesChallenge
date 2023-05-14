# run for unsupervised learning
# python main.py --experiment "lossv0-rn18-distilbert-fhb-mmhs" --epochs 100 --wd 0 --lossv0 -b 64

# run for supervised learning with added dense layer

MEMOTION_ADDR=/home/khizirs/contr/src/memotion_dataset_7k

cp "$MEMOTION_ADDR/train_600.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-600samples-incremental-resume" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 14 --task c --cl-ckpt "runs/convirt-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-600samples-incremental/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std--c--classifier--2x512--bn--selfattend--tanh--14.pth.tar"
#----python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-600samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 4 --task b
#----python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-rn18-distilbert-fbh-mmhs--memotion-600samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 3


cp "$MEMOTION_ADDR/train_60.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-60samples-incremental-resume" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 14 --task c --cl-ckpt "runs/convirt-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-60samples-incremental/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std--c--classifier--2x512--bn--selfattend--tanh--14.pth.tar" 
#----python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-60samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 4 --task b
#----python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-rn18-distilbert-fbh-mmhs--memotion-60samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 3


cp "$MEMOTION_ADDR/train_1200.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-1200samples-incremental-resume" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 14 --task c  --cl-ckpt "runs/convirt-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-1200samples-incremental/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std--c--classifier--2x512--bn--selfattend--tanh--14.pth.tar"

#----python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-1200samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 4 --task b
#----python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-rn18-distilbert-fbh-mmhs--memotion-1200samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 3


cp "$MEMOTION_ADDR/train_3000.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-3000samples-incremental-resume" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 14 --task c --cl-ckpt "runs/convirt-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-3000samples-incremental/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std--c--classifier--2x512--bn--selfattend--tanh--14.pth.tar" 

#----python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-3000samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 4 --task b
#----python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-rn18-distilbert-fbh-mmhs--memotion-3000samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --projector std --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 3

# run for visualization of embeddings
# python main.py --ckpt ./runs/Jun12_03-05-38_LS121/checkpoint_0100.pth.tar --vis-embed

# enable to send twilio-sms to contact nunmber in environment

echo "Training Complete at LCS2 Server" | /home/khizirs/twilio-sms $CONTACT_NUMBER 
