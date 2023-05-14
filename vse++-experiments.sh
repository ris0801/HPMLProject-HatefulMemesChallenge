MEMOTION_ADDR=/home/khizirs/contr/src/memotion_dataset_7k

cp "$MEMOTION_ADDR/train_600.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-600samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 14 --task c 
python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-600samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 4 --task b 
python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-a-rn18-distilbert-fbh-mmhs--memotion-600samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 3 

cp "$MEMOTION_ADDR/train_60.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-60samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 14 --task c  
python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-60samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 4 --task b 
python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-a-rn18-distilbert-fbh-mmhs--memotion-60samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 3 

cp "$MEMOTION_ADDR/train_1200.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-1200samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 14 --task c  
python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-1200samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 4 --task b 
python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-a-rn18-distilbert-fbh-mmhs--memotion-1200samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 3 

cp "$MEMOTION_ADDR/train_3000.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-3000samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 14 --task c  
python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-3000samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 4 --task b 
python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-task-a-rn18-distilbert-fbh-mmhs--memotion-3000samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/khizirs/contr/src/memotion_dataset_7k' --num-classes 3 

echo "Training Complete at LCS2 Server" | /home/khizirs/twilio-sms $CONTACT_NUMBER
