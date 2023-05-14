HARMEME_ADDR=/home/khizirs/uspol_harmeme_data/datasets/memes/defaults/annotations

cp "$HARMEME_ADDR/train_1.jsonl" "$HARMEME_ADDR/train.jsonl"

python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-rn18-distilbert-fbh-mmhs--harmeme-1samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2

# python main.py --ckpt "runs/memeloss-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--pie.pth.tar" --supervised --experiment "memeloss-supervised-rn18-distilbert-fbh-mmhs--harmeme-1samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --projector pie --fuse_type pie --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2

# python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "covirt-supervised-rn18-distilbert-fbh-mmhs--harmeme-1samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2

cp "$HARMEME_ADDR/train_10.jsonl" "$HARMEME_ADDR/train.jsonl"

python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-rn18-distilbert-fbh-mmhs--harmeme-10samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2

# python main.py --ckpt "runs/memeloss-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--pie.pth.tar" --supervised --experiment "memeloss-supervised-rn18-distilbert-fbh-mmhs--harmeme-10samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --projector pie --fuse_type pie --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2

# python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "covirt-supervised-rn18-distilbert-fbh-mmhs--harmeme-10samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2

cp "$HARMEME_ADDR/train_20.jsonl" "$HARMEME_ADDR/train.jsonl"

python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-rn18-distilbert-fbh-mmhs--harmeme-20samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2

# python main.py --ckpt "runs/memeloss-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--pie.pth.tar" --supervised --experiment "memeloss-supervised-rn18-distilbert-fbh-mmhs--harmeme-20samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --projector pie --fuse_type pie --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2

# python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "covirt-supervised-rn18-distilbert-fbh-mmhs--harmeme-20samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2

cp "$HARMEME_ADDR/train_50.jsonl" "$HARMEME_ADDR/train.jsonl"

python main.py --ckpt "runs/vse++-rn18-distilbert-fbh-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "vse++-supervised-rn18-distilbert-fbh-mmhs--harmeme-50samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2

# python main.py --ckpt "runs/memeloss-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--pie.pth.tar" --supervised --experiment "memeloss-supervised-rn18-distilbert-fbh-mmhs--harmeme-50samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --projector pie --fuse_type pie --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2
# python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "covirt-supervised-rn18-distilbert-fbh-mmhs--harmeme-50samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/khizirs/uspol_harmeme_data/datasets/memes' --num-classes 2

# enable to send twilio-sms to contact nunmber in environment

echo "Training Complete at LCS2 Server" | /home/khizirs/twilio-sms $CONTACT_NUMBER
