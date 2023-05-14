# run for unsupervised learning
# python main.py --experiment "lossv0-rn18-distilbert-fhb-mmhs" --epochs 100 --wd 0 --lossv0 -b 64

# run for supervised learning with added dense layer

python main.py --ckpt "runs/memeloss-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--pie.pth.tar" --supervised --experiment "memeloss-supervised-rn18-distilbert-fbh-mmhs--hateful-80samples-run3" --epochs 100 -b 32 --n-samples 40 --wd 0 --bn --lr 0.0005 --projector pie --fuse_type pie --no-tqdm
python main.py --ckpt "runs/memeloss-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--pie.pth.tar" --supervised --experiment "memeloss-supervised-rn18-distilbert-fbh-mmhs--hateful-1600samples-run3" --epochs 100 -b 256 --n-samples 800 --wd 0 --bn --lr 0.0005 --projector pie --fuse_type pie --no-tqdm
python main.py --ckpt "runs/memeloss-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--pie.pth.tar" --supervised --experiment "memeloss-supervised-rn18-distilbert-fbh-mmhs--hateful-4000samples-run3" --epochs 100 -b 256 --n-samples 2000 --wd 0 --bn --lr 0.0005 --projector pie --fuse_type pie --no-tqdm

python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-rn18-distilbert-fbh-mmhs--hateful-80samples-run2" --epochs 100 -b 32 --n-samples 40 --wd 0 --bn --lr 0.0005  --no-tqdm
python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-rn18-distilbert-fbh-mmhs--hateful-1600samples-run2" --epochs 100 -b 256 --n-samples 800 --wd 0 --bn --lr 0.0005  --no-tqdm
python main.py --ckpt "runs/convirt-rn18-distilbert-fhb-mmhs/last_checkpoint-resnet18--distilbert-base-uncased--2x512d--0.20p--std.pth.tar" --supervised --experiment "convirt-supervised-rn18-distilbert-fbh-mmhs--hateful-4000samples-run2" --epochs 100 -b 256 --n-samples 2000 --wd 0 --bn --lr 0.0005  --no-tqdm

# run for visualization of embeddings
# python main.py --ckpt ./runs/Jun12_03-05-38_LS121/checkpoint_0100.pth.tar --vis-embed

# enable to send twilio-sms to contact nunmber in environment

echo "Training Complete at LCS2 Server" | /home/khizirs/twilio-sms $CONTACT_NUMBER

