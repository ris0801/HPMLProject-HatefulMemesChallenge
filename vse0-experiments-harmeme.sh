HARMEME_ADDR=/home/USERNAME/uspol_harmeme_data/datasets/memes/defaults/annotations

cp "$HARMEME_ADDR/train_1.jsonl" "$HARMEME_ADDR/train.jsonl"

python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-rn18-distilbert-fbh-mmhs--harmeme-1samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/USERNAME/uspol_harmeme_data/datasets/memes' --num-classes 2

cp "$HARMEME_ADDR/train_10.jsonl" "$HARMEME_ADDR/train.jsonl"

python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-rn18-distilbert-fbh-mmhs--harmeme-10samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/USERNAME/uspol_harmeme_data/datasets/memes' --num-classes 2

cp "$HARMEME_ADDR/train_20.jsonl" "$HARMEME_ADDR/train.jsonl"

python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-rn18-distilbert-fbh-mmhs--harmeme-20samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/USERNAME/uspol_harmeme_data/datasets/memes' --num-classes 2

cp "$HARMEME_ADDR/train_50.jsonl" "$HARMEME_ADDR/train.jsonl"

python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-rn18-distilbert-fbh-mmhs--harmeme-50samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.001 --no-tqdm -dataset-name 'harmeme' -data '/home/USERNAME/uspol_harmeme_data/datasets/memes' --num-classes 2

