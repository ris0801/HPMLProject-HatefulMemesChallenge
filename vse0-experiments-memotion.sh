MEMOTION_ADDR=/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k

cp "$MEMOTION_ADDR/train_600.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-600samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 14 --task c 
python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-600samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 4 --task b 
python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-a-rn18-distilbert-fbh-mmhs--memotion-600samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 3 

cp "$MEMOTION_ADDR/train_60.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-60samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 14 --task c  
python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-60samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 4 --task b 
python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-a-rn18-distilbert-fbh-mmhs--memotion-60samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 3 

cp "$MEMOTION_ADDR/train_1200.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-1200samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 14 --task c  
python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-1200samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 4 --task b 
python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-a-rn18-distilbert-fbh-mmhs--memotion-1200samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 3 

cp "$MEMOTION_ADDR/train_3000.csv" "$MEMOTION_ADDR/labels.csv"

python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-c-rn18-distilbert-fbh-mmhs--memotion-3000samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 14 --task c  
python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-b-rn18-distilbert-fbh-mmhs--memotion-3000samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 4 --task b 
python main.py --ckpt "PATH_TO_VSE0_WEIGTHS" --supervised --experiment "vse0-supervised-task-a-rn18-distilbert-fbh-mmhs--memotion-3000samples-incremental" --epochs 100 -b 256 --wd 0 --bn --lr 0.0005 --no-tqdm -dataset-name 'memotion' -data '/home/USERNAME/EXTRA_PATH_HERE/memotion_dataset_7k' --num-classes 3 
