from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torch.utils.data.dataset import Dataset
from torch.utils.data import ConcatDataset
import pandas as pd
from PIL import Image
from transformers import RobertaTokenizer, DistilBertTokenizer
from textaugment import Translate, Wordnet
import numpy as np
import torch
import json
import os

from data_aug.utils import stratified_sample_df

def np_random_sample(arr, size=1):
        return arr[np.random.choice(len(arr), size=size, replace=False)]


class SupervisionHatefulMemesDataset(Dataset):
    def __init__(self, folder_path, phase, txt_model, height=224, width=224, im_transforms=None, num_samples=0, txt_max_length=100):
        jsonpath = folder_path + '/' + phase + '.jsonl'
        data = pd.read_json(jsonpath, lines=True)
        if num_samples != 0:
            data = stratified_sample_df(data, 'label', num_samples) 
        self.data = data
        self.num_samples = len(self.data)
        self.phase = phase
        self.labels = np.asarray(self.data['label'])
        self.height = height
        self.width = width
        self.folder_path = folder_path
        self.txt_model = txt_model
        if txt_model == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained(txt_model, model_max_length=txt_max_length)
        self.encoded_captions = self.tokenizer(list(self.data.text), padding='max_length', truncation=True)
        self.image_transform = im_transforms
        print("Loaded {} Samples in {} Hateful Memes Dataset".format(self.num_samples, phase))
        print(self.data['label'].value_counts())

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        imgpath = self.folder_path + '/' + self.data.img[index]
        img_as_img = Image.open(imgpath).convert('RGB')
        img_views = self.image_transform(img_as_img)

        encoded_caption = {
            key: torch.tensor(values[index])
            for key, values in self.encoded_captions.items()
        }
        text2tokens, att_mask = encoded_caption['input_ids'], encoded_caption['attention_mask']
        return img_views, text2tokens, att_mask, single_image_label

    def __len__(self):
        return self.num_samples 

class MMHSDataset(Dataset):
    def __init__(self, folder_path, phase, txt_model, height=224, width=224, im_transforms=None, num_samples=0, txt_max_length=100):
        jsonpath = folder_path + '/splits/' + phase + '_ids.txt'
        txtpath = folder_path + '/img_txt/{}.json'
        self.image_dir = folder_path + '/img_resized/'

        with open(jsonpath, 'r') as f:
            ids = f.readlines()
        ids = [t.replace('\n', '') for t in ids]
        if num_samples != 0:
            ids = ids[:num_samples]
        self.phase = phase
        self.text = []
        self.ids = []
        for idx in ids:
            try:
                with open(txtpath.format(idx), 'r') as f:
                    d = json.load(f)
                    self.text.append(d['img_text'])
                    self.ids.append(idx)
            except FileNotFoundError:
                pass
        self.num_samples = len(self.ids)
        print("Loaded {} Samples in {} MMHS Dataset".format(self.num_samples, phase))
        self.height = height
        self.width = width
        self.folder_path = folder_path
        self.txt_model = txt_model
        if txt_model == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained(txt_model, model_max_length=txt_max_length)
        self.encoded_captions = self.tokenizer(list(self.text), padding='max_length', truncation=True)
        self.image_transform = im_transforms

    def __getitem__(self, index):
        single_image_label = 0  # Not Available 
        imgpath = self.image_dir + str(self.ids[index]) + '.jpg'
        img_as_img = Image.open(imgpath).convert('RGB')
        img_views = self.image_transform(img_as_img)

        encoded_caption = {
            key: torch.tensor(values[index])
            for key, values in self.encoded_captions.items()
        }
        text2tokens, att_mask = encoded_caption['input_ids'], encoded_caption['attention_mask']
        return img_views, text2tokens, att_mask, single_image_label

    def __len__(self):
        return self.num_samples

def get_std_image_transform(size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize])
    return data_transforms


def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomApply([color_jitter], p=0.8),
                                         transforms.RandomGrayscale(p=0.2),
                                         GaussianBlur(kernel_size=int(0.1 * size)),
                                         transforms.ToTensor(),
                                         normalize])
    return data_transforms


def get_unsupervision_dataset(args):
    num_samples, txt_model = args.n_samples, args.txtmodel
    imtransforms = get_std_image_transform(224) 
    if args.simclr:
        imtransforms = get_simclr_pipeline_transform(224)
    if args.n_views > 1:
        imtransforms = ContrastiveLearningViewGenerator(imtransforms, args.n_views)
    # BEWARE: HARDCODED PATHS FOR MEMES
    path = '/scratch/bka2022/pytorch-example/hateful_memes_data'
    dataset1 = SupervisionHatefulMemesDataset(path, 'train', txt_model, im_transforms=imtransforms, num_samples=num_samples)
    path = '/scratch/bka2022/pytorch-example/hate_speech'
    dataset2 = MMHSDataset(path, 'train', txt_model, im_transforms=imtransforms, num_samples=num_samples)
    return ConcatDataset([dataset1, dataset2])


def get_supervision_dataset_hateful(args):
    path, txt_model = args.data, args.txtmodel
    num_samples = args.n_samples
    train_transform = get_simclr_pipeline_transform(224)
    val_transform = transforms.Compose([transforms.CenterCrop(size=224),
                                        transforms.ToTensor()])
    traindataset = SupervisionHatefulMemesDataset(
            path, phase='train', txt_model=txt_model, im_transforms=train_transform, num_samples=num_samples)
    valdataset = SupervisionHatefulMemesDataset(path, phase='dev', txt_model=txt_model, im_transforms=val_transform)
    return traindataset, valdataset
