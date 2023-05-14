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

class SupervisionHarmemesDataset(Dataset):
    def __init__(self, folder_path, phase, txt_model, height=224, width=224, im_transforms=None, num_samples=0, txt_max_length=100):
        jsonpath = folder_path + '/defaults/annotations/' + phase + '.jsonl'
        self.image_dir = folder_path + '/defaults/images/'
        self.data = pd.read_json(jsonpath, lines=True)
        self.phase = phase

        def label_mapping(x):
            for t in x:
                if 'not' in t:
                    return 0
            return 1
        self.data['target'] = self.data['labels'].apply(label_mapping)
        if num_samples != 0:
            self.data = stratified_sample_df(self.data, 'target', num_samples) 
        self.num_samples = len(self.data)
        self.labels = np.asarray(self.data['target'])
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
        print("Loaded {} samples in {} Harmemes Dataset".format(self.num_samples, phase))
        print(self.data['target'].value_counts())

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        imgpath = self.image_dir + self.data.image[index]
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

class MemotionDataset(Dataset):
    def __init__(self, folder_path, phase, task, txt_model, height=224, width=224, im_transforms=None, num_samples=0, txt_max_length=100):
        df_path = os.path.join(folder_path, 'labels.csv' if phase=='train' else 'test.csv')
        self.img_path = os.path.join(folder_path, 'images' if phase=='train' else 'test')
        data = pd.read_csv(df_path)
        try:
            data.drop(columns=['Unnamed: 0'], inplace=True)
        except:
            pass
        if task == 'A' or task == 'a':
            data['label'] = data['overall_sentiment'].apply(self.get_sentiment_label)
        elif task in ['b', 'B']:
            data['label'] = data.apply(lambda x: self.get_task2_label(x), axis=1)
        else:
            data['label'] = data.apply(lambda x: self.get_task3_label(x), axis=1)

        self.task = task
        if num_samples != 0 and phase =='train':
            if task in ['a', 'A']:
                data = stratified_sample_df(data, 'label', num_samples)
            else:
                data = data.sample(n=num_samples)
        if 'corrected_text' in data:
            data['text_corrected'] = data['corrected_text']
        if 'Image_name' in data:
            data['image_name'] = data['Image_name']
        self.df = data
        self.df['text_corrected'].fillna('None', inplace=True)
        self.num_samples = len(self.df)
        print("Loaded {} Samples in {} Memotion Dataset".format(self.num_samples, phase))
        self.height = height
        self.width = width
        self.phase = phase
        self.folder_path = folder_path
        self.txt_model = txt_model
        if txt_model == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained(txt_model, model_max_length=txt_max_length)
        self.encoded_captions = self.tokenizer(list(self.df['text_corrected']), padding='max_length', truncation=True)
        self.image_transform = im_transforms

    def get_task3_label(self, row):
        label = [] 
        humour_dict = {
            'not_funny': 0,
            'funny': 1,
            'very_funny': 2,
            'hilarious': 3
        }
        sarcasm_dict = {
            'not_sarcastic': 4,
            'general': 5,
            'twisted_meaning': 6,
            'very_twisted': 7
        }
        motiv_dict = {
            'not_motivational': 8,
            'motivational': 9
        }
        offen_dict = {
            'not_offensive': 10,
            'slight': 11,
            'very_offensive': 12,
            'hateful_offensive': 13
        }
        return [humour_dict[row['humour']], sarcasm_dict[row['sarcasm']], offen_dict[row['offensive']], motiv_dict[row['motivational']]]

    def get_task2_label(self, row):
        label = [] 
        if row['humour'] in ['funny', 'hilarious', 'very_funny']:
            label += [0]
        if row['sarcasm'] in ['general', 'twisted_meaning', 'very_twisted']:
            label += [1]
        if row['motivational'] == 'motivational':
            label += [3]
        if row['offensive'] != 'not_offensive':
            label += [2]
        return label

    def get_sentiment_label(self, label):
        if label in ['positive', 'very_positive']:
            return 0 
        elif label in ['neutral']:
            return 1 
        elif label in ['negative', 'very_negative']:
            return 2 
        return -1

    def __getitem__(self, index):
        if self.task in ['a', 'A']:
            single_image_label = self.df['label'].iloc[index]
        else:
            single_image_label = np.array(self.df['label'].iloc[index], dtype=int)
            zeros = np.zeros(4 if self.task in ['b', 'B'] else 14)
            zeros[single_image_label] = 1
            single_image_label = zeros
        imgpath = os.path.join(self.img_path, self.df.iloc[index]['image_name'])
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
    path = '/home/khizirs/contr/hatefulmemes/dataset/data'
    dataset1 = SupervisionHatefulMemesDataset(path, 'train', txt_model, im_transforms=imtransforms, num_samples=num_samples)
    path = '/home/khizirs/MMHS150K'
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


def get_supervision_dataset_memotion(args):
    path, txt_model = args.data, args.txtmodel
    num_samples = args.n_samples
    train_transform = get_simclr_pipeline_transform(224)
    val_transform = transforms.Compose([transforms.CenterCrop(size=224),
                                        transforms.ToTensor()])
    traindataset = MemotionDataset(
            path, phase='train', task=args.task, txt_model=txt_model, im_transforms=train_transform, num_samples=num_samples)

    valdataset = MemotionDataset(path, phase='test', task=args.task, txt_model=txt_model, im_transforms=val_transform)
    return traindataset, valdataset


def get_supervision_dataset_harmeme(args):
    path, txt_model = args.data, args.txtmodel
    num_samples = args.n_samples
    train_transform = get_simclr_pipeline_transform(224)
    val_transform = transforms.Compose([transforms.CenterCrop(size=224),
                                        transforms.ToTensor()])
    traindataset = SupervisionHarmemesDataset(
            path, phase='train', txt_model=txt_model, im_transforms=train_transform, num_samples=num_samples)
    valdataset = SupervisionHarmemesDataset(path, phase='test', txt_model=txt_model, im_transforms=val_transform)
    return traindataset, valdataset
