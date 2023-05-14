import logging
import os

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import numpy as np

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint, plot_confusion_matrix


class SupervisedLearner(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.classifier = kwargs['classifier'].to(self.args.device)
        print("="*35, "CLASSIFIER", "="*35)
        print(self.classifier)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        if self.args.experiment == '':
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir='runs/' + self.args.experiment)
        logger_fmt = '[%(asctime)s] : [%(levelname)s : %(name)s] :: %(message)s'
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'),
                level=logging.DEBUG, format=logger_fmt)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)

        if self.args.task in ['a', 'A']:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss().to(self.args.device)

    def train(self, train_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        self.model.eval()
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start Supervised training for {self.args.epochs} epochs.")
        logging.info(f"Using args: {self.args}")
        best_val_f1 = 0.0
        corr_train_f1 = 0.0
        best_val_auroc = 0.0
        corr_train_auroc = 0.0
        for epoch_counter in tqdm(range(self.args.epochs), disable=self.args.no_tqdm):
            trainloss = 0
            correct = 0
            total_preds = torch.tensor([])
            total_true = torch.tensor([])
            self.classifier.train()
            for train_loader_idx, (img1, text, mask, labels) in enumerate(train_loader):

                if self.args.dryrun and train_loader_idx >= 4:
                    print("Dry Run in train complete, exiting")
                    break

                bs = img1.shape[0]
                img1 = img1.to(self.args.device)
                text = text.to(self.args.device)
                mask = mask.to(self.args.device)
                labels = labels.to(self.args.device)
                with autocast(enabled=self.args.fp16_precision):
                    with torch.no_grad():
                        txt_repr = self.model.text_encoder(input_ids=text, attention_mask=mask)
                        img_feats = self.model.image_encoder(img1)
                    out = self.classifier(img_feats, txt_repr)
                    loss = self.criterion(out, labels)
                    total_preds = torch.cat((total_preds, out.cpu().detach()), dim=0)
                    total_true = torch.cat((total_true, labels.cpu()), dim=0)
                    trainloss += loss.item()

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

            pos_probs = total_preds[:, 1]
            if self.args.num_classes == 2:
                train_auroc = roc_auc_score(total_true, pos_probs)
            if self.args.task in ['a', 'A']:
                total_preds = total_preds.argmax(dim=1)
                train_cm = confusion_matrix(total_true, total_preds)
            else:
                total_preds = (torch.sigmoid(total_preds) > 0.5)
            train_ac = accuracy_score(total_true, total_preds)
            train_f1 = f1_score(total_true, total_preds, average='macro')

            trainloss /= len(train_loader) 

            if epoch_counter >= 3 and self.scheduler is not None:
                self.scheduler.step()

            self.classifier.eval()
            with torch.no_grad():
                valloss = 0
                total_preds = torch.tensor([])
                total_true = torch.tensor([])
                for val_loader_idx, (img1, text, mask, labels) in enumerate(val_loader):

                    if self.args.dryrun:
                        if val_loader_idx == 4:
                            print("Dry Run in val complete, exiting")
                            break

                    bs = img1.shape[0]
                    img1 = img1.to(self.args.device)
                    text = text.to(self.args.device)
                    mask = mask.to(self.args.device)
                    text = torch.squeeze(text, 1)
                    mask = torch.squeeze(mask, 1)
                    labels = labels.to(self.args.device)
                    
                    with autocast(enabled=self.args.fp16_precision):
                        txt_repr = self.model.text_encoder(text, mask)
                        img1_feats = self.model.image_encoder(img1)
                        out = self.classifier(img1_feats, txt_repr)
                        total_preds = torch.cat((total_preds, out.cpu().detach()), dim=0)
                        total_true = torch.cat((total_true, labels.cpu()), dim=0)
                        loss = self.criterion(out, labels)
                        valloss += loss.item()

                valloss /= len(val_loader) 
                pos_probs = total_preds[:, 1]

                if self.args.num_classes == 2:
                    val_auroc = roc_auc_score(total_true, pos_probs)
                if self.args.task in ['a', 'A']:
                    total_preds = total_preds.argmax(dim=1)
                    val_cm = confusion_matrix(total_true, total_preds)
                else:
                    total_preds = (torch.sigmoid(total_preds) > 0.5)

                val_ac = accuracy_score(total_true, total_preds)
                val_f1 = f1_score(total_true, total_preds, average='macro')

            if self.args.num_classes == 2:
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    corr_train_auroc = train_auroc

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                corr_train_f1 = train_f1

            if self.args.dryrun:
                break

            self.writer.add_scalar('training/loss', trainloss, global_step=n_iter)
            self.writer.add_scalar('validation/loss', valloss, global_step=n_iter)
            self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
            self.writer.add_scalar('training/f1_score', train_f1, n_iter)
            self.writer.add_scalar('validation/f1_score', val_f1, n_iter)
            if self.args.num_classes == 2:
                self.writer.add_scalar('training/auc_roc', train_auroc, n_iter)
                self.writer.add_scalar('validation/auc_roc', val_auroc, n_iter)

            self.writer.add_scalar('training/accuracy', train_ac, global_step=n_iter)
            self.writer.add_scalar('validation/accuracy', val_ac, global_step=n_iter)
            if self.args.task in ['a', 'A']:
                train_cm_fig = plot_confusion_matrix(train_cm, ['0', '1', '2'])
                self.writer.add_figure('training/confusion_matrix', train_cm_fig, global_step=n_iter)
                val_cm_fig = plot_confusion_matrix(val_cm, ['0', '1', '2'])
                self.writer.add_figure('validation/confusion_matrix', val_cm_fig, global_step=n_iter)

            msg = f"Epoch: {epoch_counter}\tTrain Loss: {trainloss}\tValidation Loss: {valloss}"
            msg += f"\n-----:---\tTrain Accuracy: {train_ac}\tValidation Accuracy: {val_ac}"
            msg += f"\n-----:---\tTrain F1: {train_f1}\tValidation F1: {val_f1}"
            if self.args.num_classes == 2:
                msg += f"\n-----:---\tTrain AUROC: {train_auroc}\tValidation AUROC: {val_auroc}"
            logging.info(msg)
            print(msg) 
            n_iter = n_iter + 1

            if epoch_counter % 10 == 9:   
                checkpoint_name = '{}_{:04d}.pth.tar'.format(self.classifier.name, epoch_counter)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.classifier.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Checkpoint created at {checkpoint_name}")

        # save model checkpoints
        checkpoint_name = 'last_checkpoint-{}--c--{}.pth.tar'.format(self.model.name, self.classifier.name)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        msg = "Training F1:{:.4f} and Testset F1:{:.4f}".format(corr_train_f1, best_val_f1)
        if self.args.num_classes == 2:
            msg += " Training AUROC: {:.4f} and Testset AUROC: {:.4f}".format(corr_train_auroc, best_val_auroc)
        logging.info("Training Completed")
        logging.info(msg)
    
    def evaluate(self, val_loader):
        self.model.eval()
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start Supervised Evaluation for {self.args.epochs} epochs.")
        logging.info(f"Using args: {self.args}")
        best_val_f1 = 0.0
        corr_train_f1 = 0.0
        best_val_auroc = 0.0
        corr_train_auroc = 0.0
        self.classifier.eval()
        with torch.no_grad():
            valloss = 0
            total_preds = torch.tensor([])
            total_true = torch.tensor([])
            for val_loader_idx, (img1, text, mask, labels) in enumerate(val_loader):
                bs = img1.shape[0]
                img1 = img1.to(self.args.device)
                text = text.to(self.args.device)
                mask = mask.to(self.args.device)
                text = torch.squeeze(text, 1)
                mask = torch.squeeze(mask, 1)
                labels = labels.to(self.args.device)
                    
                with autocast(enabled=self.args.fp16_precision):
                    txt_repr = self.model.text_encoder(text, mask)
                    img1_feats = self.model.image_encoder(img1)
                    out = self.classifier(img1_feats, txt_repr)
                    total_preds = torch.cat((total_preds, out.cpu().detach()), dim=0)
                    total_true = torch.cat((total_true, labels.cpu()), dim=0)
                    loss = self.criterion(out, labels)
                    valloss += loss.item()

            valloss /= len(val_loader) 
            pos_probs = total_preds[:, 1]

            if self.args.num_classes == 2:
                val_auroc = roc_auc_score(total_true, pos_probs)
            if self.args.task in ['a', 'A']:
                total_preds = total_preds.argmax(dim=1)
                val_cm = confusion_matrix(total_true, total_preds)
            else:
                total_preds = (torch.sigmoid(total_preds) > 0.5)

            val_ac = accuracy_score(total_true, total_preds)
            val_f1 = f1_score(total_true, total_preds, average='macro')
            if self.args.num_classes == 2:
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

        msg = "Testset F1:{:.4f}".format(best_val_f1)
        if self.args.num_classes == 2:
            msg += " and AUROC: {:.4f}".format(best_val_auroc)
        logging.info("Evaluation Completed")
        logging.info(msg)
