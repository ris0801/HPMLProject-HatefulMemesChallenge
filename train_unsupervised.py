import time
import logging
import os

import numpy as np
import h5py

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from losses import SupConLoss, MMContrastiveLoss, CLIPLoss, ConVIRT, LossV0

from sklearn.manifold import TSNE

from tqdm import tqdm
from utils import save_config_file, save_checkpoint, EarlyStopping

from lightly.loss import ntx_ent_loss

class UnsupervisedLearner(object):
    def __init__(self, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
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
        if self.args.mmcontr:
            self.mmcontr_loss= MMContrastiveLoss(
                    margin=self.args.margin,
                    measure=self.args.measure,
                    max_violation=self.args.max_violation
            ).to(self.args.device)
        if self.args.supcontr:
            self.supcontr_loss = SupConLoss(temperature=self.args.temperature).to(self.args.device)
        if self.args.simclr:
            self.simclr_loss = ntx_ent_loss.NTXentLoss(self.args.temperature, self.args.moco_size).to(self.args.device)
        if self.args.cliploss:
            self.clip_loss = CLIPLoss(self.args.temperature).to(self.args.device)
        if self.args.memeloss:
            self.meme_mmloss = MMContrastiveLoss(
                    margin=self.args.margin,
                    measure=self.args.measure,
                    max_violation=self.args.max_violation
            ).to(self.args.device)
            self.meme_floss = ntx_ent_loss.NTXentLoss(self.args.temperature, self.args.moco_size).to(self.args.device)
        if self.args.convirt:
            self.convirt_loss = ConVIRT(self.args.temperature).to(self.args.device)
        if self.args.lossv0:
            self.lossv0_loss = LossV0(self.args.temperature).to(self.args.device)

    def compute_loss(self, out, step):
        loss = 0.0
        if self.args.simclr:
            simloss = self.simclr_loss(out['image1'], out['image2'])
            loss += simloss
            self.writer.add_scalar('Unsupervised/SimCLR', simloss.item(), step)
        if self.args.supcontr:
            t = torch.cat((out['image1'].unsqueeze(1), out['image2'].unsqueeze(1)), 1)
            supsimclrloss = self.supcontr_loss(t, out['label'])
            loss += supsimclrloss
            self.writer.add_scalar('Unsupervised/SupContrLoss', supsimclrloss.item(), step)
        if self.args.mmcontr:
            immloss, tmmloss = self.mmcontr_loss(out['image'], out['text'])
            loss += immloss + tmmloss
            self.writer.add_scalar('Unsupervised/ImageMMContrLoss', immloss.item(), step)
            self.writer.add_scalar('Unsupervised/TextMMContrLoss', tmmloss.item(), step)
        if self.args.cliploss:
            cliploss = self.clip_loss(out['image'], out['text'])
            loss += cliploss
            self.writer.add_scalar('Unsupervised/CLIPLoss', cliploss.item(), step)
        if self.args.memeloss:
            f2i, i2f = self.meme_mmloss(out['fusion'], out['image'])
            fusion2i = f2i * 0.7 + 0.3 * i2f
            f2t, t2f = self.meme_mmloss(out['fusion'], out['text'])
            fusion2t = f2t * 0.7 + 0.3 * t2f
            f2f = self.meme_floss(out['fusion1'], out['fusion2'])
            loss += self.args.w_f2i * fusion2i + self.args.w_f2t * fusion2t + self.args.w_f2f * f2f
            self.writer.add_scalar('Unsupervised/Fusion2ImageLoss', fusion2i.item(), step)
            self.writer.add_scalar('Unsupervised/Fusion2TextLoss', fusion2t.item(), step)
            self.writer.add_scalar('Unsupervised/Fusion2FusionLoss', f2f.item(), step)
        if self.args.convirt:
            i2t, t2i = self.convirt_loss(out['image'], out['text'])
            loss += i2t + t2i
            self.writer.add_scalar('Unsupervised/ConVIRTImage2TextLoss', i2t.item(), step)
            self.writer.add_scalar('Unsupervised/ConVIRTText2ImageLoss', t2i.item(), step)
        if self.args.lossv0:
            lossv0 = self.lossv0_loss(out['image'], out['text'])
            loss += lossv0
            self.writer.add_scalar('Unsupervised/LossV0Loss', lossv0.item(), step)

        self.writer.add_scalar('Unsupervised/Loss', loss, step)
        return loss

    def get_batch(self, it):
        if self.args.n_views == 2:
            (image1, image2), text, mask, label = next(it)
            batch = {
                    'image1': image1,
                    'image2': image2,
                    'text': text,
                    'mask': mask,
                    'label': label
            }
        else:
            image, text, mask, label = next(it)
            batch = {
                    'image1': image,
                    'image2': None,
                    'text': text,
                    'mask': mask,
                    'label': label
            }
        batch = {k:v.to(self.args.device) if v is not None else None for k, v in batch.items()}
        return batch

    def save_embed(self, train_loader):
        self.model.eval()
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start Embedding Visualization.")
        logging.info(f"Using args: {self.args}")

        trainiterator = iter(train_loader)
        image_embeddings = []
        text_embeddings = []
        all_labels = []
        with torch.no_grad():
            for loader_idx in range(len(train_loader)):
                batch = self.get_batch(trainiterator)
                im = self.model.encode_image(batch['image'])
                txt = self.model.encode_text(batch['text'])
                image_embeddings.append(im.cpu().numpy())
                text_embeddings.append(txt.cpu().numpy())
                all_labels.append(batch['label'].cpu().numpy())
                print("Runs")

        classes= {1: 'Hateful', 0: 'Non-hateful'}
        image_embeddings = np.concatenate(image_embeddings, axis=0)
        text_embeddings = np.concatenate(text_embeddings, axis=0)
        all_labels = np.concatenate(all_labels)
        class_labels = [classes[i.item()] for i in all_labels]
        print("Image Embeddings:", image_embeddings.shape)
        print("Text Embeddings:", text_embeddings.shape)
        print("Labels:", all_labels.shape)

        h_file = h5py.File(os.path.join(self.writer.log_dir,'embeds.h5'), 'w')
        h_file.create_dataset('image_embeds', data=image_embeddings)
        h_file.create_dataset('text_embeds', data=text_embeddings)
        h_file.create_dataset('labels_embeds', data=all_labels)
        h_file.close()

        green = all_labels == 0
        red = all_labels == 1

        for perplexity in [5, 25, 50, 100]:
            t1 = time.time()
            tsne = TSNE(2, perplexity, n_iter=2500, n_jobs=12)
            Xi_embed = tsne.fit_transform(image_embeddings)
            t2 = time.time()
            logging.info(f"TSNE {perplexity} took {t2-t1}s on image_embeddings")
            t1 = time.time()
            tsne = TSNE(2, perplexity, n_iter=2500, n_jobs=12)
            Xt_embed = tsne.fit_transform(text_embeddings)
            t2 = time.time()
            logging.info(f"TSNE {perplexity} took {t2-t1}s on text_embeddings")
            h_file = h5py.File(os.path.join(self.writer.log_dir,'embeds-tsne-{}.h5'.format(perplexity)), 'w')
            h_file.create_dataset('image_embeds', data=Xi_embed)
            h_file.create_dataset('text_embeds', data=Xt_embed)
            h_file.create_dataset('labels_embeds', data=all_labels)
            h_file.close()

            plt.figure(figsize=(15, 8))
            plt.title(f"TSNE {perplexity} Text Embeddings")
            plt.scatter(Xt_embed[green, 0], Xt_embed[green, 1], c='g')
            plt.scatter(Xt_embed[red, 0], Xt_embed[red, 1], c='r')
            plt.tight_layout()
            plt.savefig(os.path.join(self.writer.log_dir,'embeds-tsne-text-{}.png'.format(perplexity)))
            plt.cla()

            plt.figure(figsize=(15, 8))
            plt.title(f"TSNE {perplexity} Image Embeddings")
            plt.scatter(Xi_embed[green, 0], Xi_embed[green, 1], c='g')
            plt.scatter(Xi_embed[red, 0], Xi_embed[red, 1], c='r')
            plt.tight_layout()
            plt.savefig(os.path.join(self.writer.log_dir,'embeds-tsne-image-{}.png'.format(perplexity)))
            plt.cla()
            logging.info(f"TSNE {perplexity} plot saved")
        logging.info("Complete")

    def train(self, train_loader):
        self.model.train()
        scaler = GradScaler(enabled=self.args.fp16_precision)
        save_config_file(self.writer.log_dir, self.args)
        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Using args: {self.args}")

        earlyStopper = EarlyStopping(patience=10)

        for epoch_counter in tqdm(range(self.args.epochs), disable=self.args.no_tqdm):
            trainiterator = iter(train_loader)
            for loader_idx in range(len(train_loader)):
                batch = self.get_batch(trainiterator)

                if self.args.dryrun:
                    if loader_idx == 4:
                        print("Dry Run in Unsupervised train complete, exiting")
                        break

                with autocast(enabled=self.args.fp16_precision):
                    out = self.model(batch)
                    loss = self.compute_loss(out, n_iter)
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                n_iter += 1

            earlyStopper(loss.item())

            logging.debug("Epoch: {}\tLoss: {}".format(epoch_counter, loss.item()))
            
            if self.args.dryrun:
                break

            if epoch_counter >= 10:
                self.scheduler.step()

            if epoch_counter % 10 == 0:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Checkpoint created at {checkpoint_name}")

            if earlyStopper.early_stop:
                logging.info("Early Stopping, Loss didn't decrease for several epochs")
                break

        logging.info("Training has finished.")

        checkpoint_name = 'last_checkpoint-{}.pth.tar'.format(self.model.name)
        filename = os.path.join(self.writer.log_dir, checkpoint_name)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=filename)

        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        logging.info("Completed self-supervised training.")
