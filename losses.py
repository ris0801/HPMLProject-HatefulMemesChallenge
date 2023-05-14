from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils import view_tensor

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, debug=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.debug = False

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        view_tensor("labels", labels, debug=self.debug)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        view_tensor("mask", mask, debug=self.debug)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        view_tensor("logits", logits, debug=self.debug)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        view_tensor("logits_mask", logits_mask, debug=self.debug)
        mask = mask * logits_mask
        view_tensor("mask", mask, debug=self.debug)

        # compute log_prob
        logits_sum = logits_mask.sum(1)
        view_tensor("logits_sum", logits_sum, debug=self.debug)
        exp_logits = torch.exp(logits) * logits_mask
        view_tensor("exp_logits", exp_logits, debug=self.debug)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        view_tensor("log_prob", log_prob, debug=self.debug)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = mask * log_prob
        view_tensor("mean_log_prob_pos", mean_log_prob_pos, debug=self.debug)
        mean_log_prob_pos = (mean_log_prob_pos).sum(1) / mask.sum(1)
        view_tensor("mean_log_prob_pos", mean_log_prob_pos, debug=self.debug)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        view_tensor("loss", loss, debug=self.debug)

        return loss

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
            - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class MMContrastiveLoss(nn.Module):
    """
    Compute Multimodal Contrastive Loss
    """
    def __init__(self, margin=0, measure=False, max_violation=False):
        super(MMContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum(), cost_im.sum()


class CLIPLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        # embeddings must already be normalized

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0
        return loss.mean()

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()


class ConVIRT(torch.nn.Module):
    def __init__(self, temperature):
        super(ConVIRT, self).__init__()
        self.temperature = temperature

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
        loss = - (target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs, norm=False):
        temperature = self.temperature
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
            
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = labels.to(hidden1.device)
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2, 0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1, 0, 1)) / temperature

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)

        return loss_a, loss_b


class LossV0(nn.Module):
    def __init__(self, temperature, lambd=0.0001):
        super().__init__()
        self.T = temperature
        self.lambd = lambd

    def forward(self, zt, zi):
        C = zt.mm(zi.t()) / self.T
        C = torch.exp(C)
        positives = torch.diag(C)
        C = C - torch.diag(positives)
        sum_1 = C.sum(1)
        sum_0 = C.sum(0)
        sum_p = sum_1 + sum_0
        lamb_sum = self.lambd * (C.sum() - sum_p)
        return -torch.log((1 + self.lambd) * torch.sum(positives / (lamb_sum + sum_p)))
