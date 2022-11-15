import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
import torch.optim as optim
import torch.nn as nn

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm

def lr_poly(base_lr, iter, max_iters, power):
    return base_lr * ((1 - float(iter) / max_iters) ** (power))

class FCDiscriminatorWoCls(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminatorWoCls, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1) # 4*4*19*64
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1) # 4*4*64*128
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 4*4*128*256
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # 4*4*256*512
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1) # 4*4*512*1
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Vanilla
        # self.bce_loss = torch.nn.BCEWithLogitsLoss()
        # LS
        self.bce_loss = torch.nn.MSELoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x

    def forward_train(self, inputs, gt_dis, retrun_inv=False):
        pred = self.forward(inputs)
        
        dev = pred.device
        losses = dict()
        loss = self.bce_loss(pred, torch.FloatTensor(pred.data.size()).fill_(gt_dis).to(device=dev))
        losses['loss_dis'] = loss

        if retrun_inv:
            loss_inv = self.bce_loss(pred, torch.FloatTensor(pred.data.size()).fill_(1-gt_dis).to(device=dev))
            losses['loss_dis_inv'] = loss_inv

        return losses

@UDA.register_module()
class GAN(UDADecorator):

    def __init__(self, **cfg):
        super(GAN, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        # self.pseudo_threshold = cfg['pseudo_threshold']
        # self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        # self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.print_grad_magnitude = cfg['print_grad_magnitude']

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None
        
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.power = cfg['power']

        self.src_lbl = 0
        self.tgt_lbl = 1

        # self.img_adv_lambda = cfg['img_adv_lambda']
        # self.px_adv_lambda = cfg['px_adv_lambda']

        self.lr_dis = cfg['lr_dis']
        self.px_wo_cls_d_model = FCDiscriminatorWoCls(self.num_classes).to(self.dev)
        self.px_wo_cls_d_optim = optim.Adam(self.px_wo_cls_d_model.parameters(), lr=self.lr_dis, betas=(0.9, 0.99))
        self.px_wo_cls_adv_lambda = cfg['px_wo_cls_adv_lambda']

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    # def get_img_d_model(self):
    #     return get_module(self.img_d_model)
    
    # def get_px_d_model(self):
    #     return get_module(self.px_d_model)
    
    # Original discriminate model without class
    def get_px_wo_cls_d_model(self):
        return self.px_wo_cls_d_model

    def adjust_learning_rate_d(self, optimizer):
        lr = lr_poly(self.lr_dis, self.local_iter, self.max_iters, self.power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        if self.local_iter == 0:
            self.px_wo_cls_d_model.train()

        self.px_wo_cls_d_optim.zero_grad()
        # Adjust learning rate
        self.adjust_learning_rate_d(self.px_wo_cls_d_optim)
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()
        self.px_wo_cls_d_optim.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Train G

        # Don't accumulate grads in D
        # for param in get_img_d_model().parameters():
        #     param.requires_grad = False
        # for param in get_px_d_model().parameters():
        #     param.requires_grad = False
        for param in self.get_px_wo_cls_d_model().parameters():
            param.requires_grad = False

        # Train on source images 
        clean_losses = self.get_model().forward_train_w_pred(
            img, img_metas, gt_semantic_seg, return_feat=True, return_pred=True)

        src_feat = clean_losses.pop('features')
        src_pred = clean_losses.pop('pred')
        src_pred = torch.softmax(src_pred, dim=1)

        clean_losses = add_prefix(clean_losses, 'src')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        # Segmentation loss
        clean_loss.backward(retain_graph=True)

        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # Adversarial
        px_wo_cls_adv_losses = self.get_px_wo_cls_d_model().forward_train(
            src_pred, self.tgt_lbl, retrun_inv=True)
        # loss_inv = px_wo_cls_adv_losses.pop('loss_dis_inv')

        px_wo_cls_adv_losses = add_prefix(px_wo_cls_adv_losses, 'adv.src')
        px_wo_cls_adv_loss, px_wo_cls_adv_log_vars = self._parse_losses(px_wo_cls_adv_losses)
        log_vars.update(px_wo_cls_adv_log_vars)

        # 1/2 * 1/2(raw + inverse)
        px_wo_cls_adv_loss = self.px_wo_cls_adv_lambda * (px_wo_cls_adv_loss / 4)
        px_wo_cls_adv_loss.backward(retain_graph=True)

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Train on target images
        tgt_pred = self.get_model().encode_decode(
            target_img, target_img_metas)
        tgt_pred = torch.softmax(tgt_pred, dim=1)

        # Adversarial
        px_wo_cls_adv_losses = self.get_px_wo_cls_d_model().forward_train(
            tgt_pred, self.src_lbl, retrun_inv=True)

        px_wo_cls_adv_losses = add_prefix(px_wo_cls_adv_losses, 'adv.tgt')
        px_wo_cls_adv_loss, px_wo_cls_adv_log_vars = self._parse_losses(px_wo_cls_adv_losses)
        log_vars.update(px_wo_cls_adv_log_vars)

        px_wo_cls_adv_loss = self.px_wo_cls_adv_lambda * (px_wo_cls_adv_loss / 4)
        px_wo_cls_adv_loss.backward()

        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            px_wo_cls_adv_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None]
            grad_mag = calc_grad_magnitude(px_wo_cls_adv_grads)
            mmcv.print_log(f'PxWoClsAdv Grad.: {grad_mag}', 'mmseg')
        
        # Train D
        
        # Bring back requires_grad
        # for param in get_img_d_model().parameters():
        #     param.requires_grad = True
        # for param in get_px_d_model().parameters():
        #     param.requires_grad = True
        for param in self.get_px_wo_cls_d_model().parameters():
            param.requires_grad = True

        # Train on source images
        # Block gradients back to the segmentation network
        src_pred = src_pred.detach()
        px_wo_cls_losses = self.get_px_wo_cls_d_model().forward_train(
            src_pred, self.src_lbl)

        px_wo_cls_losses = add_prefix(px_wo_cls_losses, 'src')
        px_wo_cls_loss, px_wo_cls_log_vars = self._parse_losses(px_wo_cls_losses)
        log_vars.update(px_wo_cls_log_vars)

        # Discriminate loss
        px_wo_cls_loss = px_wo_cls_loss / 2
        px_wo_cls_loss.backward()

        # Train on target images
        tgt_pred = tgt_pred.detach()
        px_wo_cls_losses = self.get_px_wo_cls_d_model().forward_train(
            tgt_pred, self.tgt_lbl)

        px_wo_cls_losses = add_prefix(px_wo_cls_losses, 'tgt')
        px_wo_cls_loss, px_wo_cls_log_vars = self._parse_losses(px_wo_cls_losses)
        log_vars.update(px_wo_cls_log_vars)

        # Discriminate loss
        px_wo_cls_loss = px_wo_cls_loss / 2
        px_wo_cls_loss.backward()

        self.local_iter += 1

        return log_vars
