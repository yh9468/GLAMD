from atexit import register
import torch
import torch.nn as nn
import math
# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from base_detector import BaseDetector
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from mmdet.models.losses import iou_loss, IoULoss

def masked_softmax(x, **kwargs):
    weight = torch.sum((x != 0), dim=2, keepdim=True)
    x[x == 0] = -float("inf")
    return torch.softmax(x, **kwargs) * weight

def relation_pairwise_dist2(tensor_a, tensor_b):
    # tensor shape : (B x 256 x H-num x W-num)

    # B x patchsize x 256
    tensor_a_line = tensor_a.view(tensor_a.size(0), 256, -1).permute(0,2,1)
    relation_matrix_a = torch.cdist(tensor_a_line, tensor_a_line, p=2, compute_mode="use_mm_for_euclid_dist")           # B x patch_size x patch_size
    relation_matrix_a = F.normalize(relation_matrix_a, dim=1)

    tensor_b_line = tensor_b.view(tensor_b.size(0), 256, -1).permute(0,2,1)
    relation_matrix_b = torch.cdist(tensor_b_line, tensor_b_line, p=2, compute_mode="use_mm_for_euclid_dist")
    relation_matrix_b = F.normalize(relation_matrix_b, dim=1)
    
    matrix_size = relation_matrix_a.shape[1] * relation_matrix_a.shape[2]

    diff = (relation_matrix_a - relation_matrix_b) ** 2

    diff = torch.sum(diff) / matrix_size
    return diff

def pairwise_dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    patch_num = tensor_a.shape[3]

    diff = (tensor_a - tensor_b) ** 2
    if attention_mask is not None:
        diff = diff * attention_mask
    if channel_attention_mask is not None:
        diff = diff * channel_attention_mask

    diff = diff.permute(3, 0, 1, 2)
    diff = diff.view(patch_num,-1)
    diff = torch.sum((torch.sum(diff, dim=1) + 1e-10) ** 0.5)
    return diff

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    if attention_mask is not None:
        diff = diff * attention_mask
    if channel_attention_mask is not None:
        diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff

@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        self.adaptation_type = '1x1conv'

        self.channel_wise_adaptation = nn.ModuleList([
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        ])

        self.spatial_wise_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        ])

        self.adaptation_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        ])

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)
            
        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            try:
                self.roi_head.init_weights(pretrained)
            except:
                self.roi_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def get_teacher_info(self,
                         img,
                         img_metas,
                         gt_bboxes,
                         gt_labels,
                         gt_bboxes_ignore=None,
                         gt_masks=None,
                         proposals=None,
                         t_feats=None,
                         **kwargs):
        teacher_info = {}
        x = self.extract_feat(img)
        teacher_info.update({'feat': x})

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            _, _, rpn_outs, t_reg_bbox = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)

            teacher_info.update({'rpn_out': rpn_outs, "decoded_bbox": t_reg_bbox})
            
        else:
            proposal_list = proposals
        return teacher_info


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      t_info=None,
                      epoch=None,
                      iter=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        t = 0.5
        x = self.extract_feat(img)
        losses = dict()

        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0

        total_masks = []

        if t_info is not None:
            t_feats = t_info['feat']
            for _i in range(len(t_feats)):
                t_global_attention_mask = self.generate_attention_mask(t_feats[_i], x[0].size(0), t, type="spatial")
                s_global_attention_mask = self.generate_attention_mask(x[_i], x[0].size(0), t, type="spatial")

                sum_global_attention_mask = (t_global_attention_mask + s_global_attention_mask) / 2
                sum_global_attention_mask = sum_global_attention_mask.detach()
                
                local_kd_feat, local_kd_channel, local_mask = self.calculate_local_attention_loss(t_feats[_i], x[_i], _i, patch_size=7)
 
                total_mask = (local_mask + sum_global_attention_mask) / 2
                total_masks.append(total_mask)

                kd_feat_loss += (dist2(t_feats[_i], self.adaptation_layers[_i](x[_i]), attention_mask=sum_global_attention_mask,
                                    channel_attention_mask=None) * 7e-5 + local_kd_feat) / 2

                kd_channel_loss += (torch.dist(torch.mean(t_feats[_i], [2, 3]),
                                            self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3 + local_kd_channel) / 2

                
                t_spatial_pool = torch.mean(t_feats[_i], [1]).view(t_feats[_i].size(0), 1, t_feats[_i].size(2),
                                                                   t_feats[_i].size(3))
                s_spatial_pool = torch.mean(x[_i], [1]).view(x[_i].size(0), 1, x[_i].size(2),
                                                             x[_i].size(3))

                kd_spatial_loss += torch.dist(t_spatial_pool, self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3

        losses.update({'kd_feat_loss': kd_feat_loss})
        losses.update({'kd_channel_loss': kd_channel_loss})
        losses.update({'kd_spatial_loss': kd_spatial_loss})

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            rpn_losses, proposal_list, s_outs, s_reg_outs = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
            )
            t_cls_outs = t_info['rpn_out'][0]
            s_cls_outs = s_outs[0]
            t_reg_outs = t_info['decoded_bbox']
            
            kd_cls_head_loss = 0
            kd_reg_head_loss = 0
            reg_loss = IoULoss(linear=True)

            for idx in range(len(t_cls_outs)):
                t_cls_score_sigmoid = t_cls_outs[idx].sigmoid()
                s_cls_score_sigmoid = s_cls_outs[idx].sigmoid()

                # loc
                B, C, H, W = s_outs[1][idx].shape
                reg_mask = total_masks[idx]
                reg_mask = reg_mask.expand(B, C, H, W)
                reg_mask = reg_mask.permute(0, 2, 3, 1).reshape(-1, 4)[:, 0].detach()
                kd_reg_head_loss += reg_loss(s_reg_outs[idx], t_reg_outs[idx], weight=reg_mask)

                # cls
                loss_map = F.binary_cross_entropy(s_cls_score_sigmoid, t_cls_score_sigmoid, reduction='none')
                kd_cls_head_loss += (loss_map * total_masks[idx]).sum() / total_masks[idx].sum()

            losses.update({'kd_cls_head_loss': kd_cls_head_loss * 0.1})
            losses.update({'kd_reg_head_loss': kd_reg_head_loss * 0.1})
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                          gt_bboxes, gt_labels,
                                                          gt_bboxes_ignore, gt_masks)
        losses.update(roi_losses)
        return losses

    def generate_attention_mask(self, feature, batch_size, t, type="spatial"):
        if type=="spatial":
            attention_mask = torch.mean(torch.abs(feature), [1], keepdim=True) # B x 1 x H x W
        elif type=="channel":
            attention_mask = torch.mean(torch.abs(feature), [2, 3], keepdim=True) # B x 256 x 1 x 1
        else:
            raise NameError
        
        size = attention_mask.size()  
        attention_mask = attention_mask.view(batch_size, -1)

        if type=="spatial":
            attention_mask = torch.softmax(attention_mask / t, dim=1) * size[-1] * size[-2]
        elif type=="channel":
            attention_mask = torch.softmax(attention_mask / t, dim=1) * 256
        else:
            raise NameError
        
        attention_mask = attention_mask.view(size)
        return attention_mask

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)


    def calculate_local_attention_loss(self, t_feature, s_feature, index, patch_size=7):
        t = 0.5
        batch_size = t_feature.size(0)
        f_size = t_feature.size()
        stride = patch_size
        adap_s_feature = self.adaptation_layers[index](s_feature)
        
        if f_size[2] % patch_size == 0:
            height_num = f_size[2] // patch_size
            height_pad = 0
        else:
            height_num = (f_size[2] // patch_size) + 1
            height_pad = patch_size - (f_size[2] % patch_size)
            
        if f_size[3] % patch_size == 0:
            width_num = f_size[3] // patch_size
            width_pad = 0
        else:
            width_num = (f_size[3] // patch_size) + 1
            width_pad = patch_size - (f_size[3] % patch_size)
        
        if height_num == 0:
            height_num = 1
        if width_num == 0:
            width_num = 1
        patch_num = height_num * width_num
        
        # get channel attention for channel distillation loss
        t_channel_attention = F.avg_pool2d(t_feature, (patch_size, patch_size), stride=stride, ceil_mode=True)
        s_channel_attention = F.avg_pool2d(s_feature, (patch_size, patch_size), stride=stride, ceil_mode=True)
        
        # PAD INPUT (value = 0.0)
        t_feature = F.pad(t_feature, (0, width_pad, 0, height_pad))     
        s_feature = F.pad(s_feature, (0, width_pad, 0, height_pad))
        adap_s_feature = F.pad(adap_s_feature, (0, width_pad, 0, height_pad))
        
        # divide input by patches (B x 256 x (7x7) x patch_num)
        t_feature = F.unfold(t_feature, patch_size, stride=stride)
        t_feature = t_feature.view(batch_size, 256, patch_size*patch_size, patch_num)
        
        s_feature = F.unfold(s_feature, patch_size, stride=stride)
        s_feature = s_feature.view(batch_size, 256, patch_size*patch_size, patch_num)
        
        adap_s_feature = F.unfold(adap_s_feature, patch_size, stride=stride)
        adap_s_feature = adap_s_feature.view(batch_size, 256, patch_size*patch_size, patch_num)
        
        # spatial attention mask (B x 1 x (7x7) x patch_num)
        t_spatial_attention_mask = torch.mean(torch.abs(t_feature), [1], keepdim=True)
        t_spatial_attention_mask = masked_softmax(t_spatial_attention_mask / t, dim=2)
        #t_spatial_attention = F.normalize(torch.mean(t_feature, [1]), dim=[1,2])
        
        s_spatial_attention_mask = torch.mean(torch.abs(s_feature), [1], keepdim=True)
        s_spatial_attention_mask = masked_softmax(s_spatial_attention_mask / t, dim=2)
        #s_spatial_attention = F.normalize(torch.mean(s_feature, [1]), dim=[1,2])
        
        sum_spatial_attention_mask = torch.abs(t_spatial_attention_mask + s_spatial_attention_mask) / 2
        sum_spatial_attention_mask = sum_spatial_attention_mask.detach()
        
        # kd feature loss
        kd_feat_loss = pairwise_dist2(t_feature, adap_s_feature, attention_mask=sum_spatial_attention_mask, 
                                            channel_attention_mask=None) * 7e-5

        # attention loss (after experiment have to fix)     B x C x h/7 x w/7
        t_channel_attention = t_channel_attention.view(batch_size, 256, patch_num).permute(2,0,1)
        s_channel_attention = s_channel_attention.view(batch_size, 256, patch_num).permute(2,0,1)
        
        # B x patch_num x 256
        s_channel_attention = self.channel_wise_adaptation[index](s_channel_attention)
        t_channel_attention = t_channel_attention.view(patch_num, -1)
        s_channel_attention = s_channel_attention.view(patch_num, -1)
        
        pdist = nn.PairwiseDistance(p=2)
        kd_channel_loss = torch.mean(pdist(t_channel_attention, s_channel_attention)) * 4e-3

        sum_spatial_attention_mask = F.fold(sum_spatial_attention_mask.squeeze(1), (f_size[2] + height_pad, f_size[3] + width_pad), patch_size, stride=stride)
        sum_spatial_attention_mask = sum_spatial_attention_mask[:,:,:f_size[2], :f_size[3]].detach()
        
        return kd_feat_loss, kd_channel_loss, sum_spatial_attention_mask

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
