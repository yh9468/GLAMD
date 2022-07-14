import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result

from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from base_detector import BaseDetector
import torch
from mmdet.models.losses import iou_loss, IoULoss

# feature l2 loss
def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    diff = diff * attention_mask
    diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff
            
def masked_softmax(x, **kwargs):
    weight = torch.sum((x != 0), dim=2, keepdim=True)
    x[x == 0] = -float("inf")
    return torch.softmax(x, **kwargs) * weight

def pairwise_dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    patch_num = tensor_a.shape[3]

    diff = (tensor_a - tensor_b) ** 2
    diff = diff * attention_mask
    diff = diff * channel_attention_mask
    
    diff = diff.permute(3, 0, 1, 2)
    diff = diff.view(patch_num,-1)
    diff = torch.sum(torch.sum(diff, dim=1) ** 0.5)

    return diff

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        
        # self.bbox_feat_adaptation = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        #   self.cls_adaptation = nn.Linear(1024, 1024)
        #   self.reg_adaptation = nn.Linear(1024, 1024)
        self.channel_wise_adaptation = nn.ModuleList([
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        ])

        self.local_channel_wise_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
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

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

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
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def get_teacher_info(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        _, head_outs, decoded_bbox_pred = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        t_info = {'feat':x, 'head_outs':head_outs, 'decoded_bbox': decoded_bbox_pred}
        return t_info

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      t_info=None,
                      epoch=None,
                      iter=None,
                      **kwargs
                      ):

        x = self.extract_feat(img)
        losses, s_outs, s_reg_outs = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        
        t = 0.1
        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0
        total_masks = []

        # feature distillation by using attention mask
        if t_info is not None:
            t_feats = t_info['feat']
            for _i in range(len(t_feats)):
                # Global part
                t_global_attention_mask = self.generate_attention_mask(t_feats[_i], x[0].size(0), t, type="spatial")
                s_global_attention_mask = self.generate_attention_mask(x[_i], x[0].size(0), t, type="spatial")
                c_t_global_attention_mask = self.generate_attention_mask(t_feats[_i], x[0].size(0), t, type="channel")
                c_s_global_attention_mask = self.generate_attention_mask(x[_i], x[0].size(0), t, type="channel")
                
                sum_global_attention_mask = (t_global_attention_mask + s_global_attention_mask) / 2
                sum_global_attention_mask = sum_global_attention_mask.detach()

                # Local part
                local_kd_feat_loss, local_kd_channel_loss, local_mask = self.calculate_local_attention_loss(t_feats[_i], x[_i], _i)
                
                total_mask = (local_mask + sum_global_attention_mask) / 2
                total_masks.append(total_mask)

                # making final feature mask using in feature distillation
                c_sum_global_attention_mask = (c_t_global_attention_mask + c_s_global_attention_mask) / 2
                c_sum_global_attention_mask = c_sum_global_attention_mask.detach()

                # feature loss by using teacher feature and student feature
                global_kd_feat_loss = dist2(t_feats[_i], self.adaptation_layers[_i](x[_i]), attention_mask=sum_global_attention_mask,
                                      channel_attention_mask=c_sum_global_attention_mask) * 7e-5 * 6

                kd_feat_loss += (local_kd_feat_loss + global_kd_feat_loss) / 2

                # original torch L2 loss & using this for channel kd loss
                kd_channel_loss += torch.dist(torch.mean(t_feats[_i], [2, 3]),
                                              self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3 * 3 + local_kd_channel_loss
                
                # spatial kd loss
                t_spatial_pool = torch.mean(t_feats[_i], [1]).view(t_feats[_i].size(0), 1, t_feats[_i].size(2),
                                                                   t_feats[_i].size(3))
                s_spatial_pool = torch.mean(x[_i], [1]).view(x[_i].size(0), 1, x[_i].size(2),
                                                             x[_i].size(3))
                kd_spatial_loss += torch.dist(t_spatial_pool, self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3 * 6

        losses.update({'kd_feat_loss': kd_feat_loss})
        losses.update({'kd_channel_loss': kd_channel_loss})
        losses.update({'kd_spatial_loss': kd_spatial_loss})

        # head output
        t_cls_outs = t_info['head_outs'][0]             # B x (80 * 9) x H x W (class)
        s_cls_outs = s_outs[0]                          # B x (4  * 9) x H x W (reg)
        t_reg_outs = t_info['decoded_bbox']
        
        # loss
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

    def calculate_local_attention_loss(self, t_feature, s_feature, index, patch_size=7):
        t = 0.1
        batch_size = t_feature.size(0)
        f_size = t_feature.size()
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

        # channel attention mask (B x 256 x 1 x patch_num)
        t_channel_attention_mask = F.avg_pool2d(t_feature.abs(), (patch_size, patch_size), stride=patch_size, ceil_mode=True)     # B x 256 x H/7 x W/7
        t_channel_attention_mask = t_channel_attention_mask.view(batch_size, 256, patch_num)                          # B x 256 x patch_num
        t_channel_attention_mask = torch.softmax(t_channel_attention_mask / t, dim=1) * 256
        t_channel_attention_mask = t_channel_attention_mask.unsqueeze(2)
        
        s_channel_attention_mask = F.avg_pool2d(s_feature.abs(), (patch_size, patch_size), stride=patch_size, ceil_mode=True)
        s_channel_attention_mask = s_channel_attention_mask.view(batch_size, 256, patch_num)
        s_channel_attention_mask = torch.softmax(s_channel_attention_mask / t, dim=1) * 256
        s_channel_attention_mask = s_channel_attention_mask.unsqueeze(2)
        
        # get channel attention for channel distillation loss
        t_channel_attention = F.avg_pool2d(t_feature, (patch_size, patch_size), stride=patch_size, ceil_mode=True)
        s_channel_attention = F.avg_pool2d(s_feature, (patch_size, patch_size), stride=patch_size, ceil_mode=True)
        
        # PAD INPUT (value = 0.0)
        t_feature = F.pad(t_feature, (0, width_pad, 0, height_pad))        
        s_feature = F.pad(s_feature, (0, width_pad, 0, height_pad))
        adap_s_feature = F.pad(adap_s_feature, (0, width_pad, 0, height_pad))
        
        # divide input by patches (B x 256 x (7x7) x patch_num)
        t_feature = F.unfold(t_feature, patch_size, stride=patch_size)
        t_feature = t_feature.view(batch_size, 256, patch_size*patch_size, patch_num)
        
        s_feature = F.unfold(s_feature, patch_size, stride=patch_size)
        s_feature = s_feature.view(batch_size, 256, patch_size*patch_size, patch_num)
        
        adap_s_feature = F.unfold(adap_s_feature, patch_size, stride=patch_size)
        adap_s_feature = adap_s_feature.view(batch_size, 256, patch_size*patch_size, patch_num)
        
        # spatial attention mask (B x 1 x (7x7) x patch_num)
        t_spatial_attention_mask = torch.mean(torch.abs(t_feature), [1], keepdim=True)
        t_spatial_attention_mask = masked_softmax(t_spatial_attention_mask / t, dim=2)
        
        s_spatial_attention_mask = torch.mean(torch.abs(s_feature), [1], keepdim=True)
        s_spatial_attention_mask = masked_softmax(s_spatial_attention_mask / t, dim=2)
        
        sum_spatial_attention_mask = (t_spatial_attention_mask + s_spatial_attention_mask) / 2
        sum_spatial_attention_mask = sum_spatial_attention_mask.detach()
                
        sum_channel_attention_mask = (t_channel_attention_mask + s_channel_attention_mask) / 2
        sum_channel_attention_mask = sum_channel_attention_mask.detach()
        
        # kd feature loss
        kd_feat_loss = pairwise_dist2(t_feature, adap_s_feature, attention_mask=sum_spatial_attention_mask, 
                                            channel_attention_mask=sum_channel_attention_mask) * 7e-5 * 6
        
        # attention loss (after experiment have to fix)     B x C x h/7 x w/7
        t_channel_attention = t_channel_attention.view(batch_size, 256, patch_num).permute(2,0,1)
        s_channel_attention = s_channel_attention.view(batch_size, 256, patch_num).permute(2,0,1)
        
        # B x patch_num x 256
        s_channel_attention = self.channel_wise_adaptation[index](s_channel_attention)
        
        t_channel_attention = t_channel_attention.view(patch_num, -1)
        s_channel_attention = s_channel_attention.view(patch_num, -1)
        
        pdist = nn.PairwiseDistance(p=2)
        kd_channel_loss = torch.mean(pdist(t_channel_attention, s_channel_attention)) * 4e-3 * 3

        # Origin size
        sum_spatial_attention_mask = F.fold(sum_spatial_attention_mask.squeeze(1), (f_size[2] + height_pad, f_size[3] + width_pad), patch_size, stride=patch_size)
        sum_spatial_attention_mask = sum_spatial_attention_mask[:,:,:f_size[2], :f_size[3]].detach()
        
        return kd_feat_loss, kd_channel_loss, sum_spatial_attention_mask

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            np.ndarray: proposals
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError
