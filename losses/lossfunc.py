import losses.pytorch_iou
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2,os

import numpy as np


def bce_iou_loss(pred,target):
    bce_loss = nn.BCELoss(size_average=True)
    iou_loss = losses.pytorch_iou.IOU(size_average=True)


    bce_out = bce_loss(pred,target)
    iou_out = iou_loss(pred,target)
    loss = bce_out + iou_out

    return loss
def bce_iou_loss_logit(pred, mask):
    # size = pred.size()[2:]
    # mask = F.interpolate(mask,size=size, mode='bilinear')
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()



# MM-Rebuttul
def loss_vit_simple_edgelist(dout,dout_c,dmid_list,de_list, sideouts,\
    edge_v, labels_v):

    labels_v2 = F.interpolate(labels_v, dout_c.shape[-1], mode='bilinear', align_corners=False)
    loss_c = bce_iou_loss_logit(dout_c,labels_v2)

    label_v_flatten1024 = labels_v.flatten(-2,-1).permute(0,2,1)
    label_v_flatten = [label_v_flatten1024]

    cps_loss = 0.0
    rrs1_loss = 0.0
    rrs2_loss = 0.0

    for i in range(len(de_list)):
        midedge = de_list[i]
        edge_vt = F.interpolate(edge_v, midedge.shape[-1], mode='bilinear', align_corners=False)
        temp_edge_loss = bce_iou_loss_logit(midedge,edge_vt) *1

        if i <= 2:
            rrs1_loss = rrs1_loss + temp_edge_loss
        else:
            rrs2_loss = rrs2_loss + temp_edge_loss

       
    for i in range(len(dmid_list)):
        midpred = dmid_list[i]
        labels_vt = F.interpolate(labels_v, midpred.shape[-1], mode='bilinear', align_corners=False)
        temp_pred_loss = bce_iou_loss_logit(midpred,labels_vt) *1

        if i <= 2:
            rrs1_loss = rrs1_loss + temp_pred_loss
        else:
            rrs2_loss = rrs2_loss + temp_pred_loss


    labels_v768 = F.interpolate(labels_v, 768, mode='bilinear', align_corners=False)
    # labels_v256 = F.interpolate(labels_v512, 384, mode='bilinear', align_corners=False)
    # labels_v128 = F.interpolate(labels_v256, 192, mode='bilinear', align_corners=False)
    # labels_v64 = F.interpolate(labels_v128, 96, mode='bilinear', align_corners=Fal768
    label_v_flatten1536 = labels_v.flatten(-2,-1).permute(0,2,1)
    label_v_flatten768 = labels_v768.flatten(-2,-1).permute(0,2,1)

    label_v_flatten = [label_v_flatten768, label_v_flatten768, label_v_flatten1536]

    for j in range(len(sideouts)):
        dp_list_temp = sideouts[j][0]
        label_v_flatten1 = label_v_flatten[j]
        pred_1d = dp_list_temp[0].squeeze(-1)
        select_idx = dp_list_temp[1]
        B = pred_1d.shape[0]
        ori_size = int(np.sqrt(label_v_flatten1.shape[1]))


        new_label_1d = label_v_flatten1.gather(1,select_idx).squeeze(-1)

        pred_iou = torch.sigmoid(pred_1d)
        inter = (pred_iou * new_label_1d)
        union = (pred_iou + new_label_1d)
        loss_wiou = 1 - (inter + 1) / (union - inter + 1)

        loss_new_pred = F.binary_cross_entropy_with_logits(pred_1d, new_label_1d,reduction='none')
        loss_new_pred = (loss_new_pred + loss_wiou)
        loss_new_pred = loss_new_pred.mean()
        if i == 0:
            rrs1_loss = rrs1_loss + loss_new_pred
        else:
            rrs2_loss = rrs2_loss + loss_new_pred


    cps_loss = cps_loss + loss_c

    myweight = [1, 1, 1]

    loss =  myweight[0]*cps_loss + myweight[1]*rrs1_loss + myweight[2]*rrs2_loss

    bceloss = loss_c 
    ssimloss = loss_c
    iouloss = loss_c 
    loss_e = (loss_c)

    return loss_c, loss, bceloss, ssimloss, iouloss, loss_e



