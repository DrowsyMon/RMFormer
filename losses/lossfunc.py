import losses.pytorch_iou
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def bce_iou_loss(pred,target):
    bce_loss = nn.BCELoss(size_average=True)
    iou_loss = losses.pytorch_iou.IOU(size_average=True)


    bce_out = bce_loss(pred,target)
    iou_out = iou_loss(pred,target)
    loss = bce_out + iou_out

    return loss
def bce_iou_loss_logit(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def loss_vit_simple_edgelist(dout,dout_c,dmid_list,de_list, sideouts,\
    edge_v, labels_v):

    lossp = bce_iou_loss_logit(dout,labels_v)
    labels_v2 = F.interpolate(labels_v, dout_c.shape[-1], mode='bilinear', align_corners=False)
    loss_c = bce_iou_loss_logit(dout_c,labels_v2)

    total_temp_pred_loss = 0.0
    total_temp_edge_loss = 0.0
    total_new_pred_loss = 0.0
        
    for i in range(len(dmid_list)):
        midpred = dmid_list[i]
        labels_vt = F.interpolate(labels_v, midpred.shape[-1], mode='bilinear', align_corners=False)
        if i != len(dmid_list)-1:
            temp_pred_loss = bce_iou_loss_logit(midpred,labels_vt) *1
            total_temp_pred_loss = total_temp_pred_loss+temp_pred_loss

    for i in range(len(de_list)):
        midedge = de_list[i]
        edge_vt = F.interpolate(edge_v, midedge.shape[-1], mode='bilinear', align_corners=False)

        temp_edge_loss = bce_iou_loss_logit(midedge,edge_vt) *1
        total_temp_edge_loss = total_temp_edge_loss+temp_edge_loss

    labels_v512 = F.interpolate(labels_v, 768, mode='bilinear', align_corners=False)
    label_v_flatten1024 = labels_v.flatten(-2,-1).permute(0,2,1)
    label_v_flatten512 = labels_v512.flatten(-2,-1).permute(0,2,1)

    label_v_flatten = [label_v_flatten512, label_v_flatten512, label_v_flatten1024]


    for j in range(len(sideouts)):
        dp_list_temp = sideouts[j][0]
        label_v_flatten1 = label_v_flatten[j]
        pred_1d = dp_list_temp[0].squeeze(-1)
        select_idx = dp_list_temp[1]
        new_label_1d = label_v_flatten1.gather(1,select_idx).squeeze(-1)

        pred_iou = torch.sigmoid(pred_1d)
        inter = (pred_iou * new_label_1d)
        union = (pred_iou + new_label_1d)
        loss_wiou = 1 - (inter + 1) / (union - inter + 1)

        loss_new_pred = F.binary_cross_entropy_with_logits(pred_1d, new_label_1d,reduction='none')
        loss_new_pred = (loss_new_pred + loss_wiou)
        loss_new_pred = loss_new_pred.mean()

        total_new_pred_loss =(total_new_pred_loss+loss_new_pred)


    loss =  lossp + loss_c + (total_temp_pred_loss)*1 + total_temp_edge_loss + total_new_pred_loss

    return loss_c, loss