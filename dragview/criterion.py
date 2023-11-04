import torch.nn as nn
from utils import img2mse
import torch

class Criterion(nn.Module):
    def __init__(self,load_depth):
        super().__init__()
        self.load_depth=load_depth

    def forward(self, outputs, ray_batch, scalars_to_log):
        depth_loss = 0
        """
        training criterion
        """
        pred_rgb = outputs["rgb"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]

        psnr_loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        if self.load_depth and ("depth_pred" in outputs.keys()) and ("depth" in ray_batch.keys()):
            depth_gt = ray_batch['depth']
            depth_pred = outputs['depth_pred']
            mask_depth = (depth_gt!=0)
            # print('when calculate depth loss',depth_gt.shape,torch.sum(mask_depth),mask_depth.shape, depth_gt.shape, depth_pred.shape)

            depth_loss = torch.nn.functional.l1_loss(depth_pred[mask_depth], depth_gt[mask_depth]) 
        return psnr_loss+depth_loss, scalars_to_log
