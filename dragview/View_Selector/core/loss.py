import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_euler_angles
from core.utils import geodesic_distance

def weighted_infoNCE_loss_func(ref_sim_pos, ref_sim_rand, ref_R_positive, ref_R_rand, src_R, id, tau=0.1):
    with torch.no_grad():
        same_cat = ((id[:, None] - id[None]) == 0).float()

        gt_sim_pos = (torch.sum(src_R.view(-1, 1, 9) * ref_R_positive.view(1, -1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        gt_dis_pos = torch.arccos(gt_sim_pos) / np.pi
        gt_dis_pos = gt_dis_pos * same_cat + gt_dis_pos.new_ones(gt_dis_pos.shape) * (1 - same_cat)

        gt_sim_rand = (torch.sum(src_R.view(-1, 1, 9) * ref_R_rand.view(1, -1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        gt_dis_rand = torch.arccos(gt_sim_rand) / np.pi
        gt_dis_rand = gt_dis_rand * same_cat + gt_dis_rand.new_ones(gt_dis_rand.shape) * (1 - same_cat)

    postive_term = (torch.diag(ref_sim_pos) / tau).exp() * torch.diag(gt_dis_pos)
    ref_sim = torch.cat([gt_dis_pos * (ref_sim_pos / tau).exp(), gt_dis_rand * (ref_sim_rand / tau).exp()], dim=-1)

    nce_loss = (-torch.log(postive_term / (torch.sum(ref_sim, dim=-1)))).mean()

    return nce_loss
