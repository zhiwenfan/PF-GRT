
import os
import time
import torch
import torch.nn as nn
import numpy as np
import imageio 
import torch.nn.functional as F
from gnt_model_api import model as gnt_model,args
from gnt.data_loaders import dataset_dict
from gnt.projection import Projector
from gnt.sample_ray import RaySamplerSingleImage
from gnt.render_image import render_single_image
from utils import img_HWC2CHW, colorize, img2psnr, lpips, ssim
from gnt.data_loaders.base_utils import to_cuda
np.random.seed(0)

if __name__ == "__main__":
    gnt_model.switch_to_eval()
    from torch.utils.data import DataLoader
    # dataset = dataset_dict["llff"](args, "validation", scenes=('room'))
    dataset = dataset_dict["co3d"](args, "validation", scenes=('apple'))
    from gnt.data_loaders.free_pose import get_render_poses
    from gnt.data_loaders.free_pose import interpolate_render_poses
    
    # interpolate_poses = get_render_poses(dataset)
    interpolate_poses = interpolate_render_poses(dataset,view_num=120)

    loader = DataLoader(dataset, batch_size=1)
    iterator = iter(loader)
    data = next(iterator)

    device = "cuda:{}".format(0)
    projector = Projector(device=device)
    from gnt.data_loaders.data_utils import  get_nearest_pose_ids
    
    for idx , interpolate_pose in enumerate(interpolate_poses):
        render_pose = np.vstack((interpolate_pose, np.array([0,0,0,1])))
        
        data["camera"][0,-16:] = torch.from_numpy(render_pose).flatten()
        print("render pose:", data["camera"][0,-16:])

        t_start = time.time()

        # get nearest_pose_ids
        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            dataset.train_poses[0],
            args.num_source_views,
            tar_id=-1,
            angular_dist_method="dist",
        )

        src_rgbs = []
        src_cameras = []
        train_rgb_files = dataset.train_rgb_files[0]
        train_intrinsics = dataset.train_intrinsics[0]
        train_poses = dataset.train_poses[0]

        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)
        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        data["src_rgbs"] =  torch.from_numpy(src_rgbs[..., :3]).unsqueeze(0).cuda()
        data["src_cameras"]  =  torch.from_numpy(src_cameras).unsqueeze(0)

        tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
        H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
        gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
        
        target_image = gt_img.detach().cpu().numpy()
        with torch.no_grad():
            featmaps = gnt_model.feature_net(data["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ray_batch = tmp_ray_sampler.get_all()
    
            ret = render_single_image(
                    ray_sampler=tmp_ray_sampler,
                    ray_batch=ray_batch,
                    model=gnt_model,
                    projector=projector,
                    chunk_size=args.chunk_size,
                    N_samples=args.N_samples,
                    inv_uniform=args.inv_uniform,
                    det=True,
                    N_importance=args.N_importance,
                    white_bkgd=args.white_bkgd,
                    render_stride=args.render_stride,
                    featmaps=featmaps,
                    ret_alpha=args.N_importance > 0,
                    single_net=args.single_net,
                )

        rgb_coarse = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())
        rgb_coarse = rgb_coarse.permute(1, 2, 0).detach().cpu().numpy()

        out_folder = os.path.join(args.rootdir, "out", "free_camera_2")
        print("outputs will be saved to {}".format(out_folder))
        os.makedirs(out_folder, exist_ok=True)
        imageio.imwrite(os.path.join(out_folder, "test_{:03d}.png".format(idx)), rgb_coarse)
        t_end = time.time()
        print("render novel view cost {} s".format((t_end- t_start)))






