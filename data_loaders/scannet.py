import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
from tqdm import tqdm
sys.path.append("../")
from .data_utils import get_nearest_pose_ids

class Scannet(Dataset):
    def __init__(
        self,
        args,
        mode,
        scenes=(),
        **kwargs
    ):
        self.folder_path = os.path.join(args.rootdir, "scannet_gtpose0228/")
        assert mode in ["train", "validation", "test"]
        self.mode = mode
        self.num_source_views = args.num_source_views

        all_scenes = ('scene0079_00_tensorf','scene0316_00_tensorf','scene0553_00_tensorf','scene0653_00_tensorf','scene0000_01_tensorf','scene0158_00_tensorf','scene0521_00_tensorf','scene0616_00_tensorf')
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes


        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        print("loading {} for {}".format(scenes, mode))
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            intrinsic = np.loadtxt(os.path.join(scene_path, "intrinsics.txt"))
            pose_files = sorted(os.listdir(os.path.join(scene_path, 'pose')))
            img_files = sorted(os.listdir(os.path.join(scene_path, 'rgb')))

            train_c2w_mats = []
            train_intrinsics = []
            val_c2w_mats = []
            val_intrinsics = []

            # if mode=='train':
            train_pose_files = [x for x in pose_files if x.startswith('0_')]
            train_img_files = [x for x in img_files if x.startswith('0_')]
            val_pose_files = [x for x in pose_files if x.startswith('1_')]
            val_img_files = [x for x in img_files if x.startswith('1_')]

            for img_fname, pose_fname in tqdm(zip(train_img_files, train_pose_files),
                                                    desc=f'loading train pose {mode} ({len(train_img_files)})'):
                assert img_fname.replace('png','txt')==pose_fname
                c2w = np.loadtxt(os.path.join(scene_path, 'pose', pose_fname))  # @ cam_trans
                train_c2w_mats.append(c2w)
                train_intrinsics.append(intrinsic)
            train_c2w_mats = np.stack(train_c2w_mats)
            train_intrinsics = np.stack(train_intrinsics)

            for img_fname, pose_fname in tqdm(zip(val_img_files, val_pose_files),
                                                    desc=f'loading val pose {mode} ({len(val_img_files)})'):
                assert img_fname.replace('png','txt')==pose_fname
                c2w = np.loadtxt(os.path.join(scene_path, 'pose', pose_fname))  # @ cam_trans
                val_c2w_mats.append(c2w)
                val_intrinsics.append(intrinsic)
            val_c2w_mats = np.stack(val_c2w_mats)
            val_intrinsics = np.stack(val_intrinsics)
            

            train_rgb_files = [
                os.path.join(scene_path, 'rgb', f)
                for f in train_img_files
                if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
            ]
            val_rgb_files = [
                os.path.join(scene_path, 'rgb', f)
                for f in val_img_files
                if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
            ]
            
            f=open(os.path.join(scene_path,'near_far.txt'))
            line=f.readlines()
            near_depth, far_depth=line[0].split(',')
            near_depth = float(near_depth)
            far_depth = float(far_depth)
            f.close()
            # near_depth, far_depth = np.loadtxt(os.path.join(scene_path, 'pose', pose_fname))

            self.train_intrinsics.append(train_intrinsics)
            self.train_poses.append(train_c2w_mats)
            self.train_rgb_files.append(np.array(train_rgb_files).tolist())
            if mode=='train':
                num_render = len(train_pose_files)
                self.render_rgb_files.extend(np.array(train_rgb_files).tolist())
                self.render_intrinsics.extend([intrinsics_ for intrinsics_ in train_intrinsics])
                self.render_poses.extend([c2w_mat for c2w_mat in train_c2w_mats])
                self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
                self.render_train_set_ids.extend([i] * num_render)
            else:
                num_render = len(val_pose_files)
                self.render_rgb_files.extend(np.array(val_rgb_files).tolist())
                self.render_intrinsics.extend([intrinsics_ for intrinsics_ in val_intrinsics])
                self.render_poses.extend([c2w_mat for c2w_mat in val_c2w_mats])
                self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
                self.render_train_set_ids.extend([i] * num_render)
    def __len__(self):
        return (
            len(self.render_rgb_files) * 100000
            if self.mode == "train"
            else len(self.render_rgb_files)
        )
    
    def __getitem__(self,idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        if self.mode == "train":
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=2)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            min(self.num_source_views * subsample_factor, 34),
            tar_id=id_render,
            angular_dist_method="dist",
        )
        nearest_pose_ids = np.random.choice(
            nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False
        )

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
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

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.1])

        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }