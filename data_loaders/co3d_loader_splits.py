# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import json
import gzip
from skimage.transform import resize
import json

sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids


def read_cameras(pose_file):
    basedir = os.path.dirname(pose_file)
    with open(pose_file, "r") as fp:
        meta = json.load(fp)

    camera_angle_x = float(meta["camera_angle_x"])
    rgb_files = []
    c2w_mats = []

    img = imageio.imread(os.path.join(basedir, meta["frames"][0]["file_path"] + ".png"))
    H, W = img.shape[:2]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    intrinsics = get_intrinsics_from_hwf(H, W, focal)

    for i, frame in enumerate(meta["frames"]):
        rgb_file = os.path.join(basedir, meta["frames"][i]["file_path"][2:] + ".png")
        rgb_files.append(rgb_file)
        c2w = np.array(frame["transform_matrix"])
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats.append(c2w_opencv)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics] * len(meta["frames"])), c2w_mats


def get_intrinsics_from_hwf(h, w, focal):
    return np.array(
        [[focal, 0, 1.0 * w / 2, 0], [0, focal, 1.0 * h / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )


def similarity_from_cameras(c2w, fix_rot=False):
    """
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    if fix_rot:
        R_align = np.eye(3)
        R = np.eye(3)
    else:
        R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    return transform, scale


class CO3DSplitDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = os.path.join(args.rootdir, "data/co3d/")
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        if mode == "validation":
            mode = "val"
        assert mode in ["train", "val", "test"]
        self.mode = mode  # train / test / val
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip

        all_cats = os.listdir(self.folder_path)
        all_scenes_ = [os.listdir(os.path.join(self.folder_path, cat)) for cat in all_cats]
        all_scenes = []
        for i, cat in enumerate(all_scenes_):
            temp = []
            for scene in cat:
                if ".json" in scene:
                    continue
                if ".jgz" in scene:
                    continue
                temp.append(scene)
            all_scenes.append(temp)

        self.render_rgb_files = []
        self.render_poses = []
        self.render_intrinsics = []
        self.render_img_sizes = []
        self.render_train_set_ids = []
        self.train_rgb_files = []
        self.train_poses = []
        self.train_intrinsics = []
        self.train_img_sizes = []

        max_image_dim = 800
        cam_scale_factor = 1.5
        cam_trans = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32))

        cntr = 0
        for i, cat in enumerate(all_cats):
            json_path = os.path.join(self.folder_path, cat, "frame_annotations.jgz")
            with gzip.open(json_path, "r") as fp:
                all_frames_data = json.load(fp)

            frame_data = {}
            for temporal_data in all_frames_data:
                if temporal_data["sequence_name"] not in frame_data:
                    frame_data[temporal_data["sequence_name"]] = []
                frame_data[temporal_data["sequence_name"]].append(temporal_data)

            data_list = json.load(open(os.path.join(self.folder_path, cat, "set_lists.json"), "r"))
            splits = {}
            for data_split in ["train_known", "train_unseen", "test_unseen"]:
                splits[data_split] = {}
                for scene in data_list[data_split]:
                    scene_id, _, img_f = scene
                    if scene_id not in splits[data_split]:
                        splits[data_split][scene_id] = []
                    splits[data_split][scene_id].append(img_f)
            for scene in all_scenes[i]:
                scene_img_fs, scene_intrinsics, scene_extrinsics, scene_image_sizes = (
                    [],
                    [],
                    [],
                    [],
                )
                for (j, frame) in enumerate(frame_data[scene]):
                    img_f = os.path.join(self.folder_path, frame["image"]["path"])

                    H, W = frame["image"]["size"]
                    max_hw = max(H, W)
                    approx_scale = max_image_dim / max_hw

                    if approx_scale < 1.0:
                        H2 = int(approx_scale * H)
                        W2 = int(approx_scale * W)
                    else:
                        H2 = H
                        W2 = W

                    image_size = np.array([H2, W2])
                    fxy = np.array(frame["viewpoint"]["focal_length"])
                    cxy = np.array(frame["viewpoint"]["principal_point"])
                    R = np.array(frame["viewpoint"]["R"])
                    T = np.array(frame["viewpoint"]["T"])

                    min_HW = min(W2, H2)
                    image_size_half = np.array([W2 * 0.5, H2 * 0.5], dtype=np.float32)
                    scale_arr = np.array([min_HW * 0.5, min_HW * 0.5], dtype=np.float32)
                    fxy_x = fxy * scale_arr
                    prp_x = np.array([W2 * 0.5, H2 * 0.5], dtype=np.float32) - cxy * scale_arr
                    cxy = (image_size_half - prp_x) / image_size_half
                    fxy = fxy_x / image_size_half

                    scale_arr = np.array([W2 * 0.5, H2 * 0.5], dtype=np.float32)
                    focal = fxy * scale_arr
                    prp = -1.0 * (cxy - 1.0) * scale_arr

                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3:] = -R @ T[..., None]
                    pose = pose @ cam_trans
                    # c2w = pose
                    # w2c_blender = np.linalg.inv(c2w)
                    # w2c_opencv = w2c_blender
                    # w2c_opencv[1:3] *= -1
                    # c2w_opencv = np.linalg.inv(w2c_opencv)

                    intrinsic = np.array(
                        [
                            [focal[0], 0.0, prp[0], 0.0],
                            [0.0, focal[1], prp[1], 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    )
                    scene_img_fs.append(img_f)
                    scene_intrinsics.append(intrinsic)
                    # scene_extrinsics.append(c2w_opencv)
                    scene_extrinsics.append(pose)
                    scene_image_sizes.append(image_size)

                i_all = np.arange(len(scene_img_fs))
                i_test = []
                for idx, img_f in enumerate(scene_img_fs):
                    scene_id = img_f.replace(self.folder_path, "").split("/")[1]
                    if scene_id not in splits["train_unseen"].keys():
                        continue
                    if img_f.replace(self.folder_path, "") not in splits["train_unseen"][scene_id]:
                        continue
                    i_test.append(idx)
                i_test = np.array(i_test)
                i_train = np.array([idx for idx in i_all if not idx in i_test])
                if mode == "train":
                    i_render = i_train
                else:
                    i_render = i_test
                if len(i_render) == 0:
                    continue

                scene_intrinsics = np.array(scene_intrinsics)
                scene_extrinsics = np.array(scene_extrinsics)
                scene_image_sizes = np.array(scene_image_sizes)

                # H_median, W_median = np.median(
                #     np.stack([image_size for image_size in scene_image_sizes]), axis=0
                # )

                # H_inlier = np.abs(scene_image_sizes[:, 0] - H_median) / H_median < 0.1
                # W_inlier = np.abs(scene_image_sizes[:, 1] - W_median) / W_median < 0.1
                # inlier = np.logical_and(H_inlier, W_inlier)
                # dists = np.linalg.norm(
                #     scene_extrinsics[:, :3, 3] - np.median(scene_extrinsics[:, :3, 3], axis=0), axis=-1
                # )
                # med = np.median(dists)
                # good_mask = dists < (med * 5.0)
                # inlier = np.logical_and(inlier, good_mask)

                # if inlier.sum() != 0:
                #     scene_intrinsics = scene_intrinsics[inlier]
                #     scene_extrinsics = scene_extrinsics[inlier]
                #     scene_image_sizes = scene_image_sizes[inlier]
                #     scene_img_fs = [scene_img_fs[i] for i in range(len(inlier)) if inlier[i]]

                # scene_extrinsics = np.stack(scene_extrinsics)
                T, sscale = similarity_from_cameras(scene_extrinsics)
                scene_extrinsics = T @ scene_extrinsics
                scene_extrinsics[:, :3, 3] *= sscale * cam_scale_factor

                # i_all = np.arange(len(scene_img_fs))
                # i_test = i_all[::10]
                # i_train = np.array([i for i in i_all if not i in i_test])
                # if mode == "train":
                #     i_render = i_train
                # else:
                #     i_render = i_test

                self.train_intrinsics.append(scene_intrinsics[i_train])
                self.train_poses.append(scene_extrinsics[i_train])
                self.train_rgb_files.append(np.array(scene_img_fs)[i_train].tolist())
                self.train_img_sizes.append(scene_image_sizes[i_train].tolist())
                num_render = len(i_render)

                self.render_rgb_files.extend(np.array(scene_img_fs)[i_render].tolist())
                self.render_intrinsics.extend(
                    [intrinsics_ for intrinsics_ in scene_intrinsics[i_render]]
                )
                self.render_poses.extend([c2w_mat for c2w_mat in scene_extrinsics[i_render]])
                self.render_img_sizes.extend(
                    [img_size for img_size in scene_image_sizes[i_render]]
                )
                self.render_train_set_ids.extend([cntr] * num_render)
                cntr += 1
        print(f"loaded {len(self.render_rgb_files)} for {mode}")

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx]
        render_img_size = self.render_img_sizes[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files, train_intrinsics, train_poses, train_img_sizes = (
            self.train_rgb_files[train_set_id],
            self.train_intrinsics[train_set_id],
            self.train_poses[train_set_id],
            self.train_img_sizes[train_set_id],
        )

        if self.mode == "train":
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])
            num_select = self.num_source_views + np.random.randint(low=-2, high=2)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        rgb = resize(rgb, render_img_size)

        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), render_intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            self.num_source_views * subsample_factor,
            tar_id=id_render,
            angular_dist_method="vector",
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
            src_rgb = resize(src_rgb, render_img_size)

            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            train_img_size = train_img_sizes[id]
            train_intrinsics_[0] *= render_img_size[1] / train_img_size[1]
            train_intrinsics_[1] *= render_img_size[0] / train_img_size[0]
            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)
        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        # near_depth = 0.1
        # far_depth = 1.0
        min_ratio = 0.1
        origin_depth = np.linalg.inv(render_pose)[2, 3]
        max_radius = 4
        near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
        far_depth = origin_depth + max_radius
        if near_depth > 0 and far_depth > 0 and far_depth > near_depth:
            pass
        else:
            near_depth = 0.1
            far_depth = 6.0
        depth_range = torch.tensor([near_depth, far_depth])

        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
