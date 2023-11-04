import os
import sys
import yaml
import torch
import math
import numpy as np
import cv2
from tqdm import tqdm
import json
import trimesh
import argparse
import glob
import pickle
from tqdm import trange
from bpy_render import render_ply
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, rotation_6d_to_matrix

sys.path.append('../')
from core.utils import (
    load_bop_meshes,
    load_bbox_3d,
    get_single_bop_annotation,
    remap_pose,
)
np.set_printoptions(threshold=np.inf)
np.random.seed(0)

def sample_6d(num):
    samples = []
    for i in range(num):
        x = np.asarray([np.random.normal() for j in range(3)]).squeeze()
        y = np.asarray([np.random.normal() for j in range(3)]).squeeze()

        x = x / max(np.linalg.norm(x, ord=2), 1e-8)
        y = y / max(np.linalg.norm(y, ord=2), 1e-8)

        samples.append(np.concatenate([x, y], axis=-1))

    return torch.from_numpy(np.asarray(samples)).float()

def src_image_generate(cfg, mode):
    if mode == 'test':
        seq_paths = ['../data/linemod_zhs/000001_test.txt','../data/linemod_zhs/000002_test.txt',\
        '../data/linemod_zhs/000004_test.txt','../data/linemod_zhs/000005_test.txt',\
        '../data/linemod_zhs/000006_test.txt','../data/linemod_zhs/000008_test.txt',\
        '../data/linemod_zhs/000009_test.txt','../data/linemod_zhs/000010_test.txt',\
        '../data/linemod_zhs/000011_test.txt','../data/linemod_zhs/000012_test.txt',\
        '../data/linemod_zhs/000013_test.txt','../data/linemod_zhs/000014_test.txt',\
        '../data/linemod_zhs/000015_test.txt'\
        ]
    elif mode == 'train':
        seq_paths = ['../data/linemod_zhs/000001_train.txt','../data/linemod_zhs/000002_train.txt',\
        '../data/linemod_zhs/000004_train.txt','../data/linemod_zhs/000005_train.txt',\
        '../data/linemod_zhs/000006_train.txt','../data/linemod_zhs/000008_train.txt',\
        '../data/linemod_zhs/000009_train.txt','../data/linemod_zhs/000010_train.txt',\
        '../data/linemod_zhs/000011_train.txt','../data/linemod_zhs/000012_train.txt',\
        '../data/linemod_zhs/000013_train.txt','../data/linemod_zhs/000014_train.txt',\
        '../data/linemod_zhs/000015_train.txt'
        ]
    else:
        raise RuntimeError("Unsupported mode")

    img_files = {}
    for seq_path in seq_paths:
        dataDir = os.path.split(seq_path)[0]
        scene = seq_path.split('/')[-1].split('.')[0]
        with open(seq_path, 'r') as f:
            img_file = f.readlines()
            img_file = [dataDir + '/' + x.strip() for x in img_file]
            img_files[scene] = img_file

    meshes, objID_2_clsID = load_bop_meshes(cfg['DATA']['MESH_DIR'])

    bbox = load_bbox_3d(cfg["DATA"]["BBOX_FILE"])

    if not os.path.exists("../data/src_images_" + mode + "_pkl/"):
        os.makedirs("../data/src_images_" + mode + "_pkl/")

    for key in img_files.keys():
        print("Generating src image for " + key.split('_')[0])
        Ks, Rs, Ts, bbxs, imgs, masks, depths, ids, kpts = [], [], [], [], [], [], [], [], []
        src_info = {}
        # Load image
        for idx in trange(len(img_files[key])):
            try:
                img = cv2.imread(img_files[key][idx], cv2.IMREAD_UNCHANGED)  # BGR(A)
                if img is None:
                    raise RuntimeError('load image error')
                #
                if img.dtype == np.uint16:
                    img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0)).astype(np.uint8)
                #
                if len(img.shape) == 2:
                    # convert gray to 3 channels
                    img = np.repeat(img.reshape(img.shape[0], img.shape[1], 1), 3, axis=2)

                elif img.shape[2] == 4:
                    # having alpha
                    tmpBack = (img[:,:,3] == 0)
                    img[:,:,0:3][tmpBack] = 255 # white background
            except:
                print('image %s not found' % img_path)
                return None

            # Load labels (BOP format)
            height, width, _ = img.shape

            K, merged_mask, class_ids, rotations, translations = get_single_bop_annotation(img_files[key][idx], objID_2_clsID)

            for i in range(len(class_ids)):
                if (merged_mask==i+1).sum() == 0:
                    continue
                bbx = np.where(merged_mask==i+1)
                x_min = int(np.min(bbx[1]))
                y_min = int(np.min(bbx[0]))
                x_max = int(np.max(bbx[1]))
                y_max = int(np.max(bbx[0]))

                h, w = y_max - y_min, x_max - x_min

                x_min = max(x_min - 0.0*w, 0)
                y_min = max(y_min - 0.0*h, 0)
                x_max = min(x_max + 0.0*w, width)
                y_max = min(y_max + 0.0*h, height)

                mask = (merged_mask==i+1).astype(np.uint8)

                K = np.asarray(K)
                R = rotations[i]
                T = translations[i]

                dst_K = np.asarray(cfg["DATA"]["INTERNAL_K"]).reshape(3, 3)
                points = np.asarray(bbox[class_ids[i]])
                R, T = remap_pose(K, R, T, points, dst_K)

                pose = np.concatenate([R.reshape(3, 3), T.reshape(3, 1)], axis=-1)

                _, depth = render_objects([meshes[int(key.split('_')[0])-1]], [0], [pose], dst_K, \
                cfg["DATA"]["INTERNAL_WIDTH"], cfg["DATA"]["INTERNAL_HEIGHT"])

                Ks.append(dst_K.reshape(3, 3))
                Rs.append(R.reshape(3, 3))
                Ts.append(T.reshape(3, 1))
                bbxs.append(np.asarray([x_min, y_min, x_max, y_max]))
                ids.append(key.split('_')[0])
                imgs.append(img)
                masks.append(mask)
                depths.append(depth)

        src_info["Ks"] = Ks
        src_info["Rs"] = Rs
        src_info["Ts"] = Ts
        src_info["imgs"] = imgs
        src_info["masks"] = masks
        src_info["depths"] = depths
        src_info["bbxs"] = bbxs
        src_info["ids"] = ids

        with open("../data/src_images_" + mode + "_pkl/" + key.split('_')[0] + ".pkl", "wb") as f:
            pickle.dump(src_info, f)
            f.close()

def reference_generation(cfg, mode, num):
    print("-----------Loading Renderer------------")
    src_info_paths = glob.glob(os.path.join("../data/src_images_test_pkl/", '*.pkl'))
    # Load the obj and ignore the textures and materials.

    if not os.path.exists(os.path.join(cfg["RENDER"]["OUTPUT_PATH"])):
        os.makedirs(os.path.join(cfg["RENDER"]["OUTPUT_PATH"]))

    ## continuous sample
    print("Number of rendered images:", num)
    samples = sample_6d(num)

    sample_R = rotation_6d_to_matrix(samples)
    sample_euler = matrix_to_euler_angles(sample_R, 'ZXZ') * 180. / np.pi
    sample_pose = [np.concatenate([sample_euler[i].numpy(), [0, 0, cfg["RENDER"]["CAM_DIST"]]], axis=-1) \
    for i in range(sample_euler.shape[0])]

    for src_info_path in src_info_paths:
        idx = int(src_info_path.split('/')[-1].split('.')[0])
        print("Now Processing obj: %06d" % (idx))
        output_path = os.path.join(cfg["RENDER"]["OUTPUT_PATH"], "%06d" % (idx))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not os.path.exists(os.path.join(output_path, "rgb")):
            os.makedirs(os.path.join(output_path, "rgb"))

        if not os.path.exists(os.path.join(output_path, "mask")):
            os.makedirs(os.path.join(output_path, "mask"))

        with open(src_info_path, "rb") as f:
            meta_info = pickle.load(f)
        f.close()

        obj = os.path.join(cfg["DATA"]["MESH_DIR"], "obj_%06d.ply" % (idx))

        print("Perform image rendering")
        ref_bbxs = render_ply(obj, output_path, sample_pose, shape=[cfg["DATA"]["INTERNAL_WIDTH"], cfg["DATA"]["INTERNAL_HEIGHT"]], \
        light_main=50, light_add=5, normalize=True, forward="X", up="Z", texture=True)

        ref_paths, ref_eulers, ref_Rs = [], [], []
        for j in range(num):
            ref_paths.append(os.path.join(output_path[1:], "rgb", str(sample_pose[j][0]) + "_" + str(sample_pose[j][1]) + "_" + str(sample_pose[j][2]) + ".png"))
            ref_eulers.append(sample_pose[j][:3])
            ref_Rs.append(sample_R[j].numpy())

        ref_info = {}

        ref_info["paths"] = ref_paths
        ref_info["eulers"] = ref_eulers
        ref_info["Rs"] = ref_Rs
        ref_info["bbxs"] = ref_bbxs

        res_path = os.path.join(cfg["RENDER"]["OUTPUT_PATH"], src_info_path.split('/')[-1])
        with open(res_path, "wb") as f:
            pickle.dump(ref_info, f)
            f.close()

if __name__ == '__main__':
    with open("../objects.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    print("---------- image generation------------")
    src_image_generate(cfg, 'train')
    src_image_generate(cfg, 'test')
    reference_generation(cfg, 'test', cfg["RENDER"]["NUM"])
