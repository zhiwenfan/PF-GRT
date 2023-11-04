# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
import numpy as np
import json
import os
import tqdm
import glob
import matplotlib as plt
from util.camera_pose_visualizer import CameraPoseVisualizer
from util.camera_parameter_loader import CameraParameterLoader

loader = CameraParameterLoader()
visualizer = CameraPoseVisualizer([4, 6], [0, 2], [3, 5])

scene_path='/anfs/gfxdisp/hanxue_nerf_data/scannet_gtpose0228/scene0079_00_tensorf'
pose_files = sorted(os.listdir(os.path.join(scene_path, 'pose')))
max_extrinsic = []
for pose_fname in pose_files:
    c2w = np.loadtxt(os.path.join(scene_path, 'pose', pose_fname))
    print(c2w.shape)
    max_extrinsic.append(c2w)
max_extrinsic=np.array(max_extrinsic)
for i in range(max_extrinsic.shape[0]):
    visualizer.extrinsic2pyramid(max_extrinsic[i], plt.cm.rainbow(i / max_extrinsic.shape[0]), 1)
visualizer.colorbar(max_extrinsic.shape[0])
visualizer.save('scannet_poses.png')   