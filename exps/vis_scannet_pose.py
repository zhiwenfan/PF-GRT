
import numpy as np
import os
import matplotlib as plt
from util.camera_pose_visualizer import CameraPoseVisualizer
from util.camera_parameter_loader import CameraParameterLoader

if __main__ == "__main__":
    loader = CameraParameterLoader()
    visualizer = CameraPoseVisualizer([4, 6], [0, 2], [3, 5])
    scene_path='data/scannet_gtpose0228/scene0079_00_tensorf'
    pose_files = sorted(os.listdir(os.path.join(scene_path, 'pose')))
    max_extrinsic = []
    for pose_fname in pose_files:
        c2w = np.loadtxt(os.path.join(scene_path, 'pose', pose_fname))
        max_extrinsic.append(c2w)
    max_extrinsic=np.array(max_extrinsic)
    for i in range(max_extrinsic.shape[0]):
        visualizer.extrinsic2pyramid(max_extrinsic[i], plt.cm.rainbow(i / max_extrinsic.shape[0]), 1)
    visualizer.colorbar(max_extrinsic.shape[0])
    visualizer.save('scannet_poses.png')   