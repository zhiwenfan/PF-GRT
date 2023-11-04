import os
import glob
import matplotlib as plt
from util.camera_pose_visualizer import CameraPoseVisualizer
from util.camera_parameter_loader import CameraParameterLoader
import numpy as np



# def plot_framewise():
#     loader = CameraParameterLoader()
#     visualizer = CameraPoseVisualizer([-6, 6], [-6, 6], [-6, 6])

#     render_poses=np.load(os.path.join('/home/protago/hanxue/','render_poses.npy'))
#     max_extrinsic = []
#     for i in range(render_poses.shape[0]):
#         mat_extrinsic = np.concatenate([render_poses[i][:,:4], np.array([[0, 0, 0, 1]])], axis=0)
#         max_extrinsic.append(mat_extrinsic)
#     max_extrinsic = np.array(max_extrinsic)
#     print(max_extrinsic.shape,np.max(max_extrinsic[:,0,3]),np.min(max_extrinsic[:,0,3]),np.max(max_extrinsic[:,1,3]),np.min(max_extrinsic[:,1,3]),np.max(max_extrinsic[:,2,3]),np.min(max_extrinsic[:,2,3]))
#     for i in range(render_poses.shape[0]):
#         mat_extrinsic = np.concatenate([render_poses[i][:,:4], np.array([[0, 0, 0, 1]])], axis=0)
#         visualizer.extrinsic2pyramid(mat_extrinsic, plt.cm.rainbow(i / render_poses.shape[0]), 1)
#     visualizer.colorbar(render_poses.shape[0])
#     visualizer.save('render_poses.png')



# import json
# def plot_framewise():
#     loader = CameraParameterLoader()
#     visualizer = CameraPoseVisualizer([-6, 6], [-6, 6], [-6, 6])
#     file = '/home/protago/hanxue/cam_mats_cropped.json'
#     # render_poses=np.load(os.path.join('/home/protago/hanxue/','cam_mats_cropped.json'))
#     with open(file, 'r') as f:
#         param_cam = json.load(f)
#     max_extrinsic = []
#     for i in range(len(param_cam)):
#         mat_extrinsic = np.concatenate([np.array(param_cam[i]['E'])[:3,:4], np.array([[0, 0, 0, 1]])], axis=0)
#         max_extrinsic.append(mat_extrinsic)
#     max_extrinsic = np.array(max_extrinsic)
#     print(max_extrinsic.shape,np.max(max_extrinsic[:,0,3]),np.min(max_extrinsic[:,0,3]),np.max(max_extrinsic[:,1,3]),np.min(max_extrinsic[:,1,3]),np.max(max_extrinsic[:,2,3]),np.min(max_extrinsic[:,2,3]))
#     for i in range(len(param_cam)):
#         mat_extrinsic = np.concatenate([np.array(param_cam[i]['E'])[:3,:4], np.array([[0, 0, 0, 1]])], axis=0)
#         visualizer.extrinsic2pyramid(mat_extrinsic, plt.cm.rainbow(i / len(param_cam)), 1)
#     visualizer.colorbar(len(param_cam))
#     visualizer.save('render_poses.png')    
# plot_framewise()

import json
loader = CameraParameterLoader()
visualizer = CameraPoseVisualizer([-6, 6], [-6, 6], [-6, 6])
# file = '/anfs/gfxdisp/hanxue_nerf_data/nerf_llff_data/calci_museum_geopards_re/poses_bounds.npy'
file = '/anfs/gfxdisp/hanxue_nerf_data/nerf_llff_data/calci_museum_whale/poses_bounds.npy'

# file='poses.npy'
poses=np.load(file)
def alongz(degree):
    cos = np.cos(degree*(np.pi/180))
    sin = np.sin(degree*(np.pi/180))
    return np.array([[cos,sin,0],[-sin,cos,0],[0,0,1]])
def alongx(degree):
    cos = np.cos(degree*(np.pi/180))
    sin = np.sin(degree*(np.pi/180))
    return np.array([[1,0,0],[0,cos,sin],[0,-sin,cos]])
def alongy(degree):
    cos = np.cos(degree*(np.pi/180))
    sin = np.sin(degree*(np.pi/180))
    return np.array([[cos,0,-sin],[0,1,0],[sin,0,cos]])

max_extrinsic = []
# for i in range(poses.shape[0]):
for i in range(75,poses.shape[0]):
    pose = poses[i][:12].reshape((3,4))
    pose[:,3]=pose[:,3]*2
    # if i<60:
    #     pose[1,3]=pose[1,3]+4.8
    
    rotate=alongx(270)
    pose = np.matmul(rotate,pose)
    if i>=135:
        pose[1,3]=pose[1,3]-1.0 #1.73
    # if i>=60:
        pose[2,3]=pose[2,3]-4.88
    # rotate=alongy(45)
    # pose = np.matmul(rotate,pose)
    
    mat_extrinsic = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    max_extrinsic.append(mat_extrinsic)
max_extrinsic = np.array(max_extrinsic)
print(max_extrinsic.shape,np.max(max_extrinsic[:60,0,3]),np.min(max_extrinsic[:60,0,3]),np.max(max_extrinsic[:60,1,3]),np.min(max_extrinsic[:60,1,3]),np.max(max_extrinsic[:60,2,3]),np.min(max_extrinsic[:60,2,3]))
print(max_extrinsic.shape,np.max(max_extrinsic[60:,0,3]),np.min(max_extrinsic[60:,0,3]),np.max(max_extrinsic[60:,1,3]),np.min(max_extrinsic[60:,1,3]),np.max(max_extrinsic[60:,2,3]),np.min(max_extrinsic[60:,2,3]))
# for i in range(poses.shape[0]):
for i in range(max_extrinsic.shape[0]):
    # pose = poses[i][:12].reshape((3,4))
    # # rotate=alongx(270)
    # # pose = np.matmul(rotate,pose)
    # rotate=alongy(45)
    # pose = np.matmul(rotate,pose)
    # pose[:,3]=pose[:,3]*3
    # mat_extrinsic = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    # visualizer.extrinsic2pyramid(max_extrinsic[i], plt.cm.rainbow(i / poses.shape[0]), 1)
    if i>=60:
        color = plt.cm.rainbow(1 / 2)
        visualizer.extrinsic2pyramid(max_extrinsic[i], color, 1)
    else:
        color = plt.cm.rainbow(0/ 2)
        visualizer.extrinsic2pyramid(max_extrinsic[i], color, 1)
# visualizer.colorbar(poses.shape[0])
# visualizer.newcolorbar(60,max_extrinsic.shape[0])
list_labels=['train views','test views']
visualizer.customize_legend(list_labels)
visualizer.save('train_render_poses.png') 
# visualizer.save('render_poses.png')   
# visualizer.save('render_poses.pdf')   
