# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
import numpy as np
import json
import os
import tqdm
import matplotlib as plt
from util.camera_pose_visualizer import CameraPoseVisualizer
from util.camera_parameter_loader import CameraParameterLoader
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose
root_path = '/anfs/gfxdisp/hanxue_nerf_data/nerf_synthetic/chair'
with open(os.path.join(root_path, f'transforms_train.json'), 'r') as f:
    transform = json.load(f)
frames = transform["frames"]
poses = []
for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
    
    
    pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
    pose = nerf_matrix_to_ngp(pose, scale=0.8, offset=[0, 0, 0])
    poses.append(pose)

    f_path_old = os.path.join(root_path, f['file_path'])
    if '.' not in os.path.basename(f_path_old):
        f_path =f_path_old+ '.png' # so silly...
    if (pose[1,3]>2.4 or pose[0,3]<0 or abs(pose[2,3])>2) and pose[0,3]<1.6:
        cmd = "cp "+f_path+" "+"./1_"+str(pose[0,3])+"_"+str(pose[1,3])+"_"+str(pose[2,3])+".png"
    else:
        cmd = "cp "+f_path+" "+"./0_"+str(pose[0,3])+"_"+str(pose[1,3])+"_"+str(pose[2,3])+".png"
    print(cmd)
    os.system(cmd)


loader = CameraParameterLoader()
visualizer = CameraPoseVisualizer([-4, 4], [0, 4], [-4, 4])
# file = '/home/hl589/geopards/poses_bounds.npy'
# # file='poses.npy'
# poses=np.load(file)
poses=np.array(poses)
# max_extrinsic = []
# for i in range(poses.shape[0]):
#     mat_extrinsic = np.concatenate([poses[i][:12].reshape((3,4)), np.array([[0, 0, 0, 1]])], axis=0)
#     max_extrinsic.append(mat_extrinsic)
# max_extrinsic = np.array(max_extrinsic)
print(poses.shape,np.max(poses[:,0,3]),np.min(poses[:,0,3]),np.max(poses[:,1,3]),np.min(poses[:,1,3]),np.max(poses[:,2,3]),np.min(poses[:,2,3]))
for i in range(poses.shape[0]):
    # mat_extrinsic = np.concatenate([poses[i][:12].reshape((3,4)), np.array([[0, 0, 0, 1]])], axis=0)

    visualizer.extrinsic2pyramid(poses[i], plt.cm.rainbow(i / poses.shape[0]), 1)
visualizer.colorbar(poses.shape[0])
visualizer.save('render_poses.png')   