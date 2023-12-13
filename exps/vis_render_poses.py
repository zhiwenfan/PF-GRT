import matplotlib as plt
from util.camera_pose_visualizer import CameraPoseVisualizer
from util.camera_parameter_loader import CameraParameterLoader
import numpy as np

if __name__ == "__main__":
    loader = CameraParameterLoader()
    visualizer = CameraPoseVisualizer([-6, 6], [-6, 6], [-6, 6])
    file = 'data/nerf_llff_data/calci_museum_whale/poses_bounds.npy'
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
    for i in range(75,poses.shape[0]):
        pose = poses[i][:12].reshape((3,4))
        pose[:,3]=pose[:,3]*2
        rotate=alongx(270)
        pose = np.matmul(rotate,pose)
        if i>=135:
            pose[1,3]=pose[1,3]-1.0 #1.73
            pose[2,3]=pose[2,3]-4.88
        mat_extrinsic = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
        max_extrinsic.append(mat_extrinsic)
    max_extrinsic = np.array(max_extrinsic)
    print(max_extrinsic.shape,np.max(max_extrinsic[:60,0,3]),np.min(max_extrinsic[:60,0,3]),np.max(max_extrinsic[:60,1,3]),np.min(max_extrinsic[:60,1,3]),np.max(max_extrinsic[:60,2,3]),np.min(max_extrinsic[:60,2,3]))
    print(max_extrinsic.shape,np.max(max_extrinsic[60:,0,3]),np.min(max_extrinsic[60:,0,3]),np.max(max_extrinsic[60:,1,3]),np.min(max_extrinsic[60:,1,3]),np.max(max_extrinsic[60:,2,3]),np.min(max_extrinsic[60:,2,3]))

    for i in range(max_extrinsic.shape[0]):
        if i>=60:
            color = plt.cm.rainbow(1 / 2)
            visualizer.extrinsic2pyramid(max_extrinsic[i], color, 1)
        else:
            color = plt.cm.rainbow(0/ 2)
            visualizer.extrinsic2pyramid(max_extrinsic[i], color, 1)

    list_labels=['train views','test views']
    visualizer.customize_legend(list_labels)
    visualizer.save('train_render_poses.png') 

