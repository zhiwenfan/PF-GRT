import pyrender
import numpy as np
import cv2

def render_objects(meshes, ids, poses, K, w, h):
    assert(K[0][1] == 0 and K[1][0] == 0 and K[2][0] ==0 and K[2][1] == 0 and K[2][2] == 1)
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    objCnt = len(ids)
    assert(len(poses) == objCnt)

    # set background with 0 alpha, important for RGBA rendering
    scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 1.0]), ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
    # pyrender.Viewer(scene, use_raymond_lighting=True)
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera = pyrender.IntrinsicsCamera(fx=fx,fy=fy,cx=cx,cy=cy,znear=0.05,zfar=100000)
    camera_pose = np.eye(4)
    # reverse the direction of Y and Z, check: https://pyrender.readthedocs.io/en/latest/examples/cameras.html
    camera_pose[1][1] = -1
    camera_pose[2][2] = -1
    scene.add(camera, pose=camera_pose)
    #light = pyrender.SpotLight(color=np.ones(3), intensity=4.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    #light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=camera_pose)
    for i in range(objCnt):
        clsId = int(ids[i])
        mesh = pyrender.Mesh.from_trimesh(meshes[clsId])
        H = np.zeros((4,4))
        H[0:3] = poses[i][0:3]
        H[3][3] = 1.0
        scene.add(mesh, pose=H)
    # pyrender.Viewer(scene, use_raymond_lighting=True)

    r = pyrender.OffscreenRenderer(w, h)
    # flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY
    #flags = pyrender.RenderFlags.OFFSCREEN
    #flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.RGBA
    #color, depth = r.render(scene, flags=flags)
    color, depth = r.render(scene)

    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR) # RGB to BGR (for OpenCV)
    #color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA) # RGBA to BGRA (for OpenCV)

    return color, depth
