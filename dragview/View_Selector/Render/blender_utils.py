import bpy
from mathutils import Matrix
import numpy as np
import math
import os
import cv2

def matrixToNumpyArray(mat):
    new_mat = np.array([[mat[0][0],mat[0][1],mat[0][2],mat[0][3]],
                        [mat[1][0],mat[1][1],mat[1][2],mat[1][3]],
                        [mat[2][0],mat[2][1],mat[2][2],mat[2][3]],
                        [mat[3][0],mat[3][1],mat[3][2],mat[3][3]]])
    return new_mat

def numpyArrayToMatrix(array):
    mat = Matrix(((array[0,0],array[0,1], array[0,2], array[0,3]),
                (array[1,0],array[1,1], array[1,2], array[1,3]),
                (array[2,0],array[2,1], array[2,2], array[2,3]) ,
                (array[3,0],array[3,1], array[3,2], array[3,3])))
    return mat
    
def save_visual(rgb, mask, output_path, euler):
    path = os.path.join(output_path, "rgb", str(euler[0]) + "_" + str(euler[1]) + "_" + str(euler[2]) + ".png")
    cv2.imwrite(path, rgb)
    path = os.path.join(output_path, "mask", str(euler[0]) + "_" + str(euler[1]) + "_" + str(euler[2]) + ".png")
    cv2.imwrite(path, mask)

def draw_bounding_box(cvImg, kpts_2d, color, thickness):
    x = np.int32(kpts_2d[0])
    y = np.int32(kpts_2d[1])
    bbox_lines = [0, 1, 0, 2, 0, 4, 5, 1, 5, 4, 6, 2, 6, 4, 3, 2, 3, 1, 7, 3, 7, 5, 7, 6]
    for i in range(12):
        id1 = bbox_lines[2*i]
        id2 = bbox_lines[2*i+1]
        cvImg = cv2.line(cvImg, (x[id1],y[id1]), (x[id2],y[id2]), color, thickness=thickness, lineType=cv2.LINE_AA)
    return cvImg

# we could also define the camera matrix
# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_calibration_matrix_K_from_blender(camera):
    f_in_mm = camera.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera.sensor_width
    sensor_height_in_mm = camera.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camera.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = Matrix(((alpha_u, skew, u_0),
                (0, alpha_v, v_0),
                (0, 0, 1)))

    return K


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(camera):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Use matrix_world instead to account for all constraints
    location, rotation = camera.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((R_world2cv[0][:] + (T_world2cv[0],),
                 R_world2cv[1][:] + (T_world2cv[1],),
                 R_world2cv[2][:] + (T_world2cv[2],)))
    return RT


def get_3x4_P_matrix_from_blender(camera):
    K = get_calibration_matrix_K_from_blender(camera.data)
    RT = get_3x4_RT_matrix_from_blender(camera)
    return K*RT


def get_K_P_from_blender(camera):
    K = get_calibration_matrix_K_from_blender(camera.data)
    RT = get_3x4_RT_matrix_from_blender(camera)
    return {"K": np.asarray(K, dtype=np.float32), "RT": np.asarray(RT, dtype=np.float32)}

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return q1, q2, q3, q4


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx * cx + ty * cy, -1), 1)
    # roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return q1, q2, q3, q4


def camRotQuaternion(cx, cy, cz, theta):
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return q1, q2, q3, q4


def quaternionProduct(qx, qy):
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e
    return q1, q2, q3, q4

def obj_location(dist, azi, ele):
    ele = math.radians(ele)
    azi = math.radians(azi)
    x = dist * math.cos(azi) * math.cos(ele)
    y = dist * math.sin(azi) * math.cos(ele)
    z = dist * math.sin(ele)
    return x, y, z

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return x, y, z
