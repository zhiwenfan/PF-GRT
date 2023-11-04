import os
import io
import json
import trimesh
import random
import numpy as np
import cv2
import torch
from pytorch3d.ops import sample_farthest_points
import transforms3d

def save_json(path, meta):
    meta_dump = json.dumps(meta)
    f = open(path, 'w')
    f.write(meta_dump)
    f.close()

def load_json(path):
    f = open(path, 'r')
    info = json.load(f)
    return info

def load_checkpoint(model, optimizer, pth_file):
    """load state and network weights"""
    checkpoint = torch.load(pth_file, map_location=lambda storage, loc: storage.cuda())
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    try:
        best_acc = checkpoint['best_acc']
    except:
        best_acc = 0
        print("Best acc was not saved")
    print('Previous weight loaded')
    return model, optimizer, start_epoch, best_acc

def load_bop_meshes(model_path):
    # load meshes
    meshFiles = [f for f in os.listdir(model_path) if f.endswith('.ply')]
    meshFiles.sort()
    meshes = []
    objID_2_clsID = {}

    for i in range(len(meshFiles)):
        mFile = meshFiles[i]
        objId = int(os.path.splitext(mFile)[0][4:])
        objID_2_clsID[str(objId)] = i
        meshes.append(trimesh.load(model_path + mFile))

    return meshes, objID_2_clsID

def load_bbox_3d(jsonFile):
    with open(jsonFile, 'r') as f:
        bbox_3d = json.load(f)
    return bbox_3d

def get_single_bop_annotation(img_path, objID_2_clsID):
    # add attributes to function, for fast loading
    if not hasattr(get_single_bop_annotation, "dir_annots"):
        get_single_bop_annotation.dir_annots = {}
    #
    img_path = img_path.strip()
    cvImg = cv2.imread(img_path)
    height, width, _ = cvImg.shape
    #
    gt_dir, tmp, imgName = img_path.rsplit('/', 2)
    assert(tmp == 'rgb')
    imgBaseName, _ = os.path.splitext(imgName)
    im_id = int(imgBaseName)
    #
    camera_file = gt_dir + '/scene_camera.json'
    gt_file = gt_dir + "/scene_gt.json"
    # gt_info_file = gt_dir + "/scene_gt_info.json"
    gt_mask_visib = gt_dir + "/mask_visib/"

    if gt_dir in get_single_bop_annotation.dir_annots:
        gt_json, cam_json = get_single_bop_annotation.dir_annots[gt_dir]
    else:
        gt_json = json.load(open(gt_file))
        # gt_info_json = json.load(open(gt_info_file))
        cam_json = json.load(open(camera_file))
        #
        get_single_bop_annotation.dir_annots[gt_dir] = [gt_json, cam_json]

    if str(im_id) in cam_json:
        annot_camera = cam_json[str(im_id)]
    else:
        annot_camera = cam_json[("%06d" % im_id)]
    if str(im_id) in gt_json:
        annot_poses = gt_json[str(im_id)]
    else:
        annot_poses = gt_json[("%06d" % im_id)]
    # annot_infos = gt_info_json[str(im_id)]

    objCnt = len(annot_poses)
    K = np.array(annot_camera['cam_K']).reshape(3,3)

    class_ids = []
    # bbox_objs = []
    rotations = []
    translations = []
    merged_mask = np.zeros((height, width), np.uint8) # segmenation masks
    for i in range(objCnt):
        mask_vis_file = gt_mask_visib + ("%06d_%06d.png" %(im_id, i))
        mask_vis = cv2.imread(mask_vis_file, cv2.IMREAD_UNCHANGED)
        #
        # bbox = annot_infos[i]['bbox_visib']
        # bbox = annot_infos[i]['bbox_obj']
        # contourImg = cv2.rectangle(contourImg, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,0,255))
        # cv2.imshow(str(i), mask_vis)
        #
        R = np.array(annot_poses[i]['cam_R_m2c']).reshape(3,3)
        T = np.array(annot_poses[i]['cam_t_m2c']).reshape(3,1)
        obj_id = annot_poses[i]['obj_id']
        cls_id = objID_2_clsID[str(obj_id)]
        #
        # bbox_objs.append(bbox)
        class_ids.append(cls_id)
        rotations.append(R)
        translations.append(T)
        # compose segmentation labels
        merged_mask[mask_vis==255] = (i+1)

    return K, merged_mask, class_ids, rotations, translations

def remap_pose(srcK, srcR, srcT, pt3d, dstK):
    ptCnt = len(pt3d)
    pts = np.matmul(srcK, np.matmul(srcR, pt3d.transpose()) + srcT)
    xs = pts[0] / (pts[2] + 1e-12)
    ys = pts[1] / (pts[2] + 1e-12)
    xy2d = np.concatenate((xs.reshape(-1,1),ys.reshape(-1,1)), axis=1)

    #retval, rot, trans, inliers = cv2.solvePnPRansac(pt3d, xy2d, dstK, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=5.0)
    retval, rot, trans = cv2.solvePnP(pt3d.reshape(ptCnt,1,3), xy2d.reshape(ptCnt,1,2), dstK, None, flags=cv2.SOLVEPNP_EPNP)
    if retval:
        newR = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
        newT = trans.reshape(-1, 1)

        return newR, newT
    else:
        print('Error in pose remapping!')
        return srcR, srcT
        
def pose_symmetry_handling(R, sym_types):
    if len(sym_types) == 0:
        return R, T

    assert(len(sym_types) % 2 == 0)
    itemCnt = int(len(sym_types) / 2)

    for i in range(itemCnt):
        axis = sym_types[2*i]
        mod = sym_types[2*i + 1] * np.pi / 180
        if axis == 'X':
            ai, aj, ak = transforms3d.euler.mat2euler(R, axes='sxyz')
            ai = 0 if mod == 0 else (ai % mod)
            R = transforms3d.euler.euler2mat(ai, aj, ak, axes='sxyz')
        elif axis == 'Y':
            ai, aj, ak = transforms3d.euler.mat2euler(R, axes='syzx')
            ai = 0 if mod == 0 else (ai % mod)
            R = transforms3d.euler.euler2mat(ai, aj, ak, axes='syzx')
        elif axis == 'Z':
            ai, aj, ak = transforms3d.euler.mat2euler(R, axes='szyx')
            ai = 0 if mod == 0 else (ai % mod)
            R = transforms3d.euler.euler2mat(ai, aj, ak, axes='szyx')
        else:
            print("symmetry axis should be 'X', 'Y' or 'Z'")
            assert(0)
    return R.astype(np.float32)

def geodesic_distance(src_R, ref_R):
    sim = (torch.sum(src_R.view(-1, 9) * ref_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
    geo_dis = torch.arccos(sim) * 180. / np.pi
    return sim, geo_dis

def farthest_point_sample_6d(Rs, K, random_start_point):
    x_col = Rs.view(-1, 3, 3)[:, :, 0]
    y_col = Rs.view(-1, 3, 3)[:, :, 1]

    x_col = x_col / torch.norm(x_col, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    y_col = y_col / torch.norm(y_col, p=2, dim=-1, keepdim=True).clamp(min=1e-8)

    Rs_6d = torch.cat([x_col, y_col], dim=-1)

    _, anchor_indices = sample_farthest_points(Rs_6d[None], K=K, random_start_point=random_start_point)
    anchor_indices = anchor_indices.squeeze(0)

    anchor = Rs[anchor_indices]

    return anchor, anchor_indices

def distort_hsv(img, h_ratio, s_ratio, v_ratio):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
    h = img_hsv[:, :, 0].astype(np.float32)  # hue
    s = img_hsv[:, :, 1].astype(np.float32)  # saturation
    v = img_hsv[:, :, 2].astype(np.float32)  # value
    a = random.uniform(-1, 1) * h_ratio + 1
    b = random.uniform(-1, 1) * s_ratio + 1
    c = random.uniform(-1, 1) * v_ratio + 1
    h *= a
    s *= b
    v *= c
    img_hsv[:, :, 0] = h if a < 1 else h.clip(None, 179)
    img_hsv[:, :, 1] = s if b < 1 else s.clip(None, 255)
    img_hsv[:, :, 2] = v if c < 1 else v.clip(None, 255)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

def distort_noise(img, noise_ratio=0):
    # add noise
    noisesigma = random.uniform(0, noise_ratio)
    gauss = np.random.normal(0, noisesigma, img.shape) * 255
    img = img + gauss

    img[img > 255] = 255
    img[img < 0] = 0

    return np.uint8(img)

def distort_smooth(img, smooth_ratio=0):
    # add smooth
    smoothsigma = random.uniform(0, smooth_ratio)
    res = cv2.GaussianBlur(img, (7, 7), smoothsigma, cv2.BORDER_DEFAULT)
    return res
