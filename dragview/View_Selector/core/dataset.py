import torch.utils.data as data
from torch.utils.data import DataLoader
import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import cv2
import glob
import pickle
import imutils
from tqdm import trange
from pytorch3d.ops import sample_farthest_points
import core.transform as transform
from core.utils import pose_symmetry_handling, geodesic_distance

np.set_printoptions(threshold=np.inf)

def resize_pad(im, dim):
    _, h, w = im.shape
    im = transforms.functional.resize(im, int(dim * min(w, h) / max(w, h)))
    left = int(np.ceil((dim - im.shape[2]) / 2))
    right = int(np.floor((dim - im.shape[2]) / 2))
    top = int(np.ceil((dim - im.shape[1]) / 2))
    bottom = int(np.floor((dim - im.shape[1]) / 2))
    im = transforms.functional.pad(im, (left, top, right, bottom))

    return im

def bbx_resize(bbx, img_w, img_h):
    w, h = bbx[2] - bbx[0], bbx[3] - bbx[1]
    dim = max(w, h)

    left = int(np.ceil((dim - w) / 2))
    right = int(np.floor((dim - w) / 2))
    top = int(np.ceil((dim - h) / 2))
    bottom = int(np.floor((dim - h) / 2))

    bbx[0] = max(bbx[0] - left, 0)
    bbx[1] = max(bbx[1] - top, 0)
    bbx[2] = min(bbx[2] + right, img_w)
    bbx[3] = min(bbx[3] + bottom, img_h)

    return bbx

def crop(img, bbx):
    if len(img.shape) < 4:
        crop_img = img[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])]
    else:
        crop_img = [img[i, int(bbx[i, 1]):int(bbx[i, 3]), int(bbx[i, 0]):int(bbx[i, 2])] for i in range(img.shape[0])]
    return crop_img

class ref_image_loader(data.Dataset):
    def __init__(self, cfg, ref_info):
        self.ref_info = ref_info
        self.cfg = cfg
        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    self.cfg['DATA']['PIXEL_MEAN'],
                    self.cfg['DATA']['PIXEL_STD']),
            ]
        )
        self.mask_trans = transforms.ToTensor()

    def __len__(self):
        return len(self.ref_info["paths"])

    def __getitem__(self, idx):
        ref_bbx = np.array(self.ref_info["bbxs"][idx])
        ref_img = cv2.imread(self.ref_info["paths"][idx])
        ref_mask = cv2.imread(self.ref_info["paths"][idx].split('rgb')[0] + 'mask' + \
        self.ref_info["paths"][idx].split('rgb')[1], 0).astype(np.uint8)

        ref_img = crop(ref_img, ref_bbx)
        ref_mask = crop(ref_mask[..., None], ref_bbx)

        ref_img = self.trans(ref_img) * self.mask_trans(ref_mask)
        ref_img = resize_pad(ref_img, self.cfg["DATA"]["CROP_SIZE"])

        return ref_img

class ref_f_loader(data.Dataset):
    def __init__(self, ref_f):
        self.ref_f = ref_f

    def __len__(self):
        return len(self.ref_f)

    def __getitem__(self, idx):
        ref = self.ref_f[idx]
        return ref

## reference loader
class ref_loader_so3():
    def __init__(self, cfg):
        self.cfg = cfg
        self.ref_path = glob.glob(os.path.join(self.cfg['DATA']['META_DIR'], self.cfg['DATA']['RENDER_DIR'], '*.pkl'))

        self.ref_info = {}
        clsIDs = []
        for ref_file in self.ref_path:
            clsID = ref_file.split('/')[-1].split('.')[0]
            with open(ref_file, 'rb') as f:
                self.ref_info[clsID] = pickle.load(f)
            f.close()
            clsIDs.append(clsID)

    def load(self, clsID):
        ref_info_clsID = self.ref_info[clsID]

        ref_paths = ref_info_clsID["paths"]
        ref_bbxs = np.array(ref_info_clsID["bbxs"])
        ref_Rs = np.array(ref_info_clsID["Rs"])

        if clsID in self.cfg["LINEMOD"]["SYMMETRIC_OBJ"].keys():
            ref_Rs = [pose_symmetry_handling(ref_Rs[i], self.cfg["LINEMOD"]["SYMMETRIC_OBJ"][clsID]) for i in range(ref_Rs.shape[0])]
            ref_Rs = torch.from_numpy(np.asarray(ref_Rs))
        else:
            ref_Rs = torch.from_numpy(ref_Rs)

        ref_database = ref_image_loader(self.cfg, ref_info_clsID)
        dataset_loader = DataLoader(ref_database, batch_size=128, shuffle=False, num_workers=self.cfg["TRAIN"]["WORKERS"], drop_last=False)

        ref_imgs = []
        for i, ref_img in enumerate(dataset_loader):
            ref_imgs.append(ref_img)

        ref_imgs = torch.cat(ref_imgs, dim=0)
        ref_info_clsID = {}
        ref_info_clsID['imgs'] = ref_imgs
        ref_info_clsID['Rs'] = ref_Rs
        return ref_info_clsID


class LINEMOD_SO3(data.Dataset):
    def __init__(self, cfg, mode, clsID):
        self.cfg = cfg
        self.mode = mode
        self.unseen_cat = [cfg["LINEMOD"][cat] for cat in cfg["TEST"]["UNSEEN"]]

        if self.mode == 'train':
            self.src_path = glob.glob(os.path.join(self.cfg['DATA']['META_DIR'], 'src_images_test_pkl', '*.pkl'))
            self.ref_path = glob.glob(os.path.join(self.cfg['DATA']['META_DIR'], self.cfg['DATA']['RENDER_DIR'], '*.pkl'))

            for cat in self.unseen_cat:
                self.src_path.remove(os.path.join(self.cfg['DATA']['META_DIR'], 'src_images_test_pkl', '%06d.pkl' % (cat)))
                self.ref_path.remove(os.path.join(self.cfg['DATA']['META_DIR'], self.cfg['DATA']['RENDER_DIR'], '%06d.pkl' % (cat)))

            self.trans = transforms.Compose(
                [
                    transform.RandomHSV(0.2, 0.5, 0.5),
                    transform.RandomNoise(0.1),
                    transform.RandomSmooth(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        self.cfg['DATA']['PIXEL_MEAN'],
                        self.cfg['DATA']['PIXEL_STD']),
                ]
            )
            self.mask_trans = transforms.ToTensor()

        elif self.mode == 'test':
            self.src_path = [os.path.join(self.cfg['DATA']['META_DIR'], 'src_images_test_pkl', '%06d.pkl' % (clsID))]

            self.trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        self.cfg['DATA']['PIXEL_MEAN'],
                        self.cfg['DATA']['PIXEL_STD']),
                ]
            )
        else:
            raise RuntimeError('Unsupported mode')

        print(">>>>>>> Loading source and reference data")
        self.src_info, self.ref_info = {}, {}
        src_imgs, src_masks, src_Ks, src_bbxs, src_ids = [], [], [], [], []
        ref_info = {}

        for i in range(len(self.src_path)):
            with open(self.src_path[i], 'rb') as f:
                src_info_i = pickle.load(f)
            f.close()

            if src_info_i["ids"][0] in cfg["LINEMOD"]["SYMMETRIC_OBJ"].keys():
                src_info_i["Rs"] = [pose_symmetry_handling(src_info_i["Rs"][j], self.cfg["LINEMOD"]["SYMMETRIC_OBJ"][src_info_i["ids"][0]]) for j in range(len(src_info_i["Rs"]))]
                src_info_i["Rs"] = np.asarray(src_info_i["Rs"])
            else:
                src_info_i["Rs"] = np.asarray(src_info_i["Rs"])

            src_Rs = src_info_i["Rs"] if i == 0 else np.concatenate([src_Rs, src_info_i["Rs"]], axis=0)

            src_imgs += src_info_i["imgs"]
            src_masks += src_info_i["masks"]
            src_Ks += src_info_i["Ks"]
            src_bbxs += src_info_i["bbxs"]
            src_ids += src_info_i["ids"]

            if self.mode == 'train':
                with open(self.ref_path[i], 'rb') as f:
                    ref_info_i = pickle.load(f)
                f.close()

                if src_info_i["ids"][0] in cfg["LINEMOD"]["SYMMETRIC_OBJ"].keys():
                    ref_info_i["Rs"] = [pose_symmetry_handling(ref_info_i["Rs"][j], self.cfg["LINEMOD"]["SYMMETRIC_OBJ"][src_info_i["ids"][0]]) for j in range(len(ref_info_i["Rs"]))]
                    ref_info_i["Rs"] = np.asarray(ref_info_i["Rs"])
                else:
                    ref_info_i["Rs"] = np.asarray(ref_info_i["Rs"])

                self.ref_info[src_info_i["ids"][0]] = ref_info_i

        indices = np.random.permutation(len(src_imgs))
        self.src_info["imgs"] = np.asarray(src_imgs)[indices]
        self.src_info["masks"] = np.asarray(src_masks)[indices]
        self.src_info["Ks"] = np.asarray(src_Ks)[indices]
        self.src_info["Rs"] = np.asarray(src_Rs)[indices]
        self.src_info["bbxs"] = np.asarray(src_bbxs)[indices]
        self.src_info["ids"] = np.asarray(src_ids)[indices]

    def load_sample(self, index, ref_paths, ref_bbxs):
        path = ref_paths[index].split('rgb')

        img = cv2.imread(ref_paths[index])
        mask = cv2.imread(path[0] + 'mask' + path[1], 0)

        img = crop(img, ref_bbxs[index])
        mask = crop(mask, ref_bbxs[index])

        return img, mask

    def sampling(self, src_R, ref_Rs, ref_bbxs, ref_paths):
        ### anchor sample
        _, anchor_indices = sample_farthest_points(ref_Rs.view(1, -1, 9), K=self.cfg["DATA"]["ANCHOR_NUM"], random_start_point=True)
        anchor_indices = anchor_indices[0]

        gt_sim, _ = geodesic_distance(src_R, ref_Rs[anchor_indices])
        anchor_index = anchor_indices[torch.argmax(gt_sim)]
        anchor_img, anchor_mask = self.load_sample(anchor_index, ref_paths, ref_bbxs)

        ### positive sample
        gt_sim, _ = geodesic_distance(src_R, ref_Rs)
        pos_index = torch.argmax(gt_sim)
        pos_img, pos_mask = self.load_sample(pos_index, ref_paths, ref_bbxs)

        ### negative sample
        random_index = torch.randperm(gt_sim.shape[0])[0]
        random_img, random_mask = self.load_sample(random_index, ref_paths, ref_bbxs)

        ref_R = torch.stack([ref_Rs[anchor_index], ref_Rs[pos_index], ref_Rs[random_index]])

        return [anchor_img, pos_img, random_img], [anchor_mask, pos_mask, random_mask], ref_R

    def __len__(self):
        return self.src_info["imgs"].shape[0]

    def __getitem__(self, idx):
        src_img = self.src_info["imgs"][idx].astype(np.uint8)
        src_mask = self.src_info["masks"][idx].astype(np.uint8)
        bbx = self.src_info["bbxs"][idx]
        id = int(self.src_info["ids"][idx])
        K = torch.from_numpy(self.src_info["Ks"][idx]).float()
        src_R = torch.from_numpy(self.src_info["Rs"][idx]).float()

        if self.mode == 'train':
            ref_info = self.ref_info[self.src_info["ids"][idx]]
            ref_paths = ref_info["paths"]
            ref_bbxs = np.array(ref_info["bbxs"])
            ref_Rs = torch.from_numpy(ref_info["Rs"])

            if random.random() > 0.5 and self.cfg['TRAIN']['ROTATION_AG'] is True:
                r = max(-60, min(60, torch.randn(1) * 30))
                src_img = imutils.rotate_bound(src_img, angle=-r)
                src_mask = imutils.rotate_bound(src_mask, angle=-r)
                r = r * np.pi / 180.
                delta_R = torch.tensor([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]]).float()
                src_R = torch.matmul(torch.inverse(delta_R), src_R)

                bbx = np.where(src_mask>0)
                x_min = int(np.min(bbx[1]))
                y_min = int(np.min(bbx[0]))
                x_max = int(np.max(bbx[1]))
                y_max = int(np.max(bbx[0]))
                bbx = np.asarray([x_min, y_min, x_max, y_max])

            ref_img, ref_mask, ref_R = self.sampling(src_R, ref_Rs, ref_bbxs, ref_paths)
            bbx = bbx_resize(bbx, src_mask.shape[1], src_mask.shape[0])

            src_img = crop(src_img, bbx)
            src_mask = crop(src_mask, bbx)

            src_img = self.trans(src_img)

            src_img = resize_pad(src_img, self.cfg["DATA"]["CROP_SIZE"])
            ref_img = [resize_pad(self.trans(ref_img[i]) * self.mask_trans(ref_mask[i][..., None]), self.cfg["DATA"]["CROP_SIZE"])\
            for i in range(len(ref_img))]

            return src_img, ref_img, src_R, ref_R, id

        else:
            bbx = bbx_resize(bbx, src_mask.shape[1], src_mask.shape[0])
            src_img = crop(src_img, bbx)
            src_img = self.trans(src_img)
            src_img = resize_pad(src_img, self.cfg["DATA"]["CROP_SIZE"])

            return src_img, src_R, id
