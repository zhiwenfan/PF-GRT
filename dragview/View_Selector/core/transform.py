import numpy as np
import torch
import cv2
import random
import glob
import os
from core.utils import distort_hsv, distort_noise, distort_smooth

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img

    def __repr__(self):
        format_str = self.__class__.__name__ + '('
        for t in self.transforms:
            format_str += '\n'
            format_str += f'    {t}'
        format_str += '\n)'

        return format_str

class RandomHSV:
    def __init__(self, h_ratio, s_ratio, v_ratio):
        self.h_ratio = h_ratio
        self.s_ratio = s_ratio
        self.v_ratio = v_ratio
    def __call__(self, img):
        img = distort_hsv(img, self.h_ratio, self.s_ratio, self.v_ratio)
        return img

class RandomNoise:
    def __init__(self, noise_ratio):
        self.noise_ratio = noise_ratio
    def __call__(self, img):
        img = distort_noise(img, self.noise_ratio)
        return img

class RandomSmooth:
    def __init__(self, smooth_ratio):
        self.smooth_ratio = smooth_ratio
    def __call__(self, img):
        img = distort_smooth(img, self.smooth_ratio)
        return img

class ToTensor:
    def __call__(self, img):
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        img = img - np.array(self.mean).reshape(1,1,3)
        img = img / np.array(self.std).reshape(1,1,3)
        return img

class RandomBackground:
    def __init__(self, background_dir):
        self.background_files = []
        try:
            if os.path.exists(background_dir):
                png_files = glob.glob(os.path.join(background_dir, '*.png'))
                jpg_files = glob.glob(os.path.join(background_dir, '*.jpg'))
                self.background_files += png_files
                self.background_files += jpg_files
        except:
            print("can not read background directory, remains empty")
            pass
        print("Number of background images: %d" % len(self.background_files))

    def __call__(self, img, mask):
        if len(self.background_files) > 0:
            if img.shape[2] == 4:
                img = self.merge_background_alpha(img, self.get_a_random_background())
            else:
                img = self.merge_background_mask(img, self.get_a_random_background(), mask)
        else:
            img = img[:,:,0:3]
        return img

    def get_a_random_background(self):
        backImg = None
        while backImg is None:
            backIdx = random.randint(0, len(self.background_files) - 1)
            img_path = self.background_files[backIdx]
            try:
                backImg = cv2.imread(img_path)
                if backImg is None:
                    raise RuntimeError('load image error')
            except:
                print('Error in loading background image: %s' % img_path)
                backImg = None
        return backImg

    def merge_background_alpha(self, foreImg, backImg):
        assert(foreImg.shape[2] == 4)
        forergb = foreImg[:, :, :3]
        alpha = foreImg[:, :, 3] / 255.0
        if forergb.shape != backImg.shape:
            backImg = cv2.resize(backImg, (foreImg.shape[1], foreImg.shape[0]))
        alpha = np.repeat(alpha, 3).reshape(foreImg.shape[0], foreImg.shape[1], 3)
        mergedImg = np.uint8(backImg * (1 - alpha) + forergb * alpha)
        # backImg[alpha > 128] = forergb[alpha > 128]
        return mergedImg

    def merge_background_mask(self, foreImg, backImg, maskImg):
        forergb = foreImg[:, :, :3]
        if forergb.shape != backImg.shape:
            backImg = cv2.resize(backImg, (foreImg.shape[1], foreImg.shape[0]))
        alpha = np.ones((foreImg.shape[0], foreImg.shape[1], 3), np.float32)
        alpha[maskImg == 0] = 0
        mergedImg = np.uint8(backImg * (1 - alpha) + forergb * alpha)
        # backImg[alpha > 128] = forergb[alpha > 128]
        return mergedImg

class RandomOcclusion:
    """
    randomly erasing holes
    ref: https://arxiv.org/abs/1708.04896
    """
    def __init__(self, prob = 0):
        self.prob = prob

    def __call__(self, img):
        if self.prob > 0:
            height, width, _ = img.shape
            bw = width
            bh = height
            x1 = 0
            x2 = width
            y1 = 0
            y2 = height
            if random.uniform(0, 1) <= self.prob and bw > 2 and bh > 2:
                bb_size = bw*bh
                size = random.uniform(0.02, 0.7) * bb_size
                ratio = random.uniform(0.5, 2.0)
                ew = int(np.sqrt(size * ratio))
                eh = int(np.sqrt(size / ratio))
                ecx = random.uniform(x1, x2)
                ecy = random.uniform(y1, y2)
                esx = int(np.clip((ecx - ew/2 + 0.5), 0, width-1))
                esy = int(np.clip((ecy - eh/2 + 0.5), 0, height-1))
                eex = int(np.clip((ecx + ew/2 + 0.5), 0, width-1))
                eey = int(np.clip((ecy + eh/2 + 0.5), 0, height-1))
                targetshape = img[esy:eey, esx:eex, :].shape
                img[esy:eey, esx:eex, :] = np.random.randint(256, size=targetshape)
        return img, target
