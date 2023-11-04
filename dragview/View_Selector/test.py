import os, sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
import imageio
from tqdm import tqdm, trange
from core.model import RetrievalNet as Model
from core.model import Sim_predictor as Predictor
from core.dataset import LINEMOD_SO3 as LINEMOD
from core.dataset import ref_loader_so3 as ref_loader
from core.utils import geodesic_distance
from pytorch3d.ops import sample_farthest_points

np.set_printoptions(threshold=np.inf)
np.random.seed(0)

def visual(cfg, src_img, ref_img, gt_ref_img):
    src_img = src_img.permute(1, 2, 0).cpu().detach().numpy()
    ref_img = ref_img.permute(1, 2, 0).cpu().detach().numpy()
    gt_ref_img = gt_ref_img.permute(1, 2, 0).cpu().detach().numpy()

    ref_mask = np.absolute(ref_img).sum(axis=-1, keepdims=True) > 0
    gt_ref_mask = np.absolute(gt_ref_img).sum(axis=-1, keepdims=True) > 0

    src_img = src_img * np.array(cfg['DATA']['PIXEL_STD']).reshape(1, 1, 3) \
    + np.array(cfg['DATA']['PIXEL_MEAN']).reshape(1, 1, 3)
    src_img = (255*src_img).astype(np.uint8)

    ref_img = ref_img * np.array(cfg['DATA']['PIXEL_STD']).reshape(1, 1, 3) \
    + np.array(cfg['DATA']['PIXEL_MEAN']).reshape(1, 1, 3)
    ref_img = ref_img * ref_mask
    ref_img = (255*ref_img).astype(np.uint8)

    gt_ref_img = gt_ref_img * np.array(cfg['DATA']['PIXEL_STD']).reshape(1, 1, 3) \
    + np.array(cfg['DATA']['PIXEL_MEAN']).reshape(1, 1, 3)
    gt_ref_img = gt_ref_img * gt_ref_mask
    gt_ref_img = (255*gt_ref_img).astype(np.uint8)

    h, w, _ = src_img.shape
    viz_img = np.zeros([h, 3*w, 3]).astype(np.uint8)

    viz_img[:, :w, :] = src_img
    viz_img[:, w:2*w, :] = gt_ref_img
    viz_img[:, 2*w:, :] = ref_img
    return viz_img

def val(cfg, model, predictor, device):
    model.eval()
    predictor.eval()
    print(">>>>>>>>>>>>>> Loading reference database")
    ref_database = ref_loader(cfg)

    ref_info = {}
    with torch.no_grad():
        for clsID in ref_database.ref_info.keys():
            print(">>>>>>>>>>>>>> Estimating features for ref " + clsID)
            ref_info_clsID = ref_database.load(clsID)
            anchors, anchor_indices = sample_farthest_points(ref_info_clsID["Rs"].view(1, -1, 9), K=cfg["DATA"]["ANCHOR_NUM"], random_start_point=False)
            anchors, anchor_indices = anchors[0], anchor_indices[0]

            ref_info_clsID["anchors"] = anchors.to(device)
            ref_info_clsID["indices"] = anchor_indices.to(device)
            ref_info_clsID["Rs"] = ref_info_clsID["Rs"].to(device)
            ref_info_clsID["ref_f"] = []

            ref_imgs = ref_info_clsID["imgs"]
            for i in trange(ref_imgs.shape[0]):
                ref_img = ref_imgs[i][None].to(device)
                ref_f = model(ref_img)
                ref_f = [ref_f[j].cpu().detach() for j in range(len(ref_f))]
                ref_info_clsID["ref_f"].append(ref_f)

            del ref_img, ref_imgs, ref_f
            torch.cuda.empty_cache()

            ref_info_clsID["anchor_f"] = []
            for j in range(len(ref_info_clsID["ref_f"][0])):
                ref_info_clsID["anchor_f"].append(torch.cat([ref_info_clsID["ref_f"][idx][j].to(device) for idx in ref_info_clsID["indices"]], dim=0))

            ref_info[clsID] = ref_info_clsID

    if cfg["TEST"]["VISUAL"] is True:
        if not os.path.exists(cfg["TEST"]["VISUAL_PATH"]):
            os.makedirs(cfg["TEST"]["VISUAL_PATH"])

    print(">>>>>>>>>>>>>> START TESTING")
    test_cat_acc_all, test_R_acc_all = [], []
    for cat in cfg["TEST"]["UNSEEN"]:
        objID = cfg[cfg["DATA"]["DATASET"]][cat]

        test_cat_acc, test_R_acc = test_category(cfg, model, predictor, ref_info, device, objID)

        test_cat_acc_all += [test_cat_acc]
        test_R_acc_all += [test_R_acc]

    test_cat_acc_all = np.asarray(test_cat_acc_all).mean()
    test_R_acc_all = np.asarray(test_R_acc_all).mean()

    print('All categories -- || Testing Cls Acc: %.2f || Testing R Acc: %.2f' % (test_cat_acc_all, test_R_acc_all))

    return [test_R_acc_all, test_cat_acc_all]

def iterative_retrieval(src_R, src_f, ref_info_clsID, predictor, device, max_iter=5, init_K=4096, fps_k=256, shrink_ratio=2):
    pred_sims = predictor(src_f, ref_info_clsID["anchor_f"]).squeeze(0)
    pred_index = torch.argmax(pred_sims)

    anchor = ref_info_clsID["anchors"][pred_index]

    for i in range(max_iter):
        neighbor_indices = torch.topk((anchor.view(1, 9) * ref_info_clsID["Rs"].view(-1, 9)).sum(dim=-1), k=init_K//(shrink_ratio**i), sorted=False)[1]

        if neighbor_indices.shape[0] > fps_k:
            _, fps_indices = sample_farthest_points(ref_info_clsID["Rs"][neighbor_indices].view(1, -1, 9), K=fps_k, random_start_point=False)
            neighbor_indices = neighbor_indices[fps_indices.squeeze(0)]

        ref_f = []
        for j in range(len(ref_info_clsID["ref_f"][0])):
            ref_f.append(torch.cat([ref_info_clsID["ref_f"][neighbor_indices[idx]][j].to(device) \
            for idx in range(neighbor_indices.shape[0])], dim=0))

        pred_sims = predictor(src_f, ref_f).squeeze(0)

        pred_index = neighbor_indices[torch.argmax(pred_sims)]
        pred_sim = torch.max(pred_sims)

        anchor_new = ref_info_clsID["Rs"][pred_index]

        anchor = anchor_new

    return pred_index, pred_sim

def test_category(cfg, model, predictor, ref_info, device, objID):
    dataset_test = LINEMOD(cfg, 'test', objID)
    dataset_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["WORKERS"], drop_last=False)

    print(">>>>>>>>>> TESTING DATA of %06d:" % (objID), len(dataset_loader))

    if cfg["TEST"]["VISUAL"] is True:
        filename_output = cfg["TEST"]["VISUAL_PATH"] + '/%06d.gif' % (objID)
        writer = imageio.get_writer(filename_output, mode='I', duration=1.0)

    test_cat_acc, test_R_acc, test_errs, gt_errs = [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset_loader)):
            # load data and label
            src_img, src_R, id = data
            src_img = src_img.to(device)
            src_R = src_R.to(device)

            src_f = model(src_img)

            '''Category prediction'''
            gt_clsID = "%06d" % (id)

            anchor_sims = []
            for clsID in ref_info.keys():
                pred_sims = predictor(src_f, ref_info[clsID]["anchor_f"]).squeeze(0)
                anchor_sims.append(torch.topk(pred_sims, k=3)[0].mean())
            pred_cls_index = torch.argmax(torch.stack(anchor_sims))
            pred_clsID = list(ref_info)[pred_cls_index]

            '''Fast Retrieval'''
            pred_index, pred_sim = iterative_retrieval(src_R, src_f, ref_info[pred_clsID], predictor, device, \
            max_iter=4, init_K=cfg["TEST"]["INIT_K"], fps_k=cfg["TEST"]["FPS_K"], shrink_ratio=2)

            ref_sim, ref_err = geodesic_distance(src_R, ref_info[gt_clsID]["Rs"])

            gt_index = torch.argmin(ref_err).item()
            gt_err = torch.min(ref_err).item()

            pred_err = ref_err[pred_index].item()

            cls_acc = int(pred_clsID == gt_clsID)

            test_cat_acc.append(cls_acc)
            test_errs.append(pred_err)
            test_R_acc.append(float(pred_err <= cfg["TEST"]["THR_SO3"]) * cls_acc)
            gt_errs.append(gt_err)
            if i % 20 == 0:
                if cfg["TEST"]["VISUAL"] is True:
                    ref_img_pred = ref_info[pred_clsID]["imgs"][pred_index]
                    ref_img_gt = ref_info[gt_clsID]["imgs"][gt_index]
                    viz_img = visual(cfg, src_img.squeeze(0), ref_img_pred, ref_img_gt)
                    writer.append_data(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))

        if cfg["TEST"]["VISUAL"] is True:
            writer.close()

        test_cat_acc = 100 * np.asarray(test_cat_acc).mean()
        test_R_acc = 100 * np.asarray(test_R_acc).mean()
        test_errs = np.asarray(test_errs).mean()
        gt_errs = np.asarray(gt_errs).mean()
        print('Category: %02d -- || GT Err: %.2f || Test Err: %.2f || Testing Classifcation Acc: %.2f || Testing R1: %.2f' % \
        (objID, gt_errs, test_errs, test_cat_acc, test_R_acc))
        return test_cat_acc, test_R_acc

def test(cfg, device):
    if not os.path.exists(cfg["TRAIN"]["WORKING_DIR"]):
        os.makedirs(cfg["TRAIN"]["WORKING_DIR"])
    logname = os.path.join(cfg["TRAIN"]["WORKING_DIR"], 'testing_log.txt')

    if cfg["TEST"]["VISUAL"] is True:
        if not os.path.exists(cfg["TEST"]["VISUAL_PATH"]):
            os.makedirs(cfg["TEST"]["VISUAL_PATH"])

    print(">>>>>>>>>>>>>> LOADING NETWORK")
    model = Model(cfg).to(device)
    checkpoint = torch.load(cfg["TRAIN"]["WORKING_DIR"] + "checkpoint_1.pth", map_location=lambda storage, loc: storage.cuda())
    pretrained_dict = checkpoint['state_dict']
    best_epoch = checkpoint["epoch"]
    model.load_state_dict(pretrained_dict)
    model.eval()

    predictor = Predictor(cfg).to(device)
    checkpoint = torch.load(cfg["TRAIN"]["WORKING_DIR"] + "checkpoint_2.pth", map_location=lambda storage, loc: storage.cuda())
    pretrained_dict = checkpoint['state_dict']
    predictor.load_state_dict(pretrained_dict)
    predictor.eval()

    print(">>>>>>>>>>>>>> Network trained on epoch %03d has been loaded" % (best_epoch))

    print(">>>>>>>>>>>>>> Loading reference database")
    ref_database = ref_loader(cfg)
    ref_info = {}

    with torch.no_grad():
        for clsID in ref_database.ref_info.keys():
            print(">>>>>>>>>>>>>> Estimating features for ref " + clsID)
            ref_info_clsID = ref_database.load(clsID)
            anchors, anchor_indices = sample_farthest_points(ref_info_clsID["Rs"].reshape(1, -1, 9), K=cfg["DATA"]["ANCHOR_NUM"], random_start_point=False)
            anchors, anchor_indices = anchors[0], anchor_indices[0]

            ref_info_clsID["anchors"] = anchors.to(device)
            ref_info_clsID["indices"] = anchor_indices.to(device)
            ref_info_clsID["Rs"] = ref_info_clsID["Rs"].to(device)
            ref_info_clsID["ref_f"] = []

            ref_imgs = ref_info_clsID["imgs"]
            for i in trange(ref_imgs.shape[0]):
                ref_img = ref_imgs[i][None].to(device)
                ref_f = model(ref_img)
                ref_f = [ref_f[j].cpu().detach() for j in range(len(ref_f))]
                ref_info_clsID["ref_f"].append(ref_f)

            del ref_img, ref_imgs, ref_f
            torch.cuda.empty_cache()

            ref_info_clsID["anchor_f"] = []
            for j in range(len(ref_info_clsID["ref_f"][0])):
                ref_info_clsID["anchor_f"].append(torch.cat([ref_info_clsID["ref_f"][idx][j].to(device) for idx in ref_info_clsID["indices"]], dim=0))

            ref_info[clsID] = ref_info_clsID


    test_cat_acc_all, test_R_acc_all = [], []
    for cat in cfg["TEST"]["UNSEEN"]:
        objID = cfg[cfg["DATA"]["DATASET"]][cat]
        test_cat_acc, test_R_acc = test_category(cfg, model, predictor, ref_info, device, objID)

        test_cat_acc_all += [test_cat_acc]
        test_R_acc_all += [test_R_acc]

        with open(logname, 'a') as f:
            f.write('objID: %02d -- || Testing Cls: %.2f || Testing R Acc: %.2f\n' % (objID, test_cat_acc, test_R_acc))
        f.close()

    test_cat_acc_all = np.asarray(test_cat_acc_all).mean()
    test_R_acc_all = np.asarray(test_R_acc_all).mean()

    with open(logname, 'a') as f:
        f.write('All categories -- || Testing Cls Acc: %.2f || Testing R Acc: %.2f\n' % (test_cat_acc_all, test_R_acc_all))
    f.close()

if __name__ == '__main__':

    with open("./objects.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    cfg["TRAIN"]["BS"] = 1

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    test(cfg, device)
