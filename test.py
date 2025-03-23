import os
import scipy.spatial
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from skimage import filters
from tqdm import tqdm

import options
from util import calc

parser = options.get_parser()
args = parser.parse_args()

thresh_low = args.test_thresh_low
thresh_high = args.test_thresh_high

criterion_sig = nn.Sigmoid()
criterion_softmax = nn.Softmax(dim=1)

# BGR
# epithelial red
# inflammatory blue
# stromal green
color_set = {0: (0, 0, 255), 1: (255, 118, 0), 2: (0, 255, 0)}

def visualize_union(model_teacher, device_teacher, model_student, device_student, test_loader, n_classes,
                    checkpoints_save_path):
    img_path = os.path.join(checkpoints_save_path, 'img')
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    with torch.no_grad():
        n_classes_out = n_classes + 1

        model_teacher.eval()
        model_student.eval()

        tp_count_all = np.zeros(n_classes_out)
        fp_count_all = np.zeros(n_classes_out)
        fn_count_all = np.zeros(n_classes_out)
        for i, (img, gt_dmap, gt_dots, img_name) in tqdm(enumerate(test_loader)):
            img_name = img_name[0]
            vutils.save_image(img, os.path.join(img_path, img_name))
            img_gt = cv2.imread(os.path.join(img_path, img_name))
            img_et = cv2.imread(os.path.join(img_path, img_name))
            img_gt_et = cv2.imread(os.path.join(img_path, img_name))

            # Get the detection ground truth maps from the classes ground truth maps
            gt_dots_det = gt_dots.max(1)[0]

            # Convert ground truth maps to numpy arrays
            gt_dots = gt_dots.detach().cpu().numpy()
            gt_dots_det = gt_dots_det.detach().cpu().numpy()

            # forward Propagation
            img = img.to(device_teacher)
            et_dmap_lst_teacher = model_teacher(img)
            et_dmap_det_teacher = et_dmap_lst_teacher[0]  # The cell detection prediction
            et_dmap_cls_teacher = et_dmap_lst_teacher[1]  # The cell classification prediction

            img = img.to(device_student)
            et_dmap_lst_student = model_student(img)
            et_dmap_det_student = et_dmap_lst_student[0]  # The cell detection prediction
            et_dmap_cls_student = et_dmap_lst_student[1]  # The cell classification prediction

            # Apply Sigmoid and Softmax activations to the detection and classification predictions, respectively.
            et_det_sig_teacher = criterion_sig(et_dmap_det_teacher).detach().cpu().numpy()
            et_cls_sig_teacher = criterion_softmax(et_dmap_cls_teacher).detach().cpu().numpy()

            et_det_sig_student = criterion_sig(et_dmap_det_student).detach().cpu().numpy()
            et_cls_sig_student = criterion_softmax(et_dmap_cls_student).detach().cpu().numpy()

            # Use average as final predictions
            et_det_sig = (et_det_sig_teacher + et_det_sig_student) / 2
            et_cls_sig = (et_cls_sig_teacher + et_cls_sig_student) / 2

            # Apply a threshold on detection output and convert to binary mask
            e_hard = filters.apply_hysteresis_threshold(et_det_sig.squeeze(), thresh_low, thresh_high)
            e_hard2 = (e_hard > 0).astype(np.uint8)

            # Get predicted cell centers by finding center of contours in binary mask
            e_dot = np.zeros((img.shape[-2], img.shape[-1]))
            contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for idx in range(len(contours)):
                contour_i = contours[idx]
                M = cv2.moments(contour_i)
                if (M['m00'] == 0):
                    continue;
                cx = round(M['m10'] / M['m00'])
                cy = round(M['m01'] / M['m00'])
                e_dot[cy, cx] = 1
            e_dot_all = e_dot.copy()

            g_dot_vis = gt_dots_det.copy().squeeze()
            tp_count, fp_count, fn_count = calc(g_dot_vis, e_dot)

            # Update TP, FP, FN counts for detection with counts from current image predictions
            tp_count_all[-1] = tp_count_all[-1] + tp_count
            fp_count_all[-1] = fp_count_all[-1] + fp_count
            fn_count_all[-1] = fn_count_all[-1] + fn_count

            # Get predicted cell classes
            et_class_argmax = et_cls_sig.squeeze().argmax(axis=0)
            e_hard2.copy()
            # For each class get the TP, FP, FN counts similar to previous detection code.
            for s in range(n_classes):
                gt_dots[0, s, :, :].sum()
                e_hard2 = (et_class_argmax == s)
                e_dot = e_hard2 * e_dot_all
                g_dot = gt_dots[0, s, :, :].squeeze()
                gt_centers = np.where(g_dot > 0)
                for idx in range(len(gt_centers[0])):
                    cx = gt_centers[1][idx]
                    cy = gt_centers[0][idx]
                    cx = int(cx + 0.5)
                    cy = int(cy + 0.5)
                    cv2.circle(img_gt, (cx, cy), 3, color_set[s], -1, cv2.LINE_AA)
                    cv2.circle(img_gt_et, (cx, cy), 8, color_set[s], 1, cv2.LINE_AA)
                et_centers = np.where(e_dot > 0)
                for idx in range(len(et_centers[0])):
                    cx = et_centers[1][idx]
                    cy = et_centers[0][idx]
                    cx = int(cx + 0.5)
                    cy = int(cy + 0.5)
                    cv2.circle(img_et, (cx, cy), 3, color_set[s], -1, cv2.LINE_AA)
                    cv2.circle(img_gt_et, (cx, cy), 3, color_set[s], -1, cv2.LINE_AA)

            cv2.imwrite(os.path.join(img_path, img_name.replace('.png', '_gt.png')), img_gt)
            cv2.imwrite(os.path.join(img_path, img_name.replace('.png', '_et.png')), img_et)
            cv2.imwrite(os.path.join(img_path, img_name.replace('.png', '_gt_et.png')), img_gt_et)


def test_solo(model, device, test_loader, n_classes):
    with torch.no_grad():
        n_classes_out = n_classes + 1

        model.eval()

        paired_all = []  # unique matched index pair
        unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
        unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
        true_inst_type_all = []  # each index is 1 independent data point
        pred_inst_type_all = []  # each index is 1 independent data point
        for file_idx, (img, gt_dmap, gt_dots, img_name) in enumerate(
                test_loader):
            ''' 
                img: input image
                gt_dmap: ground truth map for cell classes (lymphocytes, epithelial/tumor, stromal) with dilated dots. This can be a binary mask or a density map ( in which case it will be converted to a binary mask)
                gt_dots: ground truth binary dot map for cell classes (lymphocytes, epithelial/tumor, stromal). 
                img_name: img filename
            '''
            img = img.to(device)
            # Get the detection ground truth maps from the classes ground truth maps
            gt_dots_all = gt_dots.max(1)[0]

            # Convert ground truth maps to numpy arrays
            gt_dots = gt_dots.detach().cpu().numpy()
            gt_dots_all = gt_dots_all.detach().cpu().numpy()

            # forward Propagation
            et_dmap_lst = model(img)
            et_dmap_all = et_dmap_lst[0]  # The cell detection prediction
            et_dmap_class = et_dmap_lst[1]  # The cell classification prediction

            # Apply Sigmoid and Softmax activations to the detection and classification predictions, respectively.
            et_all_sig = criterion_sig(et_dmap_all).detach().cpu().numpy()
            et_class_sig = criterion_softmax(et_dmap_class).detach().cpu().numpy()

            # Apply a 0.5 threshold on detection output and convert to binary mask
            e_hard = filters.apply_hysteresis_threshold(et_all_sig.squeeze(), thresh_low, thresh_high)
            e_hard2 = (e_hard > 0).astype(np.uint8)

            # Get predicted cell centers by finding center of contours in binary mask
            e_dot = np.zeros((img.shape[-2], img.shape[-1]))
            contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for idx in range(len(contours)):
                contour_i = contours[idx]
                M = cv2.moments(contour_i)
                if (M['m00'] == 0):
                    continue;
                cx = round(M['m10'] / M['m00'])
                cy = round(M['m01'] / M['m00'])
                e_dot[cy, cx] = 1
            e_dot_all = e_dot.copy()

            g_dot_vis = gt_dots_all.copy().squeeze()

            pred_centroid = []
            pred_inst_type = []
            true_centroid = []
            true_inst_type = []

            et_class_argmax = et_class_sig.squeeze().argmax(axis=0)
            for s in range(n_classes):
                gt_dots[0, s, :, :].sum()
                e_hard2 = (et_class_argmax == s)
                e_dot = e_hard2 * e_dot_all
                g_dot = gt_dots[0, s, :, :].squeeze()

                centroids = np.argwhere(e_dot > 0)
                pred_centroid.extend(centroids)
                pred_inst_type.extend([s + 1] * len(centroids))

                centroids = np.argwhere(g_dot > 0)
                true_centroid.extend(centroids)
                true_inst_type.extend([s + 1] * len(centroids))

            pred_centroid = np.array(pred_centroid)
            pred_inst_type = np.array(pred_inst_type)
            true_centroid = np.array(true_centroid)
            true_inst_type = np.array(true_inst_type)

            true_centroid = np.asarray(true_centroid).astype("float32")
            true_inst_type = np.asarray(true_inst_type).astype("int32")

            if true_centroid.shape[0] != 0:
                true_inst_type = true_inst_type[:]
            else:  # no instance at all
                true_centroid = np.array([[0, 0]])
                true_inst_type = np.array([0])

            pred_centroid = np.asarray(pred_centroid).astype("float32")
            pred_inst_type = np.asarray(pred_inst_type).astype("int32")

            if pred_centroid.shape[0] != 0:
                pred_inst_type = pred_inst_type[:]
            else:  # no instance at all
                pred_centroid = np.array([[0, 0]])
                pred_inst_type = np.array([0])

            distance = 6
            # paired, unpaired_true, unpaired_pred = pair_coordinates(true_centroid, pred_centroid, distance)
            paired, unpaired_true, unpaired_pred = pair_coordinates(true_centroid, pred_centroid, distance)

            true_idx_offset = (true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0)
            pred_idx_offset = (pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0)
            true_inst_type_all.append(true_inst_type)
            pred_inst_type_all.append(pred_inst_type)

            if paired.shape[0] != 0:  # ! sanity
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                paired_all.append(paired)

            unpaired_true += true_idx_offset
            unpaired_pred += pred_idx_offset
            unpaired_true_all.append(unpaired_true)
            unpaired_pred_all.append(unpaired_pred)

        paired_all = np.concatenate(paired_all, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

        paired_true_type = true_inst_type_all[paired_all[:, 0]]
        paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
        unpaired_true_type = true_inst_type_all[unpaired_true_all]
        unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

        def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
            type_samples = (paired_true == type_id) | (paired_pred == type_id)

            paired_true = paired_true[type_samples]
            paired_pred = paired_pred[type_samples]

            tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
            tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
            fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
            fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

            fp_d = (unpaired_pred == type_id).sum()
            fn_d = (unpaired_true == type_id).sum()

            f1_type = (2 * (tp_dt + tn_dt)) / (
                    2 * (tp_dt + tn_dt)
                    + w[0] * fp_dt
                    + w[1] * fn_dt
                    + w[2] * fp_d
                    + w[3] * fn_d
            )
            return f1_type

        # overall
        # * quite meaningless for not exhaustive annotated dataset
        w = [1, 1]
        tp_d = paired_pred_type.shape[0]
        fp_d = unpaired_pred_type.shape[0]
        fn_d = unpaired_true_type.shape[0]

        tp_tn_dt = (paired_pred_type == paired_true_type).sum()
        fp_fn_dt = (paired_pred_type != paired_true_type).sum()

        acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
        f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

        w = [2, 2, 1, 1]

        type_uid_list = np.unique(true_inst_type_all).tolist()

        results_list = [f1_d, acc_type]
        for type_uid in type_uid_list:
            f1_type = _f1_type(
                paired_true_type,
                paired_pred_type,
                unpaired_true_type,
                unpaired_pred_type,
                type_uid,
                w,
            )
            results_list.append(f1_type)

        np.set_printoptions(formatter={"float": "{: 0.5f}".format})
        return results_list


def test_union(model_teacher, device_teacher, model_student, device_student, test_loader, n_classes):
    with torch.no_grad():
        n_classes_out = n_classes + 1

        model_teacher.eval()
        model_student.eval()

        paired_all = []  # unique matched index pair
        unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
        unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
        true_inst_type_all = []  # each index is 1 independent data point
        pred_inst_type_all = []  # each index is 1 independent data point
        for file_idx, (img, gt_dmap, gt_dots, img_name) in enumerate(test_loader):
            # Get the detection ground truth maps from the classes ground truth maps
            gt_dots_det = gt_dots.max(1)[0]

            # Convert ground truth maps to numpy arrays
            gt_dots = gt_dots.detach().cpu().numpy()
            gt_dots_det = gt_dots_det.detach().cpu().numpy()

            # forward Propagation
            img = img.to(device_teacher)
            et_dmap_lst_teacher = model_teacher(img)
            et_dmap_det_teacher = et_dmap_lst_teacher[0]  # The cell detection prediction
            et_dmap_cls_teacher = et_dmap_lst_teacher[1]  # The cell classification prediction

            img = img.to(device_student)
            et_dmap_lst_student = model_student(img)
            et_dmap_det_student = et_dmap_lst_student[0]  # The cell detection prediction
            et_dmap_cls_student = et_dmap_lst_student[1]  # The cell classification prediction

            # Apply Sigmoid and Softmax activations to the detection and classification predictions, respectively.
            et_det_sig_teacher = criterion_sig(et_dmap_det_teacher).detach().cpu().numpy()
            et_cls_sig_teacher = criterion_softmax(et_dmap_cls_teacher).detach().cpu().numpy()

            et_det_sig_student = criterion_sig(et_dmap_det_student).detach().cpu().numpy()
            et_cls_sig_student = criterion_softmax(et_dmap_cls_student).detach().cpu().numpy()

            et_det_sig = (et_det_sig_teacher + et_det_sig_student) / 2
            et_cls_sig = (et_cls_sig_teacher + et_cls_sig_student) / 2

            # Apply a 0.5 threshold on detection output and convert to binary mask
            e_hard = filters.apply_hysteresis_threshold(et_det_sig.squeeze(), thresh_low, thresh_high)
            e_hard2 = (e_hard > 0).astype(np.uint8)

            # Get predicted cell centers by finding center of contours in binary mask
            e_dot = np.zeros((img.shape[-2], img.shape[-1]))
            contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for idx in range(len(contours)):
                contour_i = contours[idx]
                M = cv2.moments(contour_i)
                if (M['m00'] == 0):
                    continue;
                cx = round(M['m10'] / M['m00'])
                cy = round(M['m01'] / M['m00'])
                e_dot[cy, cx] = 1
            e_dot_all = e_dot.copy()

            pred_centroid = []
            pred_inst_type = []
            true_centroid = []
            true_inst_type = []

            et_class_argmax = et_cls_sig.squeeze().argmax(axis=0)
            for s in range(n_classes):
                gt_dots[0, s, :, :].sum()
                e_hard2 = (et_class_argmax == s)
                e_dot = e_hard2 * e_dot_all
                g_dot = gt_dots[0, s, :, :].squeeze()

                centroids = np.argwhere(e_dot > 0)
                pred_centroid.extend(centroids)
                pred_inst_type.extend([s + 1] * len(centroids))

                centroids = np.argwhere(g_dot > 0)
                true_centroid.extend(centroids)
                true_inst_type.extend([s + 1] * len(centroids))

            pred_centroid = np.array(pred_centroid)
            pred_inst_type = np.array(pred_inst_type)
            true_centroid = np.array(true_centroid)
            true_inst_type = np.array(true_inst_type)

            true_centroid = np.asarray(true_centroid).astype("float32")
            true_inst_type = np.asarray(true_inst_type).astype("int32")

            if true_centroid.shape[0] != 0:
                true_inst_type = true_inst_type[:]
            else:  # no instance at all
                true_centroid = np.array([[0, 0]])
                true_inst_type = np.array([0])

            pred_centroid = np.asarray(pred_centroid).astype("float32")
            pred_inst_type = np.asarray(pred_inst_type).astype("int32")

            if pred_centroid.shape[0] != 0:
                pred_inst_type = pred_inst_type[:]
            else:  # no instance at all
                pred_centroid = np.array([[0, 0]])
                pred_inst_type = np.array([0])

            distance = 6
            # paired, unpaired_true, unpaired_pred = pair_coordinates(true_centroid, pred_centroid, distance)
            paired, unpaired_true, unpaired_pred = pair_coordinates(true_centroid, pred_centroid, distance)

            true_idx_offset = (true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0)
            pred_idx_offset = (pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0)
            true_inst_type_all.append(true_inst_type)
            pred_inst_type_all.append(pred_inst_type)

            if paired.shape[0] != 0:  # ! sanity
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                paired_all.append(paired)

            unpaired_true += true_idx_offset
            unpaired_pred += pred_idx_offset
            unpaired_true_all.append(unpaired_true)
            unpaired_pred_all.append(unpaired_pred)

        paired_all = np.concatenate(paired_all, axis=0)
        unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
        unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
        true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
        pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

        paired_true_type = true_inst_type_all[paired_all[:, 0]]
        paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
        unpaired_true_type = true_inst_type_all[unpaired_true_all]
        unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

        def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
            type_samples = (paired_true == type_id) | (paired_pred == type_id)

            paired_true = paired_true[type_samples]
            paired_pred = paired_pred[type_samples]

            tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
            tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
            fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
            fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

            fp_d = (unpaired_pred == type_id).sum()
            fn_d = (unpaired_true == type_id).sum()

            f1_type = (2 * (tp_dt + tn_dt)) / (
                    2 * (tp_dt + tn_dt)
                    + w[0] * fp_dt
                    + w[1] * fn_dt
                    + w[2] * fp_d
                    + w[3] * fn_d
            )
            return f1_type

        # overall
        # * quite meaningless for not exhaustive annotated dataset
        w = [1, 1]
        tp_d = paired_pred_type.shape[0]
        fp_d = unpaired_pred_type.shape[0]
        fn_d = unpaired_true_type.shape[0]

        tp_tn_dt = (paired_pred_type == paired_true_type).sum()
        fp_fn_dt = (paired_pred_type != paired_true_type).sum()

        acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
        f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

        w = [2, 2, 1, 1]

        type_uid_list = np.unique(true_inst_type_all).tolist()

        results_list = [f1_d, acc_type]
        for type_uid in type_uid_list:
            f1_type = _f1_type(
                paired_true_type,
                paired_pred_type,
                unpaired_true_type,
                unpaired_pred_type,
                type_uid,
                w,
            )
            results_list.append(f1_type)

        np.set_printoptions(formatter={"float": "{: 0.5f}".format})
        return results_list


def pair_coordinates(setA, setB, radius):
    leafsize = 2048
    k = 50
    if len(setB) > 0:
        tree = scipy.spatial.KDTree(setB, leafsize=leafsize)

    if len(setB) == 0:
        pairing = np.zeros((0, 2))
        unpairedA = range(0, len(setA))
        unpairedA = np.array(unpairedA)
        unpairedB = np.zeros((0,))

    else:
        pairing = []
        unpairedA = []
        unpairedB = []
        pairedA = []
        pairedB = []
        for i in range(0, len(setA)):
            p = setA[i]
            distances, locations = tree.query(p, k=k, distance_upper_bound=radius)
            match = False
            for nn in range(min(k, len(locations))):
                if ((len(locations) > 0) and (locations[nn] < tree.data.shape[0]) and (locations[nn] not in pairedB)):
                    pairing.append([i, locations[nn]])
                    pairedA.append(i)
                    pairedB.append(locations[nn])
                    match = True
                    break
            if (not match):
                unpairedA.append(i)

        pairing = np.array(pairing)
        pairedA = np.array(pairedA)
        pairedB = np.array(pairedB)
        if (len(pairing) == 0):
            pairing = np.zeros((0, 2))
        if (len(pairedA) == 0):
            unpairedA = np.arange(setA.shape[0])
        else:
            unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
        if (len(pairedB) == 0):
            unpairedB = np.arange(setB.shape[0])
        else:
            unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)

    return pairing, unpairedA, unpairedB
