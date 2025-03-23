import os

import numpy as np
import scipy.spatial
import torch
import random

from numpy import mean
from tqdm import tqdm

dist_thresh = 6  # will compute fscore at distance thresholds in range (1,max_dist_thresh) # mpp = 0.254 at 40x,  ppm at 20x = 1/(0.254*2),  mpp at 20x = 0.254*2 = 0.508, 6 px = 0.508*6 = 3.048 microns, , 30 px = 0.508*30=15.24 microns
color_set = {'tp': (0, 162, 232), 'fp': (0, 255, 0), 'fn': (255, 255, 0)}


class CNNArchUtilsPyTorch:
    @staticmethod
    def crop_a_to_b(input_a, input_b):
        shape_a = input_a.size()
        shape_b = input_b.size()
        cropped = input_a[:, :, (shape_a[2] - shape_b[2]) // 2: (shape_a[2] - shape_b[2]) // 2 + shape_b[2],
                  (shape_a[3] - shape_b[3]) // 2: (shape_a[3] - shape_b[3]) // 2 + shape_b[3]]

        return cropped


def adjust_learning_rate(optimizer, epoch, g_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # if epoch < 40:
    #     lr = g_lr
    # elif epoch < 70:
    #     lr = g_lr * 0.1
    # else:
    #     lr = g_lr * 0.01

    lr = g_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, epoch, checkpoints_save_path):
    filepath = os.path.join(checkpoints_save_path,
                            'best_model.pth')
    torch.save(model.state_dict(), filepath)


def save_model_union(model_teacher, model_student, epoch, checkpoints_save_path):
    teacher_filepath = os.path.join(checkpoints_save_path,
                                    'best_teacher.pth')
    torch.save(model_teacher.state_dict(), teacher_filepath)
    student_filepath = os.path.join(checkpoints_save_path,
                                    'best_student.pth')
    torch.save(model_student.state_dict(), student_filepath)


def print_result(result_list, epoch, log_file):
    tqdm.write_msg = f'epoch {epoch} detection F1:{result_list[0]}'
    tqdm.write(tqdm.write_msg)
    log_file.write(tqdm.write_msg + '\n')
    log_file.flush()
    for i in range(2, len(result_list)):
        tqdm.write_msg = f'epoch {epoch} class {i - 1} F1:{result_list[i]}'
        tqdm.write(tqdm.write_msg)
        log_file.write(tqdm.write_msg + '\n')
        log_file.flush()
    tqdm.write_msg = f'epoch {epoch} classification F1:{mean(result_list[2:])}'
    tqdm.write(tqdm.write_msg)
    log_file.write(tqdm.write_msg + '\n')
    log_file.flush()


def seed_torch(seed=403):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calc(g_dot, e_dot):
    '''
        Calculates number of TP, FP, FN for class_indx at different distance thresholds.
        For a threshold t, a TP prediction is within t pixels from a ground truth prediction that was not previously processed.
    '''
    leafsize = 2048
    k = 50
    e_coords = np.where(e_dot > 0)
    # Build kdtree from prediction cell centers
    z = np.zeros((len(e_coords[0]), 2))
    z[:, 0] = e_coords[0]
    z[:, 1] = e_coords[1]
    if (len(e_coords[0]) > 0):
        tree = scipy.spatial.KDTree(z, leafsize=leafsize)

    img_f = np.zeros((e_dot.shape[0], e_dot.shape[1], 3))
    if (len(e_coords[0]) == 0):  # case: no predictions
        tp_img = 0
        fn_img = (g_dot > 0).sum()
        fp_img = 0

    else:
        tp_img = 0
        fn_img = 0
        fp_img = 0

        e_dot_processing = np.copy(e_dot)

        gt_points = np.where(g_dot > 0)
        ''' 
            Loop over ground truth points and find nearest prediction within threshold distance
                If there is a match and the matching point exists in e_dot_processing, 
                    then this is a TP, remove from the matching point from e_dot_processing so that each prediction is matched only once.
                Otherwise
                    This is a FN
            Remaining predictions in e_dot_processing are counted as FPs                
        '''
        for pi in range(len(gt_points[0])):
            p = [[gt_points[0][pi], gt_points[1][pi]]]
            distances, locations = tree.query(p, k=k, distance_upper_bound=dist_thresh)
            match = False
            for nn in range(min(k, len(locations[0]))):
                if ((len(locations[0]) > 0) and (locations[0][nn] < tree.data.shape[0]) and (e_dot_processing[int(
                        tree.data[locations[0][nn]][0]), int(tree.data[locations[0][nn]][1])] > 0)):
                    # if((len(locations[0]) > 0) and (locations[0][nn] < tree.data.shape[0]) ):
                    tp_img += 1
                    e_dot_processing[int(tree.data[locations[0][nn]][0]), int(tree.data[locations[0][nn]][1])] = 0
                    match = True
                    py = int(tree.data[locations[0][nn]][0])
                    px = int(tree.data[locations[0][nn]][1])
                    img_f[max(0, py - 2):min(img_f.shape[0], py + 3), max(0, px - 2):min(img_f.shape[1], px + 3)] = \
                        color_set['tp']
                    break
            if (not match):
                fn_img += 1
                py = gt_points[0][pi]
                px = gt_points[1][pi]
                img_f[max(0, py - 2):min(img_f.shape[0], py + 3), max(0, px - 2):min(img_f.shape[1], px + 3)] = \
                    color_set['fn']

        fp_img += e_dot_processing.sum()
        fp_points = np.where(e_dot_processing > 0)
        for pi in range(len(fp_points[0])):
            py = fp_points[0][pi]
            px = fp_points[1][pi]
            img_f[max(0, py - 2):min(img_f.shape[0], py + 3), max(0, px - 2):min(img_f.shape[1], px + 3)] = color_set[
                'fp']

    return tp_img, fp_img, fn_img


def loop_iterable(iterable):
    while True:
        yield from iterable


def copy_dir(src_path, target_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    if os.path.isdir(src_path) and os.path.isdir(target_path):
        filelist_src = os.listdir(src_path)
        for file in filelist_src:
            path = os.path.join(os.path.abspath(src_path), file)
            if os.path.isdir(path):
                continue
            else:
                with open(path, 'rb') as read_stream:
                    contents = read_stream.read()
                    path1 = os.path.join(target_path, file)
                    with open(path1, 'wb') as write_stream:
                        write_stream.write(contents)
        return True

    else:
        return False


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
