import os
import warnings

import torch
from tqdm import tqdm as tqdm

import options
import test
from dataloader import CellsDataset
from model_arch import UnetVggMultihead
from util import print_result

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # load options
    parser = options.get_parser()
    args = parser.parse_args()

    checkpoints_folder_name = args.checkpoints_folder_name
    checkpoints_root_dir = args.checkpoints_root_dir
    model_param_path = args.model_param_path
    n_classes = args.n_classes

    test_data_root = args.test_data_root
    test_split_filepath = args.test_split_filepath

    device_choose_0 = args.device_choose_0
    device_teacher = torch.device(device_choose_0)
    device_choose_1 = args.device_choose_1
    device_student = torch.device(device_choose_1)

    # checkpoints_save_path: path to save checkpoints
    checkpoints_save_path = os.path.join(checkpoints_root_dir, checkpoints_folder_name)

    if not os.path.exists(checkpoints_save_path):
        print('no checkpoints exists!')
        exit(-1)

    # create log file
    log_file_path = os.path.join(checkpoints_root_dir, checkpoints_folder_name, f'test_log.txt')
    log_file = open(log_file_path, 'a+')  # Initialize log file

    # print arguments
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
        log_file.write(k + '=' + str(v) + '\n')
        log_file.flush()

    # configure target domain testing dataset
    test_image_root = os.path.join(test_data_root, 'images')
    test_dmap_root = os.path.join(test_data_root, 'gt_custom')
    test_dots_root = os.path.join(test_data_root, 'gt_custom')

    # set network options
    dropout_prob = 0.2
    initial_pad = 126  # add padding so that final output has same size as input since we do not use same padding conv.
    interpolate = 'False'
    conv_init = 'he'
    n_channels = 3  # input channels
    n_classes_out = n_classes + 1  # number of output classes = number of cell classes (epithelial, inflammatory, stromal) + 1 (for cell detection channel)
    class_indx = list(range(1, n_classes_out))
    class_indx = [str(x) for x in class_indx]
    class_indx = ','.join(class_indx)  # the index of the classes channels in the ground truth

    tqdm.writes_per_epoch = 1  # tqdm.write frequency per epoch

    # create teacher model, student model and discriminator
    model_teacher = UnetVggMultihead(
        kwargs={'dropout_prob': dropout_prob, 'initial_pad': initial_pad, 'interpolate': interpolate,
                'conv_init': conv_init, 'n_classes': n_classes, 'n_channels': n_channels, 'n_heads': 2,
                'head_classes': [1, n_classes]})
    model_student = UnetVggMultihead(
        kwargs={'dropout_prob': dropout_prob, 'initial_pad': initial_pad, 'interpolate': interpolate,
                'conv_init': conv_init, 'n_classes': n_classes, 'n_channels': n_channels, 'n_heads': 2,
                'head_classes': [1, n_classes]})

    # put models into GPU
    model_teacher.to(device_teacher)
    model_student.to(device_student)

    # initialize target domain testing dataset loader
    test_dataset = CellsDataset(test_image_root, test_dmap_root, test_dots_root,
                                class_indx, split_filepath=test_split_filepath,
                                train_phase='test',
                                fixed_size=-1, max_scale=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # testing phase
    tqdm.write('---testing phase---')
    # load best models
    teacher_filepath = os.path.join(checkpoints_save_path, 'best_teacher.pth')
    student_filepath = os.path.join(checkpoints_save_path, 'best_student.pth')
    model_teacher.load_state_dict(torch.load(teacher_filepath), strict=False)
    model_student.load_state_dict(torch.load(student_filepath), strict=False)

    # teacher model testing
    tqdm.write_msg = f'teacher testing'
    tqdm.write(tqdm.write_msg)
    log_file.write(tqdm.write_msg + '\n')
    log_file.flush()
    result_list = test.test_solo(model=model_teacher, device=device_teacher,
                                 test_loader=test_loader, n_classes=n_classes)
    print_result(result_list, -1, log_file)

    # student model testing
    tqdm.write_msg = f'student testing'
    tqdm.write(tqdm.write_msg)
    log_file.write(tqdm.write_msg + '\n')
    log_file.flush()
    result_list = test.test_solo(model=model_student, device=device_student,
                                 test_loader=test_loader, n_classes=n_classes)
    print_result(result_list, -1, log_file)

    # teacher and student model testing
    tqdm.write_msg = f'teacher and student testing'
    tqdm.write(tqdm.write_msg)
    log_file.write(tqdm.write_msg + '\n')
    log_file.flush()
    result_list = test.test_union(model_teacher=model_teacher,
                                  device_teacher=device_teacher,
                                  model_student=model_student,
                                  device_student=device_student,
                                  test_loader=test_loader, n_classes=n_classes)
    print_result(result_list, -1, log_file)

    # visualize
    tqdm.write("visualizing")
    test.visualize_union(model_teacher=model_teacher, device_teacher=device_teacher, model_student=model_student,
                         device_student=device_student, test_loader=test_loader,
                         n_classes=n_classes, checkpoints_save_path=checkpoints_save_path)
    log_file.close()
