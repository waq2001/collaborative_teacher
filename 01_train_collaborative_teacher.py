import os
import warnings

import torch
from numpy import mean
from tqdm import tqdm as tqdm

import options
import test
import train
from dataloader import CellsDataset
from discriminator import DomainDiscriminator
from model_arch import UnetVggMultihead
from util import seed_torch, loop_iterable, save_model_union, copy_dir, print_result

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # load options
    parser = options.get_parser()
    args = parser.parse_args()

    checkpoints_folder_name = args.checkpoints_folder_name
    checkpoints_root_dir = args.checkpoints_root_dir
    model_param_path = args.model_param_path
    n_classes = args.n_classes
    epochs = args.epochs
    epochs_init = args.epochs_init
    lr = args.lr
    gama = args.gama
    seed = args.seed
    batch_size = args.batch_size

    train_source_data_root = args.train_source_data_root
    train_source_split_filepath = args.train_source_split_filepath
    train_target_data_root = args.train_target_data_root
    train_target_split_filepath = args.train_target_split_filepath
    val_data_root = args.val_data_root
    val_split_filepath = args.val_split_filepath
    test_data_root = args.test_data_root
    test_split_filepath = args.test_split_filepath

    device_choose_0 = args.device_choose_0
    device_teacher = torch.device(device_choose_0)
    device_choose_1 = args.device_choose_1
    device_student = torch.device(device_choose_1)

    # set seed
    if seed is not None:
        seed_torch(seed)

    # checkpoints_save_path: path to save checkpoints
    checkpoints_save_path = os.path.join(checkpoints_root_dir, checkpoints_folder_name)

    if not os.path.exists(checkpoints_root_dir):
        os.mkdir(checkpoints_root_dir)

    if not os.path.exists(checkpoints_save_path):
        os.mkdir(checkpoints_save_path)
    else:
        print('checkpoints folder already exists!')
        exit(-1)
    copy_dir('./', checkpoints_save_path)  # save current code

    # create log file
    log_file_path = os.path.join(checkpoints_root_dir, checkpoints_folder_name, f'train_log.txt')
    log_file = open(log_file_path, 'a+')  # Initialize log file

    # print arguments
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
        log_file.write(k + '=' + str(v) + '\n')
        log_file.flush()

    # configure source domain training dataset
    train_source_image_root = os.path.join(train_source_data_root, 'images')
    train_source_dmap_root = os.path.join(train_source_data_root, 'gt_custom')
    train_source_dots_root = os.path.join(train_source_data_root, 'gt_custom')
    # configure target domain training dataset
    train_target_image_root = os.path.join(train_target_data_root, 'images')
    train_target_dmap_root = os.path.join(train_target_data_root, 'gt_custom')
    train_target_dots_root = os.path.join(train_target_data_root, 'gt_custom')
    # configure target domain validation dataset
    val_image_root = os.path.join(val_data_root, 'images')
    val_dmap_root = os.path.join(val_data_root, 'gt_custom')
    val_dots_root = os.path.join(val_data_root, 'gt_custom')
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
    discriminator_teacher = DomainDiscriminator()
    discriminator_student = DomainDiscriminator()

    # put models into GPU
    model_teacher.to(device_teacher)
    model_student.to(device_student)
    discriminator_teacher.to(device_teacher)
    discriminator_student.to(device_student)

    # load model_param if it exists
    if (not (model_param_path is None)):
        model_teacher.load_state_dict(torch.load(model_param_path), strict=False)
        model_student.load_state_dict(torch.load(model_param_path), strict=False)
        tqdm.write('model loaded')
        log_file.write('model loaded \n')
        log_file.flush()

    # initialize optimizer
    optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), lr)
    optimizer_student = torch.optim.Adam(model_student.parameters(), lr)
    optimizer_discriminator_teacher = torch.optim.Adam(discriminator_teacher.parameters(), 0.0001)
    optimizer_discriminator_student = torch.optim.Adam(discriminator_student.parameters(), 0.0001)

    # initialize source domain training dataset loader
    train_source_dataset = CellsDataset(train_source_image_root, train_source_dmap_root, train_source_dots_root,
                                        class_indx, split_filepath=train_source_split_filepath,
                                        train_phase='train',
                                        fixed_size=448, max_scale=16)
    train_source_loader = torch.utils.data.DataLoader(train_source_dataset, batch_size=batch_size, shuffle=True)
    # initialize target domain training dataset loader
    train_target_dataset = CellsDataset(train_target_image_root, train_target_dmap_root, train_target_dots_root,
                                        class_indx, split_filepath=train_target_split_filepath,
                                        train_phase='train',
                                        fixed_size=448, max_scale=16)
    train_target_loader = torch.utils.data.DataLoader(train_target_dataset, batch_size=batch_size, shuffle=True)
    # initialize target domain validation dataset loader
    val_dataset = CellsDataset(val_image_root, val_dmap_root, val_dots_root,
                               class_indx, split_filepath=val_split_filepath,
                               train_phase='test',
                               fixed_size=-1, max_scale=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    # initialize target domain testing dataset loader
    test_dataset = CellsDataset(test_image_root, test_dmap_root, test_dots_root,
                                class_indx, split_filepath=test_split_filepath,
                                train_phase='test',
                                fixed_size=-1, max_scale=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # zip source and target dataset loader
    train_iterator = zip(loop_iterable(train_source_loader), loop_iterable(train_target_loader))
    train_source_num = len(train_source_dataset)
    train_target_num = len(train_target_dataset)

    best_epoch_filepath = None
    best_epoch = None
    best_union_f1_mean = 0
    PE_teacher = 0

    # initialize teacher model
    tqdm.write('---initializing---')
    for epoch in tqdm(range(epochs_init)):
        epoch_loss = train.train_teacher_per_epoch(model_teacher=model_teacher, device_teacher=device_teacher,
                                                   model_discriminator=discriminator_teacher,
                                                   train_iterator=train_iterator,
                                                   train_sample_num=train_source_num,
                                                   optimizer_teacher=optimizer_teacher,
                                                   optimizer_discriminator=optimizer_discriminator_teacher, epoch=epoch)

    # initialize student model
    previous_teacher_dict = model_teacher.state_dict()
    model_student.load_state_dict(previous_teacher_dict)
    previous_discriminator_teacher = discriminator_teacher.state_dict()
    discriminator_student.load_state_dict(previous_discriminator_teacher)

    # training phase
    tqdm.write('---training phase---')
    for epoch in tqdm(range(epochs)):
        tqdm.write('epoch=' + str(epoch))
        log_file.write('epoch=' + str(epoch) + '\n')
        log_file.flush()

        # === teacher training phase ===
        tqdm.write('teacher training')
        epoch_loss = train.train_teacher_per_epoch(model_teacher=model_teacher, device_teacher=device_teacher,
                                                   model_discriminator=discriminator_teacher,
                                                   train_iterator=train_iterator,
                                                   train_sample_num=train_source_num,
                                                   optimizer_teacher=optimizer_teacher,
                                                   optimizer_discriminator=optimizer_discriminator_teacher,
                                                   epoch=epoch + epochs_init)
        tqdm.write("epoch: " + str(epoch) + "  teacher_loss: " + str(epoch_loss))
        log_file.write("epoch: " + str(epoch) + "  teacher_loss: " + str(epoch_loss) + '\n')
        log_file.flush()

        # === student training phase ===
        epoch_loss,PE_student = train.train_student_per_epoch(model_teacher=model_teacher, device_teacher=device_teacher,
                                                   model_student=model_student, device_student=device_student,
                                                   train_iterator=train_iterator,
                                                   train_sample_num=train_source_num,
                                                   optimizer_student=optimizer_student,
                                                   model_discriminator=discriminator_student,
                                                   optimizer_discriminator=optimizer_discriminator_student)
        tqdm.write("epoch: " + str(epoch) + "  student_loss: " + str(epoch_loss))
        log_file.write("epoch: " + str(epoch) + "  student_loss: " + str(epoch_loss) + '\n')
        log_file.flush()

        # === validating phase ===
        result_list = test.test_union(model_teacher=model_teacher,
                                               device_teacher=device_teacher,
                                               model_student=model_student,
                                               device_student=device_student,
                                               test_loader=val_loader, n_classes=n_classes)
        print_result(result_list, epoch, log_file)
        if (mean(result_list[2:]) + result_list[0] > best_union_f1_mean):
            best_union_f1_mean = mean(result_list[2:]) + result_list[0]
            tqdm.write('epoch ' + str(epoch) + ' saving')
            save_model_union(model_teacher=model_teacher, model_student=model_student,
                             checkpoints_save_path=checkpoints_save_path, epoch=epoch)
            tqdm.write_msg = f'epoch {epoch} saved.'
            tqdm.write(tqdm.write_msg)
            log_file.write(tqdm.write_msg + '\n')
            log_file.flush()

        # === identity swap mechanism ===
        # compute teacher and student model entropy
        tqdm.write('PE_student {}'.format(PE_student))
        # compare model entropy and swap identity
        if PE_teacher < PE_student:
            PE_teacher = PE_student
            tqdm.write('identity swap at epoch {}'.format(epoch))
            log_file.write('identity swap at epoch {}'.format(epoch) + '\n')
            log_file.flush()

            previous_teacher_dict = model_teacher.state_dict()
            previous_student_dict = model_student.state_dict()
            previous_discriminator_teacher = discriminator_teacher.state_dict()
            previous_discriminator_student = discriminator_student.state_dict()

            model_teacher.load_state_dict(previous_student_dict)
            model_student.load_state_dict(previous_teacher_dict)
            discriminator_teacher.load_state_dict(previous_discriminator_student)
            discriminator_student.load_state_dict(previous_discriminator_teacher)

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

    log_file.close()