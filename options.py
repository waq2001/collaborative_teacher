import argparse


def parse_common_args(parser):
    parser.add_argument('--checkpoints_root_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoints_folder_name', type=str, default='consep2glas')
    parser.add_argument('--device_choose_0', type=str, default='cuda:0')
    parser.add_argument('--device_choose_1', type=str, default='cuda:0')
    return parser


def parse_train_args(parser):
    parser.add_argument('--model_param_path', type=str, default=None)
    parser.add_argument('--n_classes', type=int, default=3,
                        help='number of nuclei classes')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--epochs_init', type=int, default=20, help='epochs before identity swap')
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=1, help='det_loss/class_loss')
    parser.add_argument('--lam', type=float, default=1, help='teacher_loss/student_loss')
    parser.add_argument('--gama', type=float, default=1, help='teacher_entropy/student_entropy')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pseudo_thresh_uncertainty', type=float, default=0.8)
    parser.add_argument('--T', type=int, default=8, help='uncertainty estimation forward pass times')
    return parser


def parse_test_args(parser):
    parser.add_argument('--dist_thresh', type=int, default=6,
                        help='will compute fscore at distance thresholds in range')
    parser.add_argument('--test_thresh_low', type=float, default=0.5)
    parser.add_argument('--test_thresh_high', type=float, default=0.5)
    return parser


def parse_dataset_args(parser):
    # -----source-----
    # consep
    parser.add_argument('--train_source_data_root', type=str,
                        default='../MCSpatNet-data-prepare/datasets_prepared/CoNSeP')
    parser.add_argument('--train_source_split_filepath', type=str,
                        default=None)

    # -----target-----
    # glas
    parser.add_argument('--train_target_data_root', type=str,
                        default='../MCSpatNet-data-prepare/datasets_prepared/Lizard')
    parser.add_argument('--train_target_split_filepath', type=str,
                        default='../MCSpatNet-data-prepare/data_splits/UDA/lizard-glas/train_split.txt')
    parser.add_argument('--val_data_root', type=str,
                        default='../MCSpatNet-data-prepare/datasets_prepared/Lizard')
    parser.add_argument('--val_split_filepath', type=str,
                        default='../MCSpatNet-data-prepare/data_splits/UDA/lizard-glas/val_split.txt')
    parser.add_argument('--test_data_root', type=str,
                        default='../MCSpatNet-data-prepare/datasets_prepared/Lizard')
    parser.add_argument('--test_split_filepath', type=str,
                        default='../MCSpatNet-data-prepare/data_splits/UDA/lizard-glas/test_split.txt')

    return parser


def get_parser():
    parser = argparse.ArgumentParser()
    parser = parse_common_args(parser)
    parser = parse_train_args(parser)
    parser = parse_test_args(parser)
    parser = parse_dataset_args(parser)
    return parser
