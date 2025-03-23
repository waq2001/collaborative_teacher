import math

import torch
import torch.nn as nn
from torchvision import transforms

import options
from util import enable_dropout

parser = options.get_parser()
args = parser.parse_args()

alpha = args.alpha
n_classes = args.n_classes
thresh_uncertainty = args.pseudo_thresh_uncertainty
T = args.T

source_domain_label = 0
target_domain_label = 1

criterion_sig = nn.Sigmoid()
criterion_softmax = nn.Softmax(dim=1)
bce_loss = nn.BCELoss()

perturb = transforms.Compose([
    transforms.ColorJitter(contrast=0.2, saturation=0.2, hue=0.2),
    transforms.GaussianBlur(3, (0.1, 1))
])


def train_teacher_per_epoch(model_teacher, device_teacher, model_discriminator,
                            train_iterator, train_sample_num, optimizer_teacher, optimizer_discriminator, epoch):
    """
    Teacher training phase.

    Parameters:
    - model_teacher: The teacher model, used for cell detection and classification.
    - device_teacher: The device on which the teacher model runs.
    - model_discriminator: The discriminator model, used to distinguish between source and target domain features.
    - train_iterator: The data iterator for training data.
    - train_sample_num: The number of training samples.
    - optimizer_teacher: The optimizer for the teacher model.
    - optimizer_discriminator: The optimizer for the discriminator model.
    - epoch: The current epoch number.

    Returns:
    - The average loss of the teacher model for this epoch.
    """
    # Set the teacher model and discriminator model to training mode
    model_teacher.train()
    model_discriminator.train()

    # Initialize the sum of epoch losses and the number of trained samples
    epoch_loss_sum = 0
    train_count = 0

    # Iterate through the training samples
    for i in range(train_sample_num):

        # Increment the number of trained samples
        train_count += 1

        # train teacher model
        # Enable gradient calculation for the teacher model and disable it for the discriminator model
        for param in model_teacher.parameters():
            param.requires_grad = True
        for param in model_discriminator.parameters():
            param.requires_grad = False

        # Get the next batch of training data
        (img_s, gt_dmap_s, gt_dots_s, img_name_s), (img_t, gt_dmap_t, gt_dots_t, img_name_t) = next(train_iterator)
        img_s = img_s.to(device_teacher)
        # Convert ground truth maps to binary mask (in case they were density maps)
        gt_dmap_s = gt_dmap_s > 0
        # Get the detection ground truth maps from the classes ground truth maps
        gt_dmap_det_s = gt_dmap_s.max(1)[0]
        # Set datatype and move to GPU
        gt_dmap_s = gt_dmap_s.type(torch.FloatTensor)
        gt_dmap_det_s = gt_dmap_det_s.type(torch.FloatTensor)
        gt_dmap_s = gt_dmap_s.to(device_teacher)
        gt_dmap_det_s = gt_dmap_det_s.to(device_teacher)

        # Forward propagation
        et_dmap_lst_s = model_teacher(img_s)
        et_dmap_det_s = et_dmap_lst_s[0]  # The cell detection prediction
        et_dmap_cls_s = et_dmap_lst_s[1]  # The cell classification prediction
        feature_decoded_s = et_dmap_lst_s[2]

        # Apply Sigmoid and Softmax activations to the detection and classification predictions, respectively.
        et_det_sig_s = criterion_sig(et_dmap_det_s)
        et_cls_sig_s = criterion_softmax(et_dmap_cls_s)

        # Compute Dice loss on the detection and classification predictions
        intersection_s = (et_cls_sig_s * gt_dmap_s).sum()
        union_s = (et_cls_sig_s ** 2).sum() + (gt_dmap_s ** 2).sum()
        loss_cls_s = 1 - ((2 * intersection_s + 1) / (union_s + 1))

        intersection_s = (et_det_sig_s * gt_dmap_det_s.unsqueeze(0)).sum()
        union_s = (et_det_sig_s ** 2).sum() + (gt_dmap_det_s.unsqueeze(0) ** 2).sum()
        loss_det_s = 1 - ((2 * intersection_s + 1) / (union_s + 1))

        loss_teacher = alpha * loss_det_s + loss_cls_s

        # Adversarial training ot fool the discriminator
        img_t = img_t.to(device_teacher)
        # Forward propagation
        et_dmap_lst_t = model_teacher(img_t)
        feature_decoded_t = et_dmap_lst_t[2]

        out_s = model_discriminator(feature_decoded_s)
        loss_adv = 0.01 * bce_loss(out_s, torch.FloatTensor(out_s.data.size()).fill_(target_domain_label).to(
            device_teacher))
        out_t = model_discriminator(feature_decoded_t)
        loss_adv += 0.01 * bce_loss(out_t, torch.FloatTensor(out_t.data.size()).fill_(source_domain_label).to(
            device_teacher))
        if epoch > 10 and epoch % 2 == 0:
            loss_teacher += loss_adv

        # Backpropagation
        epoch_loss_sum += loss_teacher.item()

        optimizer_teacher.zero_grad()
        loss_teacher.backward()
        optimizer_teacher.step()

        # train domain discriminator
        # enable training mode on discriminator networks
        for param in model_discriminator.parameters():
            param.requires_grad = True
        for param in model_teacher.parameters():
            param.requires_grad = False

        feature_decoded_s = feature_decoded_s.detach()
        feature_decoded_t = feature_decoded_t.detach()

        out_s = model_discriminator(feature_decoded_s)
        loss_discriminator = bce_loss(out_s, torch.FloatTensor(out_s.data.size()).fill_(source_domain_label).to(
            device_teacher))
        optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        optimizer_discriminator.step()

        out_t = model_discriminator(feature_decoded_t)
        loss_discriminator = bce_loss(out_t, torch.FloatTensor(out_t.data.size()).fill_(target_domain_label).to(
            device_teacher))
        optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        optimizer_discriminator.step()

    # Ensure the teacher model is in training mode and the discriminator model is in evaluation mode
    for param in model_teacher.parameters():
        param.requires_grad = True
    for param in model_discriminator.parameters():
        param.requires_grad = False

    # Return the average loss of the teacher model for this epoch
    return epoch_loss_sum / train_count



def train_student_per_epoch(model_teacher, device_teacher, model_student, device_student, train_iterator,
                            optimizer_student, train_sample_num, model_discriminator, optimizer_discriminator):
    """
    Student training phase.

    Parameters:
    - model_teacher: The teacher model, used for generating pseudo labels.
    - device_teacher: The device on which the teacher model runs.
    - model_student: The student model, being trained.
    - device_student: The device on which the student model runs.
    - train_iterator: The data iterator for training data.
    - optimizer_student: The optimizer for the student model.
    - train_sample_num: The number of training samples.
    - model_discriminator: The discriminator model, used for domain adaptation.
    - optimizer_discriminator: The optimizer for the discriminator model.

    Returns:
    - The average loss of the student model for this epoch.
    - The performance estimation of the student model.
    """

    model_teacher.eval()
    model_student.train()

    epoch_loss_sum = 0
    train_count = 0
    PE_student = 0
    for i in range(train_sample_num):
        train_count += 1
        (img_s, gt_dmap_s, gt_dots_s, img_name_s), (img_t, gt_dmap_t, gt_dots_t, img_name_t) = next(train_iterator)
        # training on target domain
        with (torch.no_grad()):
            # 1.teacher predict target sample
            model_teacher.eval()
            enable_dropout(model_teacher)
            # MC dropout
            outputs_det_teacher = []
            outputs_cls_teacher = []
            for t in range(T):
                img_t = img_t.to(device_teacher)
                et_dmap_lst_t_teacher = model_teacher(img_t)
                et_dmap_det_t_teacher = et_dmap_lst_t_teacher[0]  # The cell detection prediction
                et_dmap_cls_t_teacher = et_dmap_lst_t_teacher[1]  # The cell classification prediction

                et_det_sig_t_teacher = criterion_sig(et_dmap_det_t_teacher)
                et_cls_sig_t_teacher = criterion_softmax(et_dmap_cls_t_teacher)

                outputs_det_teacher.append(et_det_sig_t_teacher)
                outputs_cls_teacher.append(et_cls_sig_t_teacher)

            outputs_det_teacher = torch.stack(outputs_det_teacher, dim=0)
            outputs_cls_teacher = torch.stack(outputs_cls_teacher, dim=0)

            et_det_sig_t_teacher = torch.mean(outputs_det_teacher, dim=0)
            et_cls_sig_t_teacher = torch.mean(outputs_cls_teacher, dim=0)

            en_det_t_teacher = -1.0 * torch.sum(
                et_det_sig_t_teacher * torch.log(et_det_sig_t_teacher + 1e-6)
                + (1 - et_det_sig_t_teacher) * torch.log((1 - et_det_sig_t_teacher) + 1e-6),
                dim=1, keepdim=True)
            en_cls_t_teacher = -1.0 * torch.sum(
                et_cls_sig_t_teacher * torch.log(et_cls_sig_t_teacher + 1e-6),
                dim=1, keepdim=True)

            et_det_sig_t_teacher = et_det_sig_t_teacher.to(device_student).detach()
            et_cls_sig_t_teacher = et_cls_sig_t_teacher.to(device_student).detach()

            # 2.student predict target sample
            model_student.eval()
            enable_dropout(model_student)
            # MC dropout
            outputs_det_student = []
            outputs_cls_student = []
            for t in range(T):
                img_t = img_t.to(device_student)
                et_dmap_lst_t_student = model_student(img_t)
                et_dmap_det_t_student = et_dmap_lst_t_student[0]  # The cell detection prediction
                et_dmap_cls_t_student = et_dmap_lst_t_student[1]  # The cell classification prediction

                et_det_sig_t_student = criterion_sig(et_dmap_det_t_student)
                et_cls_sig_t_student = criterion_softmax(et_dmap_cls_t_student)

                outputs_det_student.append(et_det_sig_t_student)
                outputs_cls_student.append(et_cls_sig_t_student)

            outputs_det_student = torch.stack(outputs_det_student, dim=0)
            outputs_cls_student = torch.stack(outputs_cls_student, dim=0)

            et_det_sig_t_student = torch.mean(outputs_det_student, dim=0)
            et_cls_sig_t_student = torch.mean(outputs_cls_student, dim=0)

            en_det_t_student = -1.0 * torch.sum(
                et_det_sig_t_student * torch.log(et_det_sig_t_student + 1e-6)
                + (1 - et_det_sig_t_student) * torch.log((1 - et_det_sig_t_student) + 1e-6),
                dim=1, keepdim=True)
            en_cls_t_student = -1.0 * torch.sum(
                et_cls_sig_t_student * torch.log(et_cls_sig_t_student + 1e-6),
                dim=1, keepdim=True)

            et_det_sig_t_student = et_det_sig_t_student.to(device_student).detach()
            et_cls_sig_t_student = et_cls_sig_t_student.to(device_student).detach()

            # 3.generate pseudo label
            pseudo_dmap_det = torch.where(et_det_sig_t_teacher > 0.5,
                                          torch.ones_like(et_det_sig_t_teacher),
                                          torch.zeros_like(et_det_sig_t_teacher))
            pseudo_dmap_cls = (et_cls_sig_t_teacher == et_cls_sig_t_teacher.max(dim=1, keepdim=True)[0]).to(
                dtype=torch.int8)

            pseudo_dmap_det_student = torch.where(et_det_sig_t_student > 0.5,
                                                  torch.ones_like(et_det_sig_t_student),
                                                  torch.zeros_like(et_det_sig_t_student))
            pseudo_dmap_cls_student = (et_cls_sig_t_student == et_cls_sig_t_student.max(dim=1, keepdim=True)[0]).to(
                dtype=torch.int8)

            # 4.estimate uncertainty
            uncertainty_det = torch.where(pseudo_dmap_det == pseudo_dmap_det_student,
                                          torch.minimum(en_det_t_teacher, en_det_t_student), en_det_t_teacher)
            uncertainty_det = torch.where(
                torch.logical_and(pseudo_dmap_det != pseudo_dmap_det_student, en_det_t_teacher > en_det_t_student),
                2 * torch.ones_like(en_det_t_teacher), uncertainty_det)

            uncertainty_cls = torch.where(
                pseudo_dmap_cls.argmax(dim=1, keepdim=True) == pseudo_dmap_cls_student.argmax(dim=1, keepdim=True),
                torch.minimum(en_cls_t_teacher, en_cls_t_student), en_cls_t_teacher)
            uncertainty_cls = torch.where(
                torch.logical_and(
                    pseudo_dmap_cls.argmax(dim=1, keepdim=True) != pseudo_dmap_cls_student.argmax(dim=1, keepdim=True),
                    en_cls_t_teacher > en_cls_t_student),
                2 * torch.ones_like(en_cls_t_teacher), uncertainty_cls)

            uncertainty_mask_det = torch.where(uncertainty_det < thresh_uncertainty * math.log(2),
                                               torch.ones_like(et_det_sig_t_teacher),
                                               torch.zeros_like(et_det_sig_t_teacher))
            uncertainty_mask_cls = torch.where(uncertainty_cls < thresh_uncertainty * math.log(n_classes),
                                               torch.ones_like(et_det_sig_t_teacher),
                                               torch.zeros_like(et_det_sig_t_teacher))

            pseudo_dmap_det = pseudo_dmap_det.detach()
            pseudo_dmap_cls = pseudo_dmap_cls.detach()
            uncertainty_mask_det = uncertainty_mask_det.detach()
            uncertainty_mask_cls = uncertainty_mask_cls.detach()

        # 5.student predict target sample with strong augment
        model_student.train()
        img_t = perturb(img_t)
        et_dmap_lst_t_student = model_student(img_t)
        et_dmap_det_t_student = et_dmap_lst_t_student[0]  # The cell detection prediction
        et_dmap_cls_t_student = et_dmap_lst_t_student[1]  # The cell classification prediction
        feature_decoded_t = et_dmap_lst_t_student[2]

        # Apply Sigmoid and Softmax activations to the detection and classification predictions, respectively.
        et_det_sig_t_student = criterion_sig(et_dmap_det_t_student)
        et_cls_sig_t_student = criterion_softmax(et_dmap_cls_t_student)
        et_det_sig_t_student = et_det_sig_t_student * uncertainty_mask_det
        et_cls_sig_t_student = et_cls_sig_t_student * uncertainty_mask_cls * pseudo_dmap_det
        pseudo_dmap_det = pseudo_dmap_det * uncertainty_mask_det
        pseudo_dmap_cls = pseudo_dmap_cls * uncertainty_mask_cls * pseudo_dmap_det

        # Compute Dice loss on the detection and classification predictions
        intersection = (et_cls_sig_t_student * pseudo_dmap_cls).sum()
        union = (et_cls_sig_t_student ** 2).sum() + (pseudo_dmap_cls ** 2).sum()
        loss_cls_t = 1 - ((2 * intersection + 1) / (union + 1))

        intersection = (et_det_sig_t_student * pseudo_dmap_det.unsqueeze(0)).sum()
        union = (et_det_sig_t_student ** 2).sum() + (pseudo_dmap_det.unsqueeze(0) ** 2).sum()
        loss_det_t = 1 - ((2 * intersection + 1) / (union + 1))

        loss_t = alpha * loss_det_t + loss_cls_t

        # training on source domain
        img_s = perturb(img_s)
        img_s = img_s.to(device_student)
        # Convert ground truth maps to binary mask (in case they were density maps)
        gt_dmap_s = gt_dmap_s > 0
        # Get the detection ground truth maps from the classes ground truth maps
        gt_dmap_det_s = gt_dmap_s.max(1)[0]
        # Set datatype and move to GPU
        gt_dmap_s = gt_dmap_s.type(torch.FloatTensor)
        gt_dmap_det_s = gt_dmap_det_s.type(torch.FloatTensor)
        gt_dmap_s = gt_dmap_s.to(device_student)
        gt_dmap_det_s = gt_dmap_det_s.to(device_student)

        # forward propagation
        et_dmap_lst_s_student = model_student(img_s)
        et_dmap_det_s_student = et_dmap_lst_s_student[0]  # The cell detection prediction
        et_dmap_cls_s_student = et_dmap_lst_s_student[1]  # The cell classification prediction
        feature_decoded_s = et_dmap_lst_s_student[2]

        # Apply Sigmoid and Softmax activations to the detection and classification predictions, respectively.
        et_det_sig_s = criterion_sig(et_dmap_det_s_student)
        et_cls_sig_s = criterion_softmax(et_dmap_cls_s_student)

        # Compute Dice loss on the detection and classification predictions
        intersection = (et_cls_sig_s * gt_dmap_s).sum()
        union = (et_cls_sig_s ** 2).sum() + (gt_dmap_s ** 2).sum()
        loss_cls_s = 1 - ((2 * intersection + 1) / (union + 1))

        intersection = (et_det_sig_s * gt_dmap_det_s.unsqueeze(0)).sum()
        union = (et_det_sig_s ** 2).sum() + (gt_dmap_det_s.unsqueeze(0) ** 2).sum()
        loss_det_s = 1 - ((2 * intersection + 1) / (union + 1))

        loss_s = alpha * loss_det_s + loss_cls_s

        # Backpropagation
        loss_student = 0.5 * loss_s + 0.5 * loss_t
        epoch_loss_sum += loss_student.item()

        optimizer_student.zero_grad()
        loss_student.backward()
        optimizer_student.step()

        # train discriminator
        for param in model_student.parameters():
            param.requires_grad = False
        for param in model_discriminator.parameters():
            param.requires_grad = True

        feature_decoded_s = feature_decoded_s.detach()
        feature_decoded_t = feature_decoded_t.detach()

        out_s = model_discriminator(feature_decoded_s)
        loss_discriminator = bce_loss(out_s, torch.FloatTensor(out_s.data.size()).fill_(source_domain_label).to(
            device_teacher))
        optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        optimizer_discriminator.step()

        out_t = model_discriminator(feature_decoded_t)
        loss_discriminator = bce_loss(out_t, torch.FloatTensor(out_t.data.size()).fill_(target_domain_label).to(
            device_teacher))
        optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        optimizer_discriminator.step()

        for param in model_student.parameters():
            param.requires_grad = True
        for param in model_discriminator.parameters():
            param.requires_grad = False

        # Performance Estimation
        PE_student += alpha * torch.mean(torch.exp(- en_det_t_student)) + torch.mean(torch.exp(- en_cls_t_student))

    return epoch_loss_sum / train_count,  PE_student / train_count
