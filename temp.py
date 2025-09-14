import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

import datasets, hopenet
import util.uitls as utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_bins', dest='num_bins', help='Number of Bins',
          default=66, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=1, type=int)
    parser.add_argument('--save_viz', dest='save_viz', help='Save images with pose cube.',
          default=False, type=bool)
    parser.add_argument('--output_dir', dest='output_dir', help='Path to output_dir',
          default='', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = os.path.join(os.getcwd(),args.snapshot)
    output_path = os.path.join(os.getcwd(), args.output_dir ,'test_results')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    plots_path = os.path.join(output_path, "plots")
    os.makedirs(plots_path, exist_ok=True)
    
    num_bins = args.num_bins
    
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins)
    
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path, weights_only=True)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)

    print('Loading data.')
    transformations = transforms.Compose([transforms.Resize(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_filename_list = os.path.join(args.filename_list ,'test_filename.npy')
    pose_dataset = datasets.AFLW2000(args.data_dir, num_bins, test_filename_list, transformations)

    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=4)
    print('Ready to test network.')
    model.eval()
    total = 0

    idx_tensor = [idx for idx in range(num_bins)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    l1loss = torch.nn.L1Loss(reduction='sum')

    yaw_errors_list, pitch_errors_list, roll_errors_list = [], [], []
    yaw_preds_list, pitch_preds_list, roll_preds_list = [], [], []
    yaw_labels_list, pitch_labels_list, roll_labels_list = [], [], []

    for i, (images, labels, cont_labels, name) in enumerate(test_loader):
        images = Variable(images).cuda(gpu)
        total += cont_labels.size(0)

        label_yaw = cont_labels[:,0].float()
        label_pitch = cont_labels[:,1].float()
        label_roll = cont_labels[:,2].float()

        yaw, pitch, roll = model(images)

        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

        yaw_err = torch.sum(torch.abs(yaw_predicted - label_yaw))
        yaw_error += yaw_err
        pitch_err = torch.sum(torch.abs(pitch_predicted - label_pitch))
        pitch_error += pitch_err
        roll_err = torch.sum(torch.abs(roll_predicted - label_roll))
        roll_error += roll_err

        yaw_errors_list.extend(torch.abs(yaw_predicted - label_yaw).numpy())
        pitch_errors_list.extend(torch.abs(pitch_predicted - label_pitch).numpy())
        roll_errors_list.extend(torch.abs(roll_predicted - label_roll).numpy())

        yaw_preds_list.extend(yaw_predicted.numpy())
        pitch_preds_list.extend(pitch_predicted.numpy())
        roll_preds_list.extend(roll_predicted.numpy())

        yaw_labels_list.extend(label_yaw.numpy())
        pitch_labels_list.extend(label_pitch.numpy())
        roll_labels_list.extend(label_roll.numpy())
        
        if (i+1) % 10 == 0:
            print('Iter [%d/%d] MAE Error: Yaw %.4f   ||   Pitch %.4f  ||   Roll %.4f'
                       %( i+1, len(pose_dataset)//args.batch_size, yaw_err.item(), pitch_err.item(), roll_err.item()))

        if args.save_viz:
            name = name[0].split('.')[0]
            cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg'))
            
            if args.batch_size == 1:
                error_string = 'y %.2f, p %.2f, r %.2f' % (torch.sum(torch.abs(yaw_predicted - label_yaw)), torch.sum(torch.abs(pitch_predicted - label_pitch)), torch.sum(torch.abs(roll_predicted - label_roll)))
                cv2.putText(cv2_img, error_string, (30, cv2_img.shape[0]- 30), fontFace=1, fontScale=1, color=(0,0,255), thickness=2)
            utils.draw_axis(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], tdx = 200, tdy= 200, size=100)
            cv2.imwrite(os.path.join(output_path, name + '.jpg'), cv2_img)

    yaw_err = yaw_error / total
    pitch_err = pitch_error / total
    roll_err = roll_error / total
    
    total_err = (yaw_err + pitch_err + roll_err)/3
    print('Test error in degrees of the model on the ' + str(total) +
    ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, total_error: %.4f' %( yaw_err,
    pitch_err, roll_err, total_err))

    metrics = utils.compute_metrics(yaw_errors_list, pitch_errors_list, roll_errors_list)

    utils.generate_plots(yaw_errors_list, pitch_errors_list, roll_errors_list,
                         yaw_labels_list, pitch_labels_list, roll_labels_list,
                         yaw_preds_list, pitch_preds_list, roll_preds_list,
                         output_path)
