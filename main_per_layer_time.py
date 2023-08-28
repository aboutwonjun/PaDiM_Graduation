import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18, resnet50
import datasets.mvtec as mvtec

import sys
# sys.path.append('/home/wonjun/Desktop/PaDiM-Anomaly-Detection-Localization-master-main')
from sklearn.manifold import TSNE
import pdb

import pandas as pd 
import time 
import logging
import psutil

import os

#Model zoo Version
# # from segmentation.data_loader.segmentation_dataset import SegmentationDataset
# # from segmentation.data_loader.transform import Rescale, ToTensor
# # from segmentation.trainer import Trainer
# # from segmentation.predict import *
# from segmentation.models import all_models
# # from util.logger import Logger

#Segmentation Model
# import segmentation_models_pytorch as smp 

from torchvision.models.segmentation import fcn_resnet50



# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


#arguments 지정 
def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='datasets/MVtec')
    parser.add_argument('--save_path', type=str, default='Result_graduation_new/result')
    parser.add_argument('--selection', type=str, default='')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2', 'fcn_resnet50', 'resnet50', 'resnet18_weights'], default='wide_resnet50_2')
    parser.add_argument('--layer', nargs='+', type=str, default=['layer1', 'layer2', 'layer3'])
    return parser.parse_args()



# if pretrained and fixed_feature: #fine tunning
#         params_to_update = model.parameters()
#         print("Params to learn:")
#         params_to_update = []
#         for name, param in model.named_parameters():
#             if param.requires_grad == True:
#                 params_to_update.append(param)
#                 print("\t", name)
#         optimizer = torch.optim.Adadelta(params_to_update)
# else:


def main():

    args = parse_args()
    
    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 0
        d = 0

        #random
        #Full Channel 
        if args.selection == 'full':
            if 'layer1' in args.layer:
                t_d += 64
                d += 64
            if 'layer2' in args.layer:
                t_d += 128
                d += 128
            if 'layer3'in args.layer:
                t_d += 256
                d += 256
        else:
            if 'layer1' in args.layer:
                t_d += 64
                d += 16
            if 'layer2' in args.layer:
                t_d += 128
                d += 32
            if 'layer3'in args.layer:
                t_d += 256
                d += 64


    elif args.arch == 'resnet50':
        model = resnet50(pretrained=True, progress=True)
        t_d = 0
        d = 0
        if args.selection == 'full':
            if 'layer1' in args.layer:
                t_d += 256
                d += 256
            if 'layer2' in args.layer:
                t_d += 512
                d += 512
            if 'layer3' in args.layer:
                t_d += 1024
                d += 1024
        else:
            if 'layer1' in args.layer:
                t_d += 256
                d += 64
            if 'layer2' in args.layer:
                t_d += 512
                d += 128
            if 'layer3' in args.layer:
                t_d += 1024
                d += 256

    elif args.arch == 'fcn_resnet50':
        model = fcn_resnet50(pretrained=True, progress=True)
        
        t_d = 0
        d = 0
        if 'layer1' in args.layer:
            t_d += 256
            d += 100
        if 'layer2' in args.layer:
            t_d += 512
            d += 150
        if 'layer3' in args.layer:
            t_d += 1024
            d += 300
            
    if args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 0
        d = 0

        if args.selection == 'full':
            if 'layer1' in args.layer:
                t_d += 256
                d += 256
            if 'layer2' in args.layer:
                t_d += 512
                d += 512
            if 'layer3' in args.layer:
                t_d += 1024
                d += 1024

        else:
            if 'layer1' in args.layer:
                t_d += 256
                d += 100
            if 'layer2' in args.layer:
                t_d += 512
                d += 150
            if 'layer3'in args.layer:
                t_d += 1024
                d += 300

        
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)
    
    if args.arch == 'fcn_resnet50':
        if 'layer1' in args.layer:
            model.backbone.layer1[-1].register_forward_hook(hook)
        if 'layer2' in args.layer:
            model.backbone.layer2[-1].register_forward_hook(hook)
        if 'layer3' in args.layer:
            model.backbone.layer3[-1].register_forward_hook(hook)

    else:
        if 'layer1' in args.layer:
            model.layer1[-1].register_forward_hook(hook)
        if 'layer2' in args.layer:
            model.layer2[-1].register_forward_hook(hook)
        if 'layer3' in args.layer:
            model.layer3[-1].register_forward_hook(hook)

    lname = ''
    if 'layer1' in args.layer:
        lname += '1'
    if 'layer2' in args.layer:
        lname += '2'
    if 'layer3' in args.layer:
        lname += '3'
    os.makedirs(os.path.join(args.save_path, 'temp_%s_L%s' % (args.arch, lname)), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []


    #시간 측정 
    init_time = time.time()
    init_memory = psutil.Process(os.getpid()).memory_info().rss
    process = psutil.Process(os.getpid())
    

    for class_name in mvtec.CLASS_NAMES:

        train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        #시간 및 메모리 측정 
        start_time = time.time()
        start_memory_class = psutil.Process(os.getpid()).memory_info().rss

        inference_times = []


        for fidx in range(0, len(args.layer)):
            if fidx == 0:
                train_outputs = OrderedDict([(args.layer[fidx], [])])
                test_outputs = OrderedDict([(args.layer[fidx], [])])
            else:
                train_outputs.update([(args.layer[fidx], [])])
                test_outputs.update([(args.layer[fidx], [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s_L%s' % (args.arch, lname), 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)
            # Embedding concat
            # embedding_vectors = train_outputs['layer1']
            # embedding_vectors = train_outputs['layer2']
            # embedding_vectors = train_outputs['layer3']
            embedding_vectors = train_outputs[args.layer[0]]
            for fidx in range(0, len(args.layer)): #['layer2', 'layer3']:
                if fidx == 0:
                    pass
                else:
                    embedding_vectors = embedding_concat(embedding_vectors, train_outputs[args.layer[fidx]])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        
        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
           #각 클래스 테스트 타임 추가 
            test_start_time = time.time()
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = test_outputs[args.layer[0]]
        for fidx in range(0, len(args.layer)):
            if fidx == 0:
                pass
            else:
                embedding_vectors = embedding_concat(embedding_vectors, test_outputs[args.layer[fidx]])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))

        
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        test_end_time = time.time()  # Added line
        inference_times.append(test_end_time - test_start_time)  # Added line



        end_time = time.time()
        final_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_cost = abs(final_memory - start_memory_class)
        inference_time = end_time - start_time
        avg_inference_time = sum(inference_times) / len(inference_times)  # Added line


        print(f"Inference time for class {class_name}: {inference_time} seconds.")
        print(f"avg.Inference time for class {class_name}: {avg_inference_time} seconds.")
        print(f"Memory cost for class {class_name}: {memory_cost / (1024 ** 2)} MB.")
        



        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = args.save_path + '/' + f'pictures_{args.arch}_{args.layer}'
        os.makedirs(save_dir, exist_ok=True)
        # plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name, test_outputs, args)
    
        #####################################
        import datetime
        '''
        class_name = "example_class"
        img_roc_auc = 0.98
        per_pixel_rocauc = 0.96
        # '''

        total_time = time.time() - init_time
        total_memory = psutil.Process(os.getpid()).memory_info().rss - init_memory


    
        log_dir = os.path.join(args.save_path, '{}_L{}_logfiles'.format(args.arch, lname))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

   
        fname = "%s/%s_L%s_%s_logfile.txt" % (log_dir,  args.arch, lname, class_name)
        with open(fname, "a") as f:
            current_time = str(datetime.datetime.now())
            f.write(f"{current_time} - Class Name: {class_name} - Total Inference Time: {total_time:.3f} seconds\n")
            f.write(f"{current_time} - Class Name: {class_name} - Average Inference Time: {avg_inference_time:.3f} seconds\n")  # Added line
            f.write(f"{current_time} - Class Name: {class_name} - Memory Cost: {total_memory / 1024 ** 2:.3f} MB\n")
            f.write(f"{current_time} - Class Name: {class_name} - Image ROCAUC: {img_roc_auc:.3f}\n")
            f.write(f"{current_time} - Class Name: {class_name} - Per Pixel ROCAUC: {per_pixel_rocauc:.3f}\n")

       
    avg_ROCAUC = np.mean(total_roc_auc)
    print('Average ROCAUC: %.3f' % avg_ROCAUC)
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % avg_ROCAUC)
    fig_img_rocauc.legend(loc="lower right")

    pix_ROCAUC = np.mean(total_pixel_roc_auc)
    print('Average pixel ROCUAC: %.3f' % pix_ROCAUC)
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % pix_ROCAUC)
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, f"{args.arch}_L{lname}_roc_curve.png"), dpi=100)  
    
    



def plot_fig(test_img, scores, gts, threshold, save_dir, class_name, test_outputs, args):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        
        # for fidx in range(len(args.layer)-1, -1, -1):
        #     mean_t = test_outputs[args.layer[fidx]][i]
        #     C, H, W = mean_t.size()
        #     mean_t = mean_t.view(C, H*W)
        #     tsne_points_test = np.array(tsne.fit_transform(np.array(mean_t)))
        #     ax_img[5].scatter(tsne_points_test[:,0], tsne_points_test[:,1], s=1, label=fidx)
        
        #     ax_img[5].grid(True)
        
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    main()
