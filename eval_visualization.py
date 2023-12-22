import numpy as np 
import torch
from rpmnet_inference import RPMNetInference
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import src.data_loader.transforms as Transforms
import torchvision
from src.common.torch import dict_all_to_device, CheckPointManager, to_numpy
from collections import defaultdict
import json
import os
import pickle
import time
from typing import Dict, List

import numpy as np
import open3d  # Need to import before torch
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import torch

from src.arguments import rpmnet_eval_arguments
from src.common.misc import prepare_logger
from src.common.torch import dict_all_to_device, CheckPointManager, to_numpy
from src.common.math import se3
from src.common.math_torch import se3
from src.common.math.so3 import dcm2euler
from src.data_loader.datasets import get_test_datasets
import models.rpmnet
import open3d as o3d
import h5py
from scipy.spatial.transform import Rotation


def load_pc(filename):
    pc = o3d.io.read_point_cloud(filename)
    pc_pt = np.asarray(pc.points).astype(np.float32)
    normals = np.asarray(pc.normals).astype(np.float32)
    pc_with_normals = np.concatenate((pc_pt, normals), axis=-1) 
    return pc_with_normals

def main():
    #load original point cloud 

    test_transforms = [    Transforms.SplitSourceRef(),
                           Transforms.RandomCrop((0.3,1), True, 'y'),
                           Transforms.RandomTransformSE3_euler(rot_mag=120, trans_mag=0.7),
                           Transforms.Resampler(1024),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    
    pc_with_normals = load_pc('/home/chengjing/Desktop/RPMNet/ycb/071_nine_hole_peg_test/clouds/normalized_merged_cloud.ply')


    pc_with_normals = pc_with_normals[np.random.choice(pc_with_normals.shape[0], 2048, replace=False), :]

   
    model_file_path = 'model-best_new.pth'
    

    test_transforms = torchvision.transforms.Compose(test_transforms)

    #load model
    eval_arg = rpmnet_eval_arguments().parse_args()
    eval_arg.resume = model_file_path

    model = models.rpmnet.get_model(eval_arg).to('cuda')
    model.eval()

    #load data
    sample = {'points': pc_with_normals}
    sample = test_transforms(sample)
    sample['points_src'] = torch.from_numpy(sample['points_src']).unsqueeze(0)
    sample['points_ref'] = torch.from_numpy(sample['points_ref']).unsqueeze(0)
    dict_all_to_device(sample, 'cuda')

    #inference
    with torch.no_grad():
        transform, _ = model(sample, num_iter=5)

    transform = transform[-1].squeeze(0).cpu().numpy()

    #transform to euler angles
    r = Rotation.from_matrix(transform[0:3, 0:3])
    angle_pred = r.as_euler('xyz', degrees=True)
    r_gt = Rotation.from_matrix(sample['transform_gt'][0:3, 0:3])
    angle_gt = r_gt.as_euler('xyz', degrees=True)

    print(f'angle_pred: {angle_pred}, angle_gt: {angle_gt}')
    print('r_mae: ', np.mean(np.abs(angle_pred - angle_gt)))

    #visualize

    #transform point cloud


def test(model):
    #load npy

    data = np.load('/home/chengjing/Desktop/RPMNet/eval_results/val_data.npy', allow_pickle=True)

    eval_pred = np.load('/home/chengjing/Desktop/RPMNet/eval_results/pred_transforms.npy', allow_pickle=True)


    print(data.shape, eval_pred.shape)
  

    data = dict(data[()])

    pts_src = data['points_src']

    pts_ref = data['points_ref']

    idx = 4

    eval_pred = np.squeeze(eval_pred)[idx][-1]
    pts_src_single = pts_src[idx].cpu().numpy()
    pts_ref_single = pts_ref[idx].cpu().numpy()
    pts_gt_single = data['transform_gt'][idx].cpu().numpy()

    
    print(pts_src_single.shape, pts_ref_single.shape)

    #visualize

    pc_src = o3d.geometry.PointCloud()
    pc_ref = o3d.geometry.PointCloud()

    pc_src.points = o3d.utility.Vector3dVector(pts_src_single[:, :3])
    pc_ref.points = o3d.utility.Vector3dVector(pts_ref_single[:, :3])

    print(pts_src_single.shape)

    pc_src.paint_uniform_color([1, 0, 0])
    pc_ref.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pc_src, pc_ref])


    #load model

    # model_file_path = '/home/chengjing/Desktop/RPMNet/model-best_new.pth'
    # eval_arg = rpmnet_eval_arguments().parse_args()
    # eval_arg.resume = model_file_path
    # eval_arg.view_crop = True

    # model = models.rpmnet.get_model(eval_arg)
    # model.to('cuda')

    model.eval()

    #inference

    with torch.no_grad():
    
        data_t = {'points_src': torch.tensor(pts_src_single).unsqueeze(0), 'points_ref': torch.tensor(pts_ref_single).unsqueeze(0)}

        dict_all_to_device(data_t, 'cuda')

        transform, _ = model(data_t, num_iter=5)

        transform = transform[-1].squeeze(0).cpu().numpy()

        # print(len(transform))
        # print(transform[-1].shape)

        #visualize

        #transform point cloud

        transform = np.concatenate([transform, np.array([[0, 0, 0, 1]])], axis=0)

        pc_src = o3d.geometry.PointCloud()
        pc_ref = o3d.geometry.PointCloud()

        pc_src.points = o3d.utility.Vector3dVector(pts_src_single[:, :3])
        pc_ref.points = o3d.utility.Vector3dVector(pts_ref_single[:, :3])

        pc_src.paint_uniform_color([1, 0, 0])
        pc_ref.paint_uniform_color([0, 1, 0])

        pc_src.transform(transform)

        o3d.visualization.draw_geometries([pc_src, pc_ref])

        #transform to euler angles

        r = Rotation.from_matrix(transform[0:3, 0:3])
        angle_pred = r.as_euler('xyz', degrees=True)

        r_gt = Rotation.from_matrix(pts_gt_single[0:3, 0:3])
        angle_gt = r_gt.as_euler('xyz', degrees=True)


        eval_r = Rotation.from_matrix(eval_pred[0:3, 0:3])
        eval_angle_pred = eval_r.as_euler('xyz', degrees=True)

        eval_transform = np.concatenate([eval_pred, np.array([[0, 0, 0, 1]])], axis=0)
        pc_src_eval = o3d.geometry.PointCloud()
        pc_src_eval.points = o3d.utility.Vector3dVector(pts_src_single[:, :3])
        pc_src_eval.paint_uniform_color([1, 0, 0])
        pc_src_eval.transform(eval_transform)

        o3d.visualization.draw_geometries([pc_src_eval, pc_ref])


        print(f'angle_pred: {angle_pred}, angle_gt: {angle_gt}')
        print('r_mae: ', np.mean(np.abs(angle_pred - angle_gt)))
        print('eval_r_mae: ', np.mean(np.abs(eval_angle_pred - angle_gt)))


    print("--------------------batch_size of 8")

    with torch.no_grad():

        data = np.load('/home/chengjing/Desktop/RPMNet/eval_results/val_data.npy', allow_pickle=True)
        data = dict(data[()])

        dict_all_to_device(data, 'cuda')

        transform, _ = model(data, num_iter=5)

        transform =to_numpy(torch.stack(transform, dim=1))

def get_model():
    if _args.method == 'rpmnet':
        assert _args.resume is not None
        model = models.rpmnet.get_model(_args)
        model.to('cuda')
        saver = CheckPointManager(os.path.join('/home/chengjing/Desktop/RPMNet/temp', 'ckpt', 'models'))
        saver.load(_args.resume,'cuda', model)
    else:
        raise NotImplementedError
    return model


def rpmnet_inference_main():

    test_transforms = [ Transforms.SplitSourceRef(),
                        Transforms.RandomCrop((0.3,1), True, 'y'),
                        Transforms.RandomTransformSE3_euler(rot_mag=120, trans_mag=0.7),
                        Transforms.Resampler(1024),
                        Transforms.RandomJitter(),
                        Transforms.ShufflePoints()]

    
    pc_with_normals = load_pc('/home/chengjing/Desktop/RPMNet/ycb/004_sugar_box/clouds/normalized_merged_cloud.ply')


    rpmnet = RPMNetInference('/home/chengjing/Desktop/RPMNet/model-best_new.pth')


    test_transforms = torchvision.transforms.Compose(test_transforms)

    data = {'points': pc_with_normals}
    data = test_transforms(data)


    #visualize
    pc_src = o3d.geometry.PointCloud()
    pc_ref = o3d.geometry.PointCloud()

    pc_src.points = o3d.utility.Vector3dVector(data['points_src'][:, :3])
    pc_ref.points = o3d.utility.Vector3dVector(data['points_ref'][:, :3])

    pc_src.paint_uniform_color([1, 0, 0])
    pc_ref.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pc_src, pc_ref])



    transforms = rpmnet.inference(data['points_src'], data['points_ref'], num_iter=5)
    gt_transform = data['transform_gt']

    print(transforms, gt_transform)

    #transform to euler angles
    r = Rotation.from_matrix(transforms[0:3, 0:3])
    angle_pred = r.as_euler('xyz', degrees=True)
    r_gt = Rotation.from_matrix(gt_transform[0:3, 0:3])
    angle_gt = r_gt.as_euler('xyz', degrees=True)

    print(f'angle_pred: {angle_pred}, angle_gt: {angle_gt}')
    print('r_mae: ', np.mean(np.abs(angle_pred - angle_gt)))


    transforms = np.concatenate([transforms, np.array([[0, 0, 0, 1]])], axis=0)

    #visualize

    src_pc = o3d.geometry.PointCloud()
    ref_pc = o3d.geometry.PointCloud()

    src_pc.points = o3d.utility.Vector3dVector(data['points_src'][:, :3])

    ref_pc.points = o3d.utility.Vector3dVector(data['points_ref'][:, :3])

    src_pc.paint_uniform_color([1, 0, 0])
    ref_pc.paint_uniform_color([0, 1, 0])

    src_pc.transform(transforms)

    o3d.visualization.draw_geometries([src_pc, ref_pc])










        
if __name__ == '__main__':
    # main()
    parser = rpmnet_eval_arguments()
    _args = parser.parse_args()

    model = get_model()

    test(model)

    # rpmnet_inference_main()

