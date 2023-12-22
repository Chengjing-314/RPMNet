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


class RPMNetInference:

    def __init__(self, model_file_path, device = 'cuda'):
        self.device = device 
        eval_args = rpmnet_eval_arguments().parse_args('')
        eval_args.resume = model_file_path
        #args is just to get the model
        self.model = models.rpmnet.get_model(eval_args).to(self.device)

        checkpoint_manager = CheckPointManager('/home/chengjing/Desktop/RPMNet/temp')
        checkpoint_manager.load(eval_args.resume, 'cuda', self.model)


    def inference(self, src_pc, tgt_pc, num_iter=5):

        src_pc = src_pc.astype(np.float32)
        tgt_pc = tgt_pc.astype(np.float32)
        
        if tgt_pc.shape[0] > 1024:
            tgt_pc = tgt_pc[np.random.choice(tgt_pc.shape[0], 1024, replace=False), :]
        else:
            needed_points = 1024 - tgt_pc.shape[0]
            additional_points = tgt_pc[np.random.choice(tgt_pc.shape[0], needed_points, replace=True), :]
            tgt_pc = np.concatenate([tgt_pc, additional_points], axis=0)
    
        if src_pc.shape[0] > 1024:
            src_pc = src_pc[np.random.choice(src_pc.shape[0], 1024, replace=False), :]
        else:
            needed_points = 1024 - src_pc.shape[0]
            additional_points = src_pc[np.random.choice(src_pc.shape[0], needed_points, replace=True), :]
            src_pc = np.concatenate([src_pc, additional_points], axis=0)
            
            
        print(f'src_pc.shape: {src_pc.shape}, tgt_pc.shape: {tgt_pc.shape} before entering rpmnet inference')

        src_pc = torch.from_numpy(src_pc).unsqueeze(0)
        tgt_pc = torch.from_numpy(tgt_pc).unsqueeze(0) 
            

        data = {'points_src': src_pc, 'points_ref': tgt_pc}

        dict_all_to_device(data, self.device)

        with torch.no_grad():
            transform, _ = self.model(data, num_iter=num_iter)


        transform = transform[-1].cpu().numpy()

        return transform.squeeze()
    
    def _preprocess_pc(self, pc):

        centroed = np.mean(pc, axis=0)
        centroed_pc = pc - centroed
        scale = np.max(np.linalg.norm(centroed_pc, axis=1))
        pc = centroed_pc / scale
        pc_processed = pc + centroed

        return pc_processed, scale, centroed
    

    def _visualization(self, src_pc, tgt_pc, transform):

        src_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pc))
        tgt_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pc))

        src_pc.paint_uniform_color([1, 0, 0])
        tgt_pc.paint_uniform_color([0, 1, 0])


        #transform is 3x4

        print(transform.shape)

        transform = np.vstack((transform, np.array([0, 0, 0, 1])))

        src_pc.transform(transform)

        o3d.visualization.draw_geometries([src_pc, tgt_pc])

    def pipeline(self, src_pc, tgt_pc, num_iter=5, visualize=False):
            
            src_pc = src_pc.astype(np.float32)
            tgt_pc = tgt_pc.astype(np.float32)

            src_pcd = o3d.geometry.PointCloud()
            tgt_pcd = o3d.geometry.PointCloud()

            src_pcd.points = o3d.utility.Vector3dVector(src_pc)
            tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pc)


            src_pcd = src_pcd.voxel_down_sample(voxel_size=0.002)
            tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size=0.002)

            src_pc = np.asarray(src_pcd.points).astype(np.float32)
            tgt_pc = np.asarray(tgt_pcd.points).astype(np.float32)


            src_pc, src_scale, src_cent = self._preprocess_pc(src_pc)
            tgt_pc, tgt_scale, tgt_cent = self._preprocess_pc(tgt_pc)
            

            #estimate normals
            src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=15))
            tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=15))

            src_pcd.orient_normals_consistent_tangent_plane(15)
            tgt_pcd.orient_normals_consistent_tangent_plane(15)

            src_pc_pts = np.asarray(src_pcd.points)
            src_pc_normals = np.asarray(src_pcd.normals)

            tgt_pc_pts = np.asarray(tgt_pcd.points)
            tgt_pc_normals = np.asarray(tgt_pcd.normals)

            src_pc_with_normals = np.concatenate((src_pc_pts, src_pc_normals), axis=1)
            tgt_pc_with_normals = np.concatenate((tgt_pc_pts, tgt_pc_normals), axis=1)

            transform = self.inference(src_pc_with_normals, tgt_pc_with_normals, num_iter=num_iter)


            if visualize:
                self._visualization(src_pc, tgt_pc, transform)

            transform = np.vstack((transform, np.array([0, 0, 0, 1])))


            return transform






def main():

    np.random.seed(0)

    pc = np.random.rand(1024, 6)
    pc2 = np.random.rand(1024, 6)

    rpmnet = RPMNetInference('./model-best-rpmnet.pth')

    transform = rpmnet.inference(pc, pc2)

    print(transform)

    rpmnet._visualization(pc[:, :3], pc2[:, :3], transform)

    


if __name__ == '__main__':
    main()


