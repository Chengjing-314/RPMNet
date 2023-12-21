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
        eval_args = rpmnet_eval_arguments().parse_args()
        eval_args.resume = model_file_path
        #args is just to get the model
        self.model = models.rpmnet.get_model(eval_args).to(self.device)

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


        transform = transform[0].cpu().numpy()

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


