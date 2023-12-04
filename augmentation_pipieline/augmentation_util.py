import torch 
import numpy as np
import random
import open3d as o3d
from tqdm import tqdm


def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd = np.asarray(pcd.points)
    return pcd


def load_pcds(paths):
    pcds = []
    for path in paths:
        pcd = o3d.io.read_point_cloud(path)
        pcd = np.asarray(pcd.points)
        pcds.append(pcd)
    return pcds

def downsample(pcd:np.ndarray, voxel_size:float):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd = np.asarray(pcd.points)
    return pcd


def random_rotate(pcd:np.ndarray, angle_range:tuple):
    axis = np.random.uniform(-1, 1, size=3)
    angle = np.random.uniform(angle_range[0], angle_range[1])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    pcd = pcd.rotate(axis, angle)
    pcd = np.asarray(pcd.points)
    return pcd, axis, angle

def random_translate(pcd, translate_range):
    translate = np.random.uniform(translate_range[0], translate_range[1], size=3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    pcd = pcd.translate(translate)
    pcd = np.asarray(pcd.points)
    return pcd, translate

def random_chop(pcd, chop_range):
    chop = np.random.uniform(chop_range[0], chop_range[1], size=3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    pcd = pcd.crop(chop)
    pcd = np.asarray(pcd.points)
    return pcd, chop

def plane_cut(pcd):
    
    plane_normal = np.random.normal(size=3)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    indices = np.where(np.dot(pcd, plane_normal) < 0)[0]
    
    points = pcd[indices]
    
    return points


def gaussian_noise(pcd, sigma):
    noise = np.random.normal(0, sigma, size=pcd.shape)
    pcd = pcd + noise
    return pcd
    

def single_pcd_pipeline(file_path, num_transformation):
    pcd = load_pcd(file_path)
    pcds = []
    axis = []
    angle = []
    translate = []
    for _ in tqdm(range(num_transformation)):
        downsampled_pcd = downsample(pcd, 0.05)
        noisy_pcd = gaussian_noise(downsampled_pcd, 0.01)
        cut_pcd = plane_cut(noisy_pcd)
        rotated_pcd, axis_, angle_ = random_rotate(cut_pcd, (-np.pi, np.pi))
        translated_pcd, translate_ = random_translate(rotated_pcd, (-0.2, 0.2))
        pcds.append(translated_pcd)
        axis.append(axis_)
        angle.append(angle_)
        translate.append(translate_)
        
    return pcds, axis, angle, translate

