import numpy as np 
import torch
from rpmnet_inference import RPMNetInference
import open3d as o3d


def load_pc(filename):
    pc = o3d.io.read_point_cloud(filename)
    pc = np.asarray(pc.points).astype(np.float32)
    normals = np.asarray(pc.normals).astype(np.float32)
    pc_with_normals = np.concatenate((pc, normals), axis=-1) 
    return pc_with_normals

def view_point_crop(points, up_axis='y'):
    radius = 100
    pc = o3d.geometry.PointCloud()
    centroid = np.mean(points[:, :3], axis=0)
    points_centered = points[:, :3] - centroid
    pc.points = o3d.utility.Vector3dVector(points_centered)
    
    axis_dict = {'x': 0, 'y': 1, 'z': 2}
    up_index = axis_dict.get(up_axis, 1)  

    min_vals = np.min(points_centered, axis=0)
    max_vals = np.max(points_centered, axis=0)
    
    non_up_axis_offset_range = np.random.uniform(0.1, 0.35)
    up_axis_extra_height = np.random.uniform(1.1, 1.3)

    cam = np.ones(3)
    for i in range(3):
        if i == up_index:
            cam[i] = up_axis_extra_height + max_vals[i]
        else:
            cam[i] = np.random.uniform(min_vals[i] - non_up_axis_offset_range, max_vals[i] + non_up_axis_offset_range)

    # Perform hidden point removal
    _, idx = pc.hidden_point_removal(cam, radius)
    pc = pc.select_by_index(idx)
    new_points = np.asarray(pc.points)
    new_points = (new_points + centroid).astype(np.float32)
    normals = points[idx, 3:6]
    new_points = np.concatenate((new_points, normals), axis=-1)

    return new_points

def get_random_transform():
    rot = np.random.uniform(0, 0.6 * np.pi, 3) 

    # Generate random translation values
    trans = np.random.uniform(-2, 2, 3)

    # Create an identity matrix for the transformation
    transform = np.eye(4)

    # Calculate the rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rot[0]), -np.sin(rot[0])],
                   [0, np.sin(rot[0]), np.cos(rot[0])]])

    Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])],
                   [0, 1, 0],
                   [-np.sin(rot[1]), 0, np.cos(rot[1])]])

    Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0],
                   [np.sin(rot[2]), np.cos(rot[2]), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx

    transform[:3, :3] = R
    transform[:3, 3] = trans

    return transform


def main():
    #load original point cloud 
    
    pc_with_normals = load_pc('data/ycb/003_cracker_box/clouds/merged_cloud.ply')
    #crop it
    cropped_pc_with_normals = view_point_crop(pc_with_normals)
    #rigid trainsform original pc
    transform = get_random_transform()
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(pc_with_normals[:, :3])
    tgt_pcd.normals = o3d.utility.Vector3dVector(pc_with_normals[:, 3:6])
    tgt_pcd.transform(transform)
    
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(cropped_pc_with_normals[:, :3])
    src_pcd.normals = o3d.utility.Vector3dVector(cropped_pc_with_normals[:, 3:6])
    
    tgt_pcd_pts = np.asarray(tgt_pcd.points).astype(np.float32)
    tgt_pcd_normals = np.asarray(tgt_pcd.normals).astype(np.float32)
    
    tgt_pc_with_normals = np.concatenate((tgt_pcd_pts, tgt_pcd_normals), axis=-1)
    
    rpmnet = RPMNetInference('./model-best-rpmnet.pth')
    transform = rpmnet.inference(cropped_pc_with_normals, tgt_pc_with_normals)
    transform = np.vstack((transform, np.array([0, 0, 0, 1])))
    
    src_pcd.transform(transform)
    
    #paint color
    src_pcd.paint_uniform_color([1, 0, 0])
    tgt_pcd.paint_uniform_color([0, 1, 0])
    
    o3d.visualization.draw_geometries([src_pcd, tgt_pcd])
    
    

# 1. load original point cloud 2, crop it 3, rigid trainsform original pc , 4 run rpmnet, 5. visualize
    
    