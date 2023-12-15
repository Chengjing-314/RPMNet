from rpmnet_inference import RPMNetInference
import torch
import numpy as np
import open3d as o3d
import os


def main():

    rpmnet = RPMNetInference('./model-best-mnet.pth')

    pc_path = '/home/chengjing/Desktop/pc_reg_dataset'

    tgt_pc = o3d.io.read_point_cloud(os.path.join(pc_path, 'box_tgt.ply'))
    src_pc = o3d.io.read_point_cloud(os.path.join(pc_path, 'pc4.ply'))


    tgt_points = np.asarray(tgt_pc.points)
    tgt_points = (tgt_points - np.mean(tgt_points, axis=0)) / 1000
    tgt_pc.points = o3d.utility.Vector3dVector(tgt_points)



    tgt_pc_nm = np.asarray(tgt_pc.points).astype(np.float32)
    src_pc_nm = np.asarray(src_pc.points).astype(np.float32)

    tgt_pc_nm, tgt_scale, tgt_centroid = rpmnet._preprocess_pc(tgt_pc_nm)
    src_pc_nm, src_scale, src_centroid = rpmnet._preprocess_pc(src_pc_nm)

    tgt_pc.points = o3d.utility.Vector3dVector(tgt_pc_nm + np.array([5,0.5,1]))
    src_pc.points = o3d.utility.Vector3dVector(src_pc_nm)

    tgt_pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    tgt_pc.orient_normals_consistent_tangent_plane(30)

    src_pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    src_pc.orient_normals_consistent_tangent_plane(30)

    tgt_pcd = np.asarray(tgt_pc.points).astype(np.float32)
    src_pcd = np.asarray(src_pc.points).astype(np.float32)

    tgt_normals = np.asarray(tgt_pc.normals).astype(np.float32)
    src_normals = np.asarray(src_pc.normals).astype(np.float32)

    tgt_uniform_dsp_idx = np.random.choice(len(tgt_pcd), 2048)
    src_uniform_dsp_idx = np.random.choice(len(src_pcd), 2048)

    tgt_pcd = tgt_pcd[tgt_uniform_dsp_idx]
    src_pcd = src_pcd[src_uniform_dsp_idx]

    tgt_normals = tgt_normals[tgt_uniform_dsp_idx]
    src_normals = src_normals[src_uniform_dsp_idx]


    tgt_normed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pcd))
    src_normed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pcd))

    o3d.visualization.draw_geometries([tgt_normed, src_normed])
    

    tgt_pc_with_normals = np.concatenate((tgt_pcd, tgt_normals), axis=1)
    src_pc_with_normals = np.concatenate((src_pcd, src_normals), axis=1)

    transform = rpmnet.inference(src_pc_with_normals, tgt_pc_with_normals, 5)

    tgt_pts = np.asarray(tgt_pc.points)
    src_pts = np.asarray(src_pc.points)

    rpmnet._visualization(src_pts, tgt_pts, transform)



if __name__ == '__main__':

    main()
