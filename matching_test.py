from rpmnet_inference import RPMNetInference
import torch
import numpy as np
import open3d as o3d
import os
from matplotlib import pyplot as plt

np.random.seed(0)


def main():

    rpmnet = RPMNetInference('/home/chengjing/Desktop/RPMNet/model-best_new.pth')

    pc_path = '/home/chengjing/Desktop/pc_reg_dataset'

    tgt_pc = o3d.io.read_point_cloud(os.path.join(pc_path, 'box_tgt.ply'))
    src_pc = o3d.io.read_point_cloud(os.path.join(pc_path, 'pc1.ply'))

    #recenter tgt_pc, only x,y
    tgt_points = np.asarray(tgt_pc.points)

    print(np.max(tgt_points, axis=0)-np.min(tgt_points, axis=0))

    tgt_points = (tgt_points - np.mean(tgt_points, axis=0)) / 1000
    tgt_points[:, 2] += np.abs(min(tgt_points[:, 2]))
    tgt_pc.points = o3d.utility.Vector3dVector(tgt_points)


    tgt_points += np.array([-0.1, 0.2, 0])
    

    src_points = np.asarray(src_pc.points)

    tgt_pc.points = o3d.utility.Vector3dVector(tgt_points)

    tgt_pc.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), tgt_pc.get_center())



    src_pc.paint_uniform_color([1, 0, 0])
    tgt_pc.paint_uniform_color([0, 1, 0])
    # show coordinate

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([src_pc, tgt_pc, coordinate_frame])


    transform = rpmnet.pipeline(src_points, tgt_points, 5, True)

    src_pc.transform(transform)

    src_pc.paint_uniform_color([1, 0, 0])
    tgt_pc.paint_uniform_color([0, 1, 0])

    # o3d.visualization.draw_geometries([src_pc, tgt_pc])


    





    
    #3d visualization

    # src_points = np.asarray(src_pc.points)
    # tgt_points = tgt_points[np.random.choice(len(tgt_points), 2048)]
    # plt.figure(figsize=(10, 10))
    # ax = plt.axes(projection='3d')
    # ax.scatter(tgt_points[:, 0], tgt_points[:, 1], tgt_points[:, 2], c='r', s=1)
    # ax.scatter(src_points[:, 0], src_points[:, 1], src_points[:, 2], c='b', s=1)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    # o3d.visualization.draw_geometries([tgt_pc, src_pc])

    #rotate tgt_pc alone x axis by 90 degree

    # tgt_pc.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), tgt_pc.get_center())

    # cam = [0,-1,1]
    # radius = 1000

    # _, pt_map = tgt_pc.hidden_point_removal(cam, radius)

    # tgt_pc = tgt_pc.select_by_index(pt_map)

    # o3d.visualization.draw_geometries([tgt_pc])



    # tgt_points = np.asarray(tgt_pc.points)

    # tgt_points = tgt_points[np.random.choice(len(tgt_points), 2048)]
    # plt.figure(figsize=(10, 10))
    # ax = plt.axes(projection='3d')
    # ax.scatter(tgt_points[:, 0], tgt_points[:, 1], tgt_points[:, 2], c='r', s=1)
    # # ax.scatter(src_points[:, 0], src_points[:, 1], src_points[:, 2], c='b', s=1)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()


    #normalize src point 
    # src_points = np.asarray(src_pc.points)
    # src_points = (src_points - np.mean(src_points, axis=0)) / 1000
    # scale = np.max(np.linalg.norm(src_points, axis=1))
    # src_points = src_points / scale


    # #surface reconstruction

    # src_pc.points = o3d.utility.Vector3dVector(src_points)

    # src_pc.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # src_pc.orient_normals_consistent_tangent_plane(30)

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(src_pc, depth=6, width=0, scale=1.01, linear_fit=False)[0]

    # mesh.compute_vertex_normals()

    # src_pc = mesh.sample_points_uniformly(number_of_points=2048)

    # o3d.visualization.draw_geometries([new_src_pc, src_pc])

    # src_points = src_points[src_points[:, 2] > 0.01]

    # src_pc.points = o3d.utility.Vector3dVector(src_points)

    # o3d.visualization.draw_geometries([src_pc])

    # print(np.std(src_points, axis=0), np.mean(src_points, axis=0), np.max(src_points, axis=0), np.min(src_points, axis=0))


    # tgt_pc_nm = np.asarray(tgt_pc.points).astype(np.float32)
    # src_pc_nm = np.asarray(src_pc.points).astype(np.float32)

    # tgt_pcd = tgt_pc.voxel_down_sample(voxel_size=0.005)

    # tgt_pc_nm, tgt_scale, tgt_centroid = rpmnet._preprocess_pc(tgt_pc_nm)
    # src_pc_nm, src_scale, src_centroid = rpmnet._preprocess_pc(src_pc_nm)

    # tgt_pc.points = o3d.utility.Vector3dVector(tgt_pc_nm + np.array([5,0.5,0]))
    # src_pc.points = o3d.utility.Vector3dVector(src_pc_nm)

    # tgt_pc.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # tgt_pc.orient_normals_consistent_tangent_plane(30)

    # src_pc.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # src_pc.orient_normals_consistent_tangent_plane(30)

    # tgt_pcd = np.asarray(tgt_pc.points).astype(np.float32)
    # src_pcd = np.asarray(src_pc.points).astype(np.float32)

    # tgt_normals = np.asarray(tgt_pc.normals).astype(np.float32)
    # src_normals = np.asarray(src_pc.normals).astype(np.float32)

   
    # tgt_pcd = np.asarray(tgt_pcd.points)
    

    # tgt_uniform_dsp_idx = np.random.choice(len(tgt_pcd), 2048)
    # src_uniform_dsp_idx = np.random.choice(len(src_pcd), 2048)

    # tgt_pcd = tgt_pcd[tgt_uniform_dsp_idx]
    # src_pcd = src_pcd[src_uniform_dsp_idx]

    # tgt_normals = tgt_normals[tgt_uniform_dsp_idx]
    # src_normals = src_normals[src_uniform_dsp_idx]

    # before_enter_nn_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pcd))
    # before_enter_nn_tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pcd))

    # o3d.visualization.draw_geometries([before_enter_nn_src, before_enter_nn_tgt])


    # tgt_normed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pcd))
    # src_normed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pcd))

    # o3d.visualization.draw_geometries([tgt_normed, src_normed])
    

    # tgt_pc_with_normals = np.concatenate((tgt_pcd, tgt_normals), axis=1)
    # src_pc_with_normals = np.concatenate((src_pcd, src_normals), axis=1)

    # transform = rpmnet.inference(src_pc_with_normals, tgt_pc_with_normals, 5)




if __name__ == '__main__':

    main()
