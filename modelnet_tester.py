import h5py
import numpy as np
import open3d as o3d
import os
import json 
import matplotlib.pyplot as plt


#read modelnet40 dataset

def read_modelnet40_dataset(dataset_path, num_points=2048):

    #dataset_path = '/home/chengjing/Desktop/RPMNet/dataset/modelnet40_ply_hdf5_2048'

    train_files = [os.path.join(dataset_path, 'ply_data_train0.h5')]
    test_files = []

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    train_points_normal = []

    for f in train_files:
        file_path = os.path.join(dataset_path, f)
        data = h5py.File(file_path)
        train_points.append(data['data'][:].astype(np.float32))
        train_labels.append(data['label'][:].astype(np.int64))
        train_points_normal.append(data['normal'][:].astype(np.float32))

    for f in test_files:
        file_path = os.path.join(dataset_path, f)
        data = h5py.File(file_path)
        test_points.append(data['data'][:].astype(np.float32))
        test_labels.append(data['label'][:].astype(np.int64))

    train_points = np.concatenate(train_points, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    # test_points = np.concatenate(test_points, axis=0)
    # test_labels = np.concatenate(test_labels, axis=0)

    return train_points, train_labels, test_points, test_labels, train_points_normal

def o3d_hidden_point_removal(pcd, cam_z_min, cam_z_max, radius):

    cam = np.zeros(3)

    pts = np.asarray(pcd.points)

    # cam[0] = np.random.uniform(np.min(pts[:, 0]), np.max(pts[:, 0]))
    # cam[2] = np.random.uniform(np.min(pts[:, 1]), np.max(pts[:, 1]))

    # cam[1] = np.random.uniform(cam_z_min, cam_z_max)

    y_max = np.max(pts[:, 1])

    r = 100



    cam[0] = 0.5
    cam[1] = 0.7 + y_max
    cam[2] = 0

    # print(cam, np.random.uniform(cam_min, cam_max), np.random.uniform(cam_min, cam_max), np.random.uniform(cam_min, cam_max))

    print(cam)

    _, pt_map = pcd.hidden_point_removal(cam, r)
    pcd = pcd.select_by_index(pt_map)
    return pcd


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




tpts,_,_,_ = read_modelnet40_dataset('/home/chengjing/Desktop/RPMNet/modelnet40_ply_hdf5_2048')


# np.random.seed(4)

# fig = plt.figure(figsize=(10, 10))
# ax = plt.axes(projection='3d')
idx = np.random.choice(len(tpts))

# print(idx)
# ax.scatter(tpts[idx, :, 0], tpts[idx, :, 1], tpts[idx, :, 2], c='r', s=1)
# plt.show()

# idx = 822

#is it centered at origin?

# print(np.mean(tpts[idx], axis=0))
# print(np.max(tpts[idx], axis=0))
# print(np.min(tpts[idx], axis=0))

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tpts[idx]))

# pcd_after = o3d_hidden_point_removal(pcd, 0.7, 1.2, 1000)

pcd_after = view_point_crop(tpts[idx])

# pts = np.asarray(pcd_after.points)

# fig = plt.figure(figsize=(10, 10))
# ax = plt.axes(projection='3d')
# ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='r', s=1)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

pcd_after_pts = np.asarray(pcd_after)

pcd_after_pts += np.ones(3)

pcd_after = o3d.geometry.PointCloud()

pcd_after.points = o3d.utility.Vector3dVector(pcd_after_pts)

#color

pcd.paint_uniform_color([1, 0, 0])
pcd_after.paint_uniform_color([0, 1, 0])

o3d.visualization.draw_geometries([pcd_after, pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)])