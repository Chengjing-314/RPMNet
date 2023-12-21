import open3d as o3d 
import os 
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))

f = open(current_dir + '/ycb/objects.txt', 'r')
object_list = [line.strip() for line in f.readlines()]
f.close()

def normalize_point_cloud(pcd):
    # Center the point cloud
    pcd = pcd.voxel_down_sample(voxel_size=0.004)
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) - centroid)

    # Scale the point cloud to fit in a unit sphere
    distances = np.linalg.norm(np.asarray(pcd.points), axis=1)
    max_distance = np.max(distances)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) / max_distance)

    return pcd

for object_name in object_list:
    ply_file = os.path.join('ycb', object_name, 'clouds', 'merged_cloud.ply')
    pcd = o3d.io.read_point_cloud(ply_file)

    # Normalize the point cloud
    pcd = normalize_point_cloud(pcd)

    num_points = len(pcd.points)
    print('Object: {}, Num points: {}'.format(object_name, num_points))
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=15))
    pcd.orient_normals_consistent_tangent_plane(15)
    pcd_normals = np.asarray(pcd.normals).astype(np.float32)
    pcd_normals = np.reshape(pcd_normals, (num_points, 3))

    # Save normalized point cloud
    normalized_ply_file = os.path.join('ycb', object_name, 'clouds', 'normalized_merged_cloud.ply')
    o3d.io.write_point_cloud(normalized_ply_file, pcd)
    print('Normalized point cloud saved to {}'.format(normalized_ply_file))

    # Save normals
    npy_file = os.path.join('ycb', object_name, 'clouds', 'normals.npy')
    np.save(npy_file, pcd_normals)
    print('Normals saved to {}'.format(npy_file))
