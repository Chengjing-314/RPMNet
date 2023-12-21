import h5py
import numpy as np
import os
import open3d as o3d
current_dir = os.path.dirname(os.path.realpath(__file__))
orientation_issues_count = 0
total_samples_count = 0
for root, dirs, files in os.walk(current_dir):
    for file in files:
        if file.endswith('.h5'):
            file_path = os.path.join(root, file)
            print(file_path)
            with h5py.File(file_path, 'r+') as f:
                points = np.asarray(f['data'][:]).astype(np.float32)
                normals = np.empty_like(points)
                data_normals = np.asarray(f['normal'][:]).astype(np.float32)  # Load data normals
                for i in range(points.shape[0]):
                    total_samples_count += 1  # Increment the total samples counter
                    try:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points[i])
                        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
                        pcd.orient_normals_consistent_tangent_plane(30)
                        normals[i] = np.asarray(pcd.normals).astype(np.float32)
                    except RuntimeError:
                        # Fallback to data normal and increment the issue count
                        normals[i] = data_normals[i]
                        orientation_issues_count += 1
                if 'estimate_normals' in f:
                    del f['estimate_normals']
                f.create_dataset('estimate_normals', data=normals)
# Calculate the percentage of samples with orientation issues
issue_percentage = (orientation_issues_count / total_samples_count) * 100
print(f"Percentage of samples with orientation issues: {issue_percentage:.2f}%")

