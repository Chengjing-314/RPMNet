import open3d as o3d 
import os 

current_dir = os.path.dirname(os.path.realpath(__file__))

f = open(current_dir + '/ycb/objects.txt', 'r')
object_list = [line.strip() for line in f.readlines()]
f.close()

for object_name in object_list:
    ply_file = os.path.join('ycb', object_name, 'clouds', 'merged_cloud.ply')
    pcd = o3d.io.read_point_cloud(ply_file)
    num_points = len(pcd.points)
    print('Object: {}, Num points: {}'.format(object_name, num_points))