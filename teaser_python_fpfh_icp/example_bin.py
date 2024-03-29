import open3d as o3d
import teaserpp_python
import numpy as np
import os
import copy
import struct
from time import perf_counter
from teaser_python_fpfh_icp.helpers import *

VOXEL_SIZE = 0.05
VISUALIZE = True

A_pcd_raw = o3d.geometry.PointCloud()
path1 = '/media/qzj/Document/grow/research/slamDataSet/kitti/data_odometry_velodyne/dataset/sequences/00/velodyne/000000.bin'
pcl_np = np.fromfile(path1, dtype=np.float32, count=-1).reshape([-1, 4])[:10000, :3]
A_pcd_raw.points = o3d.utility.Vector3dVector(pcl_np)

B_pcd_raw = o3d.geometry.PointCloud()
path1 = '/media/qzj/Document/grow/research/slamDataSet/kitti/data_odometry_velodyne/dataset/sequences/00/velodyne/000000.bin'
pcl_np = np.fromfile(path1, dtype=np.float32, count=-1).reshape([-1, 4])[:10000, :3]
B_pcd_raw.points = o3d.utility.Vector3dVector(pcl_np)

A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red
# if VISUALIZE:
#     o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw]) # plot A and B

# voxel downsample both clouds
A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
print("A_pcd size: {}".format(np.asarray(A_pcd.points).shape[0]))
print("B_pcd size: {}".format(np.asarray(B_pcd.points).shape[0]))

T = np.array(
    [[9.96926560e-01, 6.68735757e-02, -4.06664421e-02, -1.15576939e-01],
     [-6.61289946e-02, 9.97617877e-01, 1.94008687e-02, -3.87705398e-02],
     [4.18675510e-02, -1.66517807e-02, 9.98977765e-01, 1.14874890e-01],
     [0, 0, 0, 1]])
B_pcd.transform(T)


if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd,B_pcd]) # plot downsampled A and B 

A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M

t1 = perf_counter()
# extract FPFH features
A_feats = extract_fpfh(A_pcd,VOXEL_SIZE)
B_feats = extract_fpfh(B_pcd,VOXEL_SIZE)

# establish correspondences by nearest neighbour search in feature space
corrs_A, corrs_B = find_correspondences(
    A_feats, B_feats, mutual_filter=True)
A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs

num_corrs = A_corr.shape[1]

t2=perf_counter()
print(f'FPFH generates {num_corrs} putative correspondences.')
print(f'FPFH costs {t2-t1} s.')

# visualize the point clouds together with feature correspondences
points = np.concatenate((A_corr.T,B_corr.T),axis=0)
lines = []
for i in range(num_corrs):
    lines.append([i,i+num_corrs])
colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd,B_pcd,line_set])

t1 = perf_counter()
# robust global registration using TEASER++
NOISE_BOUND = VOXEL_SIZE
teaser_solver = get_teaser_solver(NOISE_BOUND)
teaser_solver.solve(A_corr,B_corr)
solution = teaser_solver.getSolution()
R_teaser = solution.rotation
t_teaser = solution.translation
T_teaser = Rt2T(R_teaser,t_teaser)
t2 = perf_counter()
print(f'solve costs {t2-t1} s.')

# Visualize the registration results
A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_teaser)
if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd])

T_teaser = np.eye(4)
# local refinement using ICP
icp_sol = o3d.pipelines.registration.registration_icp(
      A_pcd, B_pcd, NOISE_BOUND, T_teaser,
      o3d.registration.TransformationEstimationPointToPoint(),
      o3d.registration.ICPConvergenceCriteria(max_iteration=100))
T_icp = icp_sol.transformation

# visualize the registration after ICP refinement
A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)
if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd])




