import numpy as np
from DGR.config import get_config
from DGR.core.deep_global_registration import DeepGlobalRegistration
import open3d as o3d
from teaser_python_fpfh_icp.helpers import *
import copy


if __name__ == '__main__':
    config = get_config()

    dgr = DeepGlobalRegistration(config)

    # VOXEL_SIZE = 0.05
    VOXEL_SIZE = 0.001
    VISUALIZE = True

    dataset='hactl'
    # pair = (100000,100000)
    pair = (100256,100285)

    dataset='taipocity'
    pair = (100974,100988)

    path1 = '/media/qzj/Document/grow/research/slamDataSet/ICCV/{}_day/pointcloud_20m_10overlap/{}.bin'.format(dataset,
        pair[0])
    A_pcd_raw = o3d.geometry.PointCloud()
    A_pcd_raw.points = o3d.utility.Vector3dVector(np.fromfile(path1, dtype=np.float32, count=-1).reshape([-1, 3]))

    path2 = '/media/qzj/Document/grow/research/slamDataSet/ICCV/{}_night/pointcloud_20m_10overlap/{}.bin'.format(dataset,
        pair[1])
    B_pcd_raw = o3d.geometry.PointCloud()
    B_pcd_raw.points = o3d.utility.Vector3dVector(np.fromfile(path2, dtype=np.float32, count=-1).reshape([-1, 3]))

    A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0])  # show A_pcd in blue
    B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0])  # show B_pcd in red
    # if VISUALIZE:
    #     o3d.visualization.draw_geometries([A_pcd_raw, B_pcd_raw],window_name='raw pcl')  # plot A and B

    # voxel downsample both clouds
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd, B_pcd],window_name='after voxel')  # plot downsampled A and B

    A_xyz = pcd2xyz(A_pcd)  # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd)  # np array of size 3 by M

    T_pred = dgr.register(A_xyz.T, B_xyz.T)

    # Visualize the registration results
    A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_pred)
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd_T_teaser, B_pcd],window_name='DGR result')

    NOISE_BOUND = VOXEL_SIZE
    # T_teaser = np.eye(4)
    # local refinement using ICP
    icp_sol = o3d.registration.registration_icp(
        A_pcd, B_pcd, NOISE_BOUND, T_pred,
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=100))
    T_icp = icp_sol.transformation

    # visualize the registration after ICP refinement
    A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd_T_icp, B_pcd],window_name='ICP result')