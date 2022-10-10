import numpy as np
import open3d as o3d

def icp(
        Pc0, Pc1,
        max_iter, dist_threshold,
        kd_radius=0.02, kd_num=30,
        max_correspondence_dist=0.01,
        icp_method='color', init_Tc1c0=np.identity(4),
        with_info=False
):
    if icp_method == "point_to_point":
        res = o3d.pipelines.registration.registration_icp(
            Pc0, Pc1,
            dist_threshold, init_Tc1c0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )

    else:
        Pc0.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=kd_radius, max_nn=kd_num)
        )
        Pc1.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=kd_radius, max_nn=kd_num)
        )
        res = o3d.pipelines.registration.registration_icp(
            Pc0, Pc1,
            dist_threshold, init_Tc1c0,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )

    info = None
    if with_info:
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            Pc0, Pc1, max_correspondence_dist, res.transformation
        )
    return res, info

def compute_Tc1c0(
        Pc0, Pc1,
        voxelSizes, maxIters,
        icp_method='point_to_point',
        init_Tc1c0=np.identity(4),
):
    cur_Tc1c0 = init_Tc1c0
    run_times = len(maxIters)

    for idx in range(run_times):
        with_info = idx == run_times - 1

        max_iter = maxIters[idx]
        voxel_size = voxelSizes[idx]
        dist_threshold = voxel_size * 1.4

        Pc0_down = Pc0.voxel_down_sample(voxel_size)
        Pc1_down = Pc1.voxel_down_sample(voxel_size)

        res, info = icp(
            Pc0=Pc0_down, Pc1=Pc1_down,
            max_iter=max_iter, dist_threshold=dist_threshold,
            icp_method=icp_method,
            init_Tc1c0=cur_Tc1c0,
            kd_radius=voxel_size * 2.0, kd_num=30,
            max_correspondence_dist=voxel_size * 1.4,
            with_info=with_info
        )

        cur_Tc1c0 = res.transformation

    return cur_Tc1c0, info

pcd0: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
    '/home/quan/Desktop/tempary/redwood/00003/fragments/0_pcd.ply'
)
pcd0.colors = o3d.utility.Vector3dVector(
    np.tile(np.array([[1.0, 0.0, 0.0]]), [np.asarray(pcd0.points).shape[0], 1])
)

pcd1: o3d.geometry.PointCloud = o3d.io.read_point_cloud(
    '/home/quan/Desktop/tempary/redwood/00003/fragments/1_pcd.ply'
)
pcd1.colors = o3d.utility.Vector3dVector(
    np.tile(np.array([[0.0, 0.0, 1.0]]), [np.asarray(pcd1.points).shape[0], 1])
)

Tc1c0 = np.load('/home/quan/Desktop/tempary/redwood/00003/fragments/0_Tcw.npy')

Tc1c0, info = compute_Tc1c0(pcd0, pcd1, voxelSizes=[0.05, 0.01], maxIters=[100, 100], init_Tc1c0=Tc1c0)
pcd0 = pcd0.transform(Tc1c0)

o3d.visualization.draw_geometries([pcd0, pcd1])
