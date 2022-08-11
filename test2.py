import numpy as np
import open3d as o3d
import cv2

fix: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/model1/cropped_1.ply')
crop: o3d.geometry.PointCloud = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/model1/2_fuse.ply')

# orient_box = crop.get_oriented_bounding_box()
# orient_box.color = (1, 0, 0)
# axis_box = crop.get_axis_aligned_bounding_box()
# axis_box.color = (0, 1, 0)
#
# o3d.visualization.draw_geometries([
#     crop,
#     orient_box,
#     axis_box
# ])

fix = fix.voxel_down_sample(2.0)
crop = crop.voxel_down_sample(2.0)

fix = fix.translate(-fix.get_center())
crop = crop.translate(-crop.get_center())

fix.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
crop.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
fix_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    fix, o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=100)
)
crop_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    crop, o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=100)
)

result_icp = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    crop, fix, crop_fpfh, fix_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=30.0
    )
)
crop = crop.transform(result_icp.transformation)

# o3d.visualization.draw_geometries([fix, crop])

orient_box = crop.get_oriented_bounding_box()
inlier_indx = orient_box.get_point_indices_within_bounding_box(fix.points)

### crop outside
# fix = fix.crop(orient_box)

### crop inside
fix = fix.select_by_index(inlier_indx, invert=True)

o3d.visualization.draw_geometries([fix])
