import open3d as o3d
import numpy as np
import reconstruct.utils.rmsd_kabsch as kabsch_rmsd
import matplotlib.pyplot as plt
from copy import deepcopy

def plane_Align_Axes(
        pcd: o3d.geometry.PointCloud,
        height=720, width=960,
        distance_threshold=1.0, ransac_n=100, num_iterations=30,
        debug=False, coodr_size=10, retur_pcd=False
):
    print('1) Press shift and left drag to select points')
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(height=height, width=width)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    pick_points = vis.get_picked_points()
    if len(pick_points)<=0:
        print('[DEBUG]: No Points Select')
        return

    origin_pcd = deepcopy(pcd)
    select_pick_idx = []
    for pick_point in pick_points:
        select_pick_idx.append(pick_point.index)
    pcd = origin_pcd.select_by_index(select_pick_idx)

    plane_fun, plane_index = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations
    )

    normal = np.array(plane_fun[:3])
    intercept = plane_fun[-1]
    normal = normal / np.sqrt(np.sum(np.power(normal, 2)))

    print("[DEBUG]: Find Plane %.3fxX+%.3fxY+%.3fxZ+%.3f"%(
        normal[0], normal[1], normal[2], intercept
    ))

    plane_pcd = pcd.select_by_index(plane_index)

    ### fit to right hand rule
    world_z = np.array([0, 0, 1])
    if np.sum(normal * world_z) < 0:
        normal = -normal

    point_a = np.array([0, 0, -intercept / normal[2]])
    point_b = np.array([1, 1, (-intercept - normal[0] - normal[1]) / normal[2]])
    x_axes = (point_a - point_b)
    x_axes = x_axes / np.sqrt(np.sum(np.power(x_axes, 2)))

    world_x = np.array([1, 0, 0])
    if np.sum(world_x * x_axes) < 0:
        x_axes = -x_axes

    z_axes = normal
    z_axes = z_axes.reshape((1, 3))
    x_axes = x_axes.reshape((1, 3))
    y_axes = np.cross(z_axes, x_axes)

    world_y = np.array([0, 1, 0])
    if np.sum(y_axes.reshape(-1) * world_y) < 0:
        y_axes = -y_axes

    if debug:
        plane_pcd_np = np.asarray(plane_pcd.points)
        # d = np.abs(plane_pcd_np.dot(normal.T) + intercept)/(np.sqrt(np.sum(np.power(normal, 2))))
        ### assume length of normal is one
        distance = np.abs(plane_pcd_np.dot(normal.T) + intercept)
        print('[DEBUG]: The Largest Distance noise: %.3f'% distance.max())

    plane_axes = np.concatenate((x_axes.T, y_axes.T, z_axes.T), axis=1)
    if debug:
        print('[DEBUG]: Cosin Between X&Y %.3f'%np.sum(plane_axes[:, 0] * plane_axes[:, 1]))
        print('[DEBUG]: Cosin Between X&Z %.3f'%np.sum(plane_axes[:, 0] * plane_axes[:, 2]))
        print('[DEBUG]: Cosin Between Y&Z %.3f'%np.sum(plane_axes[:, 1] * plane_axes[:, 2]))

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_xlim3d([-1.0, 1.0])
        ax.set_ylim3d([-1.0, 1.0])
        ax.set_zlim3d([-1.0, 1.0])
        ax.plot([0., plane_axes[0, 0]], [0., plane_axes[1, 0]], [0., plane_axes[2, 0]], c='r')
        ax.plot([0., plane_axes[0, 1]], [0., plane_axes[1, 1]], [0., plane_axes[2, 1]], c='g')
        ax.plot([0., plane_axes[0, 2]], [0., plane_axes[1, 2]], [0., plane_axes[2, 2]], c='b')
        plt.show()

    axes_point = plane_axes.T
    target_axes = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])
    target_center = np.mean(target_axes, axis=0)
    axes_center = np.mean(axes_point, axis=0)
    target_normal = target_axes - target_center
    axes_normal = axes_point - axes_center
    rot_mat = kabsch_rmsd.kabsch(P=target_normal, Q=axes_normal)
    # vec = target_center - (rot_mat.dot(axes_center.T)).T

    if debug:
        plane_pcd = plane_pcd.rotate(rot_mat, plane_pcd.get_center())
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=coodr_size, origin=pcd.get_center()
        )
        o3d.visualization.draw_geometries([plane_pcd, mesh_frame])

    if retur_pcd:
        origin_pcd = origin_pcd.rotate(rot_mat, origin_pcd.get_center())
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=coodr_size, origin=origin_pcd.get_center()
        )
        o3d.visualization.draw_geometries([origin_pcd, mesh_frame])
        return origin_pcd
    else:
        return rot_mat

### -------------------------------------------------------------
def create_map(height, width):
    x = np.tile(np.arange(0, width, 1).reshape((1, -1)), (height, 1))
    y = np.tile(np.arange(0, height, 1).reshape((-1, 1)), (1, width))
    map = np.concatenate((x[..., np.newaxis], y[..., np.newaxis]), axis=-1)
    map = map.reshape((-1, 2))
    return map

def render(vis):
    depth_buf = vis.capture_depth_float_buffer()
    depth_img = np.asarray(depth_buf)

    rgb_buf = vis.capture_screen_float_buffer()
    rgb_img = np.asarray(rgb_buf)

    ctr:o3d.visualization.ViewControl = vis.get_view_control()
    camera = ctr.convert_to_pinhole_camera_parameters()
    intrinsic:o3d.camera.PinholeCameraIntrinsic = camera.intrinsic
    intrinsic = intrinsic.intrinsic_matrix
    extrinsic = camera.extrinsic

    # print('[DEBUG]: intrinsic: \n')
    # print(intrinsic)
    # print('[DEBUG]: extrinsic: \n')
    # print(extrinsic)

    # ### -----------------------------------------------------------
    # ### ------ debug
    # map = create_map(height=rgb_img.shape[0], width=rgb_img.shape[1])
    # depth_np = depth_img.reshape(-1)
    # select_bool = depth_np>10
    # uv = map[select_bool]
    # depth_np = depth_np[select_bool]
    # depth_np = depth_np.reshape((-1, 1))
    #
    # uvs = np.concatenate([uv * depth_np, depth_np], axis=1)
    # Kv = np.linalg.inv(intrinsic)
    # pcd_camera = (Kv.dot(uvs.T)).T
    # pcd_camera_t = np.concatenate([pcd_camera, np.ones((pcd_camera.shape[0], 1))], axis=1)
    # 
    # extrinsic = np.linalg.inv(extrinsic)
    # pcd_world_t = (extrinsic.dot(pcd_camera_t.T)).T
    #
    # pcd_world = pcd_world_t[:, :3]
    # idx = np.arange(0, pcd_world.shape[0], 1)
    # select_idx = np.random.choice(idx, size=10000)
    # pcd_world = pcd_world[select_idx]
    #
    # pcd_orig = np.asarray(pcd.points)
    # # pcd_orig = np.asarray(mesh.vertices)
    # idx = np.arange(0, pcd_orig.shape[0], 1)
    # select_idx = np.random.choice(idx, size=10000)
    # pcd_orig = pcd_orig[select_idx]
    #
    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # ax.set_xlim3d([-120, 120])
    # ax.set_ylim3d([-120, 120])
    # ax.set_zlim3d([-120, 120])
    # ax.scatter(pcd_world[:, 0], pcd_world[:, 1], pcd_world[:, 2], s=0.01, c='r')
    # ax.scatter(pcd_orig[:, 0], pcd_orig[:, 1], pcd_orig[:, 2], s=0.01, c='b')
    # plt.show()
    # ### ------------------------------------------------------

    # plt.figure('depth')
    # plt.imshow(depth_img)
    # plt.figure('rgb')
    # plt.imshow(rgb_img)
    # plt.show()

def apriltag_Align_Axes(
        pcd: o3d.geometry.PointCloud,
        height=720, width=960,
        to_mesh=False, voxels_size=-1.0
):
    s = 'Help&Info \n'
    s += '1) R: Reset view point. \n'
    s += '2) Q/q: Exit window. \n'
    s += '3) H: Print help message. \n'
    s += '4) 1: Render point color. \n'
    s += '5) 9: normal as color.\n'
    s += '6) ,: apriltag process \n'

    if voxels_size>0:
        pcd = pcd.voxel_down_sample(voxels_size)

    if to_mesh:
        pcd, double_v = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd)
        # o3d.visualization.draw_geometries([pcd])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(height=height, width=width)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    vis.add_geometry(pcd)
    vis.register_key_callback(ord(','), render)

    vis.run()

    vis.destroy_window()

if __name__ == '__main__':
    # pcd = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/model1/cropped_3.ply')
    # pcd = plane_Align_Axes(pcd=pcd, debug=False, coodr_size=30, retur_pcd=True)
    # o3d.io.write_point_cloud('/home/psdz/HDD/quan/3d_model/test.ply', pcd)

    pcd = o3d.io.read_point_cloud('/home/psdz/HDD/quan/3d_model/model1/cropped_1.ply')
    pcd = pcd.voxel_down_sample(2.0)
    apriltag_Align_Axes(pcd, to_mesh=False)

    # mesh = o3d.io.read_triangle_mesh('/home/psdz/HDD/quan/3d_model/1.ply')
    # apriltag_Align_Axes(mesh, to_mesh=False)
