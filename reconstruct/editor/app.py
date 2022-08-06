import os
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading
import time
import argparse
import numpy as np
from functools import partial
import cv2
import matplotlib.pyplot as plt

from  reconstruct.editor.custom_gui import InfoCheckbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, help="",default=960)
    parser.add_argument("--height", type=int, help="", default=720)
    args = parser.parse_args()
    return args

class AppWindow(object):
    MENU_OPEN = 1
    MENU_QUIT = 2
    MENU_ABOUT = 11

    ### material
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    ### geometry manager
    geometry_map = {}

    ### raycasting
    rgb_snapshot_render = False
    depth_snapshot_render = False
    rgb_snapshot = None
    depth_snapshot = None

    def __init__(self, args):
        self.args = args

        self.window = gui.Application.instance.create_window("Reconstruct", args.width, args.height)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)

            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        self.window.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT, gui.Application.instance.quit)
        self.window.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)

        ### material
        self.materials = {
            AppWindow.LIT: rendering.MaterialRecord(),
            AppWindow.UNLIT: rendering.MaterialRecord(),
            AppWindow.NORMALS: rendering.MaterialRecord(),
            AppWindow.DEPTH: rendering.MaterialRecord()
        }
        self.materials[AppWindow.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self.materials[AppWindow.LIT].shader = AppWindow.LIT
        self.materials[AppWindow.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self.materials[AppWindow.UNLIT].shader = AppWindow.UNLIT
        self.materials[AppWindow.NORMALS].shader = AppWindow.NORMALS
        self.materials[AppWindow.DEPTH].shader = AppWindow.DEPTH

        ### pannel
        em = self.window.theme.font_size
        self.spacing = int(np.round(0.5 * em))
        vspacing = int(np.round(0.5 * em))
        self.margins = gui.Margins(vspacing)
        self.panel = gui.Vert(self.spacing, self.margins)

        self.geometry_layout = gui.CollapsableVert("geometry", self.spacing, self.margins)
        self.panel.add_child(self.geometry_layout)

        ### add layout
        self.window.add_child(self.panel)
        self.window.add_child(self.widget3d)

        self.window.set_on_layout(self._on_layout)

    def _on_layout(self, layout_context):
        em = layout_context.theme.font_size

        panel_width = 15 * em
        rect = self.window.content_rect

        self.widget3d.frame = gui.Rect(rect.x, rect.y, rect.get_right() - panel_width, rect.height)
        self.panel.frame = gui.Rect(self.widget3d.frame.get_right(), rect.y, panel_width, rect.height)

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load", self.window.theme)
        # dlg.add_filter(
        #     ".ply .stl .fbx .obj .off .gltf .glb", "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, .gltf, .glb)")
        # dlg.add_filter(
        #     ".xyz .xyzn .xyzrgb .ply .pcd .pts", "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, .pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        # dlg.add_filter(".stl", "Stereolithography files (.stl)")
        # dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        # dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        # dlg.add_filter(".off", "Object file format (.off)")
        # dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        # dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        # dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        # dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        # dlg.add_filter(".xyzrgb", "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        # dlg.add_filter(".pts", "3D Points files (.pts)")
        # dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def load(self, path):
        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_model(path)

        if mesh is None:
            print("[Info]", path, "appears to be a point cloud")
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None or mesh is not None:
            file_name = os.path.basename(path)
            base_name = file_name.split('.')[0]
            if mesh is not None:
                # Triangle model
                self.widget3d.scene.add_model(base_name, mesh)
                self.geometry_map[base_name] = {
                    'geometry': geometry, 'visible': True,
                    'is_pcd': False}
            else:
                # Point cloud
                self.widget3d.scene.add_geometry(base_name, geometry, self.materials[AppWindow.LIT])
                self.geometry_map[base_name] = {'geometry': geometry, 'visible':True, 'is_pcd':True}

            checkbox = gui.Checkbox(base_name)
            checkbox.checked = True
            checkbox.set_on_checked(partial(self.geometry_on_click, name=base_name))
            self.geometry_layout.add_child(checkbox)

            bounds = self.widget3d.scene.bounding_box
            self.widget3d.setup_camera(60, bounds, bounds.get_center())

    def _on_menu_about(self):
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Version 0.1"))

        ok = gui.Button("Cancel")
        ok.set_on_clicked(self.window.close_dialog)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def geometry_on_click(self, is_checked, name):
        visible = self.geometry_map[name]['visible']
        if visible != is_checked:
            is_pcd = self.geometry_map[name]['is_pcd']
            geometry = self.geometry_map[name]['geometry']
            if is_checked:
                if is_pcd:
                    self.widget3d.scene.add_geometry(name, geometry, self.materials[AppWindow.LIT])
                else:
                    self.widget3d.scene.add_model(name, geometry)
            else:
                self.widget3d.scene.remove_geometry(name)
            self.geometry_map[name]['visible'] = is_checked

    def raycasting_function_init(self):
        raycasting_layout = gui.CollapsableVert("Raycasting", self.spacing, self.margins)
        self.panel.add_child(raycasting_layout)

        raycasting_btn = gui.Button("Render")
        raycasting_btn.set_on_clicked(self.raycasting_render_fun)
        raycasting_result_btn = gui.Button("Shower")
        raycasting_result_btn.set_on_clicked(self.raycasting_result_show)

        hlayout = gui.Horiz()
        hlayout.add_child(raycasting_btn)
        hlayout.add_child(raycasting_result_btn)
        raycasting_layout.add_child(hlayout)

    def raycasting_result_show(self):
        plt.figure('rgb')
        plt.imshow(self.rgb_snapshot)
        plt.figure('depth')
        plt.imshow(self.depth_snapshot)
        plt.show()

    def raycasting_render_fun(self):
        print("[DEBUG]: Waiting For Depth Render")
        self.depth_snapshot_render = False
        self.widget3d.scene.scene.render_to_depth_image(self.depth_callback)

        print("[DEBUG]: Waiting For Rgb Render")
        self.rgb_snapshot_render = False
        self.widget3d.scene.scene.render_to_image(self.rgb_callback)

    def rgb_callback(self, rgb_img_buf):
        self.rgb_snapshot = np.asarray(rgb_img_buf)
        self.rgb_snapshot_render = True
        print('[Debug]: Finsih RGB Render')

    def depth_callback(self, depth_img_buf):
        self.depth_snapshot = np.asarray(depth_img_buf)
        self.depth_snapshot_render = True
        print('[Debug]: Finsih Depth Render')

class CustomWindow(AppWindow):

    def __init__(self, args):
        super(CustomWindow, self).__init__(args=args)

        analysis_btn = gui.Button("Analysis")
        self.panel.add_child(analysis_btn)
        analysis_btn.set_on_clicked(self.analysis_function)

        align_btn = gui.Button("Align")
        self.panel.add_child(align_btn)
        align_btn.set_on_clicked(self.align)

    def analysis_function(self):
        def capture_depth(vis):
            depth = vis.capture_depth_float_buffer()
            plt.imshow(np.asarray(depth))
            plt.show()
            return False

        def capture_rgb(vis):
            image = vis.capture_screen_float_buffer()
            plt.imshow(np.asarray(image))
            plt.show()
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

        for name in self.geometry_map.keys():
            if self.geometry_map[name]['visible']:
                vis.add_geometry(self.geometry_map[name]['geometry'])

        vis.register_key_callback(key=ord('1'), callback_func=capture_depth)
        vis.register_key_callback(key=ord('2'), callback_func=capture_rgb)

        vis.run()
        vis.destroy_window()

    def align(self):
        # print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
        # print("2) Press 'K' to lock screen and to switch to selection mode")
        # print("3) Drag for rectangle selection or use ctrl + left click for polygon selection")
        # print("4) use shift + left click for click point selection")
        # print("5) Press 'C' to get a selected geometry and to save it")
        # print("6) Press 'F' to switch to freeview mode")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

        for name in self.geometry_map.keys():
            if self.geometry_map[name]['visible']:
                vis.add_geometry(self.geometry_map[name]['geometry'])

        vis.run()
        vis.destroy_window()

        select_index = vis.get_picked_points()
        print(select_index)

def main():
    args = parse_args()

    app = gui.Application.instance
    app.initialize()

    window = CustomWindow(args)

    app.run()

if __name__ == '__main__':
    main()
