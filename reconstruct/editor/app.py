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
from reconstruct.utils.utils import eulerAngles_to_rotationMat_scipy
import reconstruct.utils.rmsd_kabsch as kabsch_rmsd

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
    geometry_combo_name = ""
    x_click, y_click, z_click = 0.0, 0.0, 0.0

    # axes_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0, 0, 0])

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

        ### panel
        em = self.window.theme.font_size
        self.spacing = int(np.round(0.1 * em))
        vspacing = int(np.round(0.1 * em))
        self.margins = gui.Margins(vspacing)
        self.panel = gui.Vert(self.spacing, self.margins)

        ### show axel
        show_axel_switch = gui.ToggleSwitch("Show axes")
        show_axel_switch.set_on_clicked(self.show_axel_switch)
        self.panel.add_child(show_axel_switch)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0, 0, 0])
        self.widget3d.scene.add_model(
            'axes', axes
        )

        self.xyz_info = gui.Label("x.y.z")
        self.panel.add_child(self.xyz_info)
        self.widget3d.set_on_mouse(self._on_mouse_widget3d)

        ### geometry list
        self.geometry_layout = gui.CollapsableVert("geometry", self.spacing, self.margins)
        self.panel.add_child(self.geometry_layout)

        ### analysis layout
        self.analysis_layout = gui.CollapsableVert("analysis", self.spacing, self.margins)
        self.panel.add_child(self.analysis_layout)

        ### delect geometry function
        self.geometry_combo = gui.Combobox()
        self.geometry_combo.add_item("empty init ......")
        self.geometry_combo.set_on_selection_changed(self.geometry_combo_select_change)
        geometry_delete_btn = gui.Button("DELETE")
        geometry_delete_btn.set_on_clicked(self.geometry_delete_on_click)
        self.analysis_layout.add_child(self.geometry_combo)
        self.analysis_layout.add_child(geometry_delete_btn)

        ### info layout
        info_layout = gui.CollapsableVert("info", self.spacing, self.margins)
        self.info_label = gui.Label("Help&Info: \n Version 0.1")
        info_layout.add_child(self.info_label)
        self.panel.add_child(info_layout)

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

            ### update gui
            checkbox = gui.Checkbox(base_name)
            checkbox.checked = True
            checkbox.set_on_checked(partial(self.geometry_on_click, name=base_name))
            self.geometry_layout.add_child(checkbox)

            self.geometry_combo.add_item(base_name)

            ### adjust camera viewpoint
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

    def geometry_combo_select_change(self, val, idx):
        self.geometry_combo_name = val

    def geometry_delete_on_click(self):
        name = self.geometry_combo_name
        if name in self.geometry_map.keys():
            visible = self.geometry_map[name]['visible']
            if visible:
                self.widget3d.scene.remove_geometry(name)

            del self.geometry_map[name]
            self.geometry_combo.remove_item(name)

    def show_axel_switch(self, is_on):
        # self.widget3d.scene.show_axes(is_on)
        # if is_on:
        pass

    def _on_mouse_widget3d(self, event:gui.MouseEvent):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):
            def depth_callback(depth_image):
                x = event.x - self.widget3d.frame.x
                y = event.y - self.widget3d.frame.y
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = "out of bounding"
                else:
                    xw, yw, zw = self.widget3d.scene.camera.unproject(
                        event.x, event.y, depth, self.widget3d.frame.width, self.widget3d.frame.height
                    )

                    self.x_click = xw
                    self.y_click = yw
                    self.z_click = zw

                    text = "({:.3f}, {:.3f}, {:.3f})".format(xw, yw, zw)

                self.xyz_info.text = text

            self.widget3d.scene.scene.render_to_depth_image(depth_callback)

            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

class CustomWindow(AppWindow):
    select_points = []

    def __init__(self, args):
        super(CustomWindow, self).__init__(args=args)

        # geometry_crop_btn = gui.Button("Crop")
        # geometry_crop_btn.set_on_clicked(self.geometry_crop_on_click)
        # self.analysis_layout.add_child(geometry_crop_btn)

        ### translate layout
        translate_layout = gui.Vert(self.spacing, self.margins)

        hlayout = gui.Horiz(self.spacing, self.margins)
        hlayout.add_child(gui.Label("x:"))
        self.x_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        hlayout.add_child(self.x_translate)
        translate_layout.add_child(hlayout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        hlayout.add_child(gui.Label("y:"))
        self.y_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        hlayout.add_child(self.y_translate)
        translate_layout.add_child(hlayout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        hlayout.add_child(gui.Label("z:"))
        self.z_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        hlayout.add_child(self.z_translate)
        translate_layout.add_child(hlayout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        hlayout.add_child(gui.Label("x_angel:"))
        self.x_angel_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        hlayout.add_child(self.x_angel_translate)
        translate_layout.add_child(hlayout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        hlayout.add_child(gui.Label("y_angel:"))
        self.y_angel_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        hlayout.add_child(self.y_angel_translate)
        translate_layout.add_child(hlayout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        hlayout.add_child(gui.Label("z_angel:"))
        self.z_angel_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        hlayout.add_child(self.z_angel_translate)
        translate_layout.add_child(hlayout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        hlayout.add_child(gui.Label("scale:"))
        self.geometry_scale = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        hlayout.add_child(self.geometry_scale)
        translate_layout.add_child(hlayout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        hlayout.add_child(gui.Label("voxel:"))
        self.geometry_voxel = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        hlayout.add_child(self.geometry_voxel)
        translate_layout.add_child(hlayout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        translate_btn = gui.Button("Translate")
        translate_btn.set_on_clicked(self.translate_on_click)
        hlayout.add_child(translate_btn)
        scale_btn = gui.Button("Scale")
        scale_btn.set_on_clicked(self.scale_on_click)
        hlayout.add_child(scale_btn)
        voxel_btn = gui.Button("Voxel")
        voxel_btn.set_on_clicked(self.voxel_on_click)
        hlayout.add_child(voxel_btn)

        translate_layout.add_child(hlayout)
        self.analysis_layout.add_child(translate_layout)

        ### orient align with direction
        self.widget3d.set_on_key(self._on_key_widget3d)

        hlayout = gui.Horiz(self.spacing, self.margins)
        align_btn = gui.Button("align")
        align_btn.set_on_clicked(self.align_to_oriention)
        hlayout.add_child(align_btn)
        move_origin_btn = gui.Button("toOrign")
        move_origin_btn.set_on_clicked(self.move_to_origin)
        hlayout.add_child(move_origin_btn)
        self.analysis_layout.add_child(hlayout)

    def geometry_crop_on_click(self):
        s = "Demo for manual geometry cropping \n"
        s += "1) Press 'K' to lock screen and to switch to selection mode \n"
        s += "2) Drag for rectangle selection, or use ctrl + left click for polygon selection \n"
        s += "3) Press 'C' to get a selected geometry \n"
        s += "4) Press 'S' to save the selected geometry \n"
        s += "5) Press 'K' again to switch to freeview mode"
        # print(s)
        self.info_label.text = s

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(width=960, height=720)
        vis.run()
        vis.destroy_window()

        # name = self.geometry_combo_name
        # if name in self.geometry_map.keys():
        #     visible = self.geometry_map[name]['visible']
        #     if visible:
        #         self.widget3d.scene.remove_geometry(name)
        #
        #     # vis = o3d.visualization.VisualizerWithEditing()
        #     # vis.create_window()
        #     # vis.add_geometry(self.geometry_map[name]['geometry'])
        #     # vis.run()
        #     # vis.destroy_window()
        #     #
        #     # del self.geometry_map[name]['geometry']
        #     # self.geometry_map[name]['geometry'] = vis.get_cropped_geometry()
        #
        #     # if visible:
        #     #     is_pcd = self.geometry_map[name]['is_pcd']
        #     #     if is_pcd:
        #     #         self.widget3d.scene.add_geometry(name, self.geometry_map[name]['geometry'], self.materials[AppWindow.LIT])
        #     #     else:
        #     #         self.widget3d.scene.add_model(name, self.geometry_map[name]['geometry'])
        # else:
        #     self.info_label.text = "No Geometry Find: Please select correct geometry combo"

    def translate_on_click(self):
        name = self.geometry_combo_name
        if name in self.geometry_map.keys():
            x = self.x_translate.double_value
            y = self.y_translate.double_value
            z = self.z_translate.double_value
            x_angel = self.x_angel_translate.double_value
            y_angel = self.y_angel_translate.double_value
            z_angel = self.z_angel_translate.double_value

            visible = self.geometry_map[name]['visible']
            if visible:
                self.widget3d.scene.remove_geometry(name)

            rot = eulerAngles_to_rotationMat_scipy([x_angel, y_angel, z_angel], degress=True)
            vec = np.array([x, y, z])

            geometry = self.geometry_map[name]['geometry']
            geometry = geometry.rotate(rot, geometry.get_center())
            geometry = geometry.translate(vec)

            is_pcd = self.geometry_map[name]['is_pcd']
            if is_pcd:
                self.widget3d.scene.add_geometry(name, geometry, self.materials[AppWindow.LIT])
            else:
                self.widget3d.scene.add_model(name, geometry)

        else:
            self.info_label.text = "No Geometry Find: Please select correct geometry combo"

    def scale_on_click(self):
        name = self.geometry_combo_name
        if name in self.geometry_map.keys():
            scale = self.geometry_scale.double_value

            if scale<0.001:
                self.info_label.text = "scale is too small"
            else:
                visible = self.geometry_map[name]['visible']
                if visible:
                    self.widget3d.scene.remove_geometry(name)

                geometry = self.geometry_map[name]['geometry']
                geometry = geometry.scale(scale, geometry.get_center())

                is_pcd = self.geometry_map[name]['is_pcd']
                if is_pcd:
                    self.widget3d.scene.add_geometry(name, geometry, self.materials[AppWindow.LIT])
                else:
                    self.widget3d.scene.add_model(name, geometry)

        else:
            self.info_label.text = "No Geometry Find: Please select correct geometry combo"

    def voxel_on_click(self):
        name = self.geometry_combo_name
        if name in self.geometry_map.keys():
            voxel = self.geometry_voxel.double_value

            visible = self.geometry_map[name]['visible']
            if visible:
                self.widget3d.scene.remove_geometry(name)

            print(self.geometry_map[name]['geometry'])

            geometry = self.geometry_map[name]['geometry']
            self.geometry_map[name]['geometry'] = geometry.voxel_down_sample(voxel)

            print(self.geometry_map[name]['geometry'])

            is_pcd = self.geometry_map[name]['is_pcd']
            if is_pcd:
                self.widget3d.scene.add_geometry(name, self.geometry_map[name]['geometry'], self.materials[AppWindow.LIT])
            else:
                self.widget3d.scene.add_model(name, self.geometry_map[name]['geometry'])

        else:
            self.info_label.text = "No Geometry Find: Please select correct geometry combo"

    def _on_key_widget3d(self, event:gui.KeyEvent):
        if event.type == gui.KeyEvent.DOWN:
            if event.key == ord('s'):
                self.select_points.append(np.array([self.x_click, self.y_click, self.z_click]))
                self.info_label.text = "[Debug]: Add point %.3f %.3f %.3f"%(self.x_click, self.y_click, self.z_click)

            elif event.key == ord('d'):
                if len(self.select_points)>0:
                    point = self.select_points[-1]
                    self.info_label.text = "[Debug]: Delete point %.3f %.3f %.3f" % (point[0], point[1], point[2])
                    self.select_points = self.select_points[:-1]

            elif event.key == ord('c'):
                self.select_points.clear()
                self.info_label.text = "[DEBUG]: Clear Points"

            elif event.key == ord('1'):
                s = '[DEBUG]: Points Num:%d \n'%len(self.select_points)
                for idx, point in enumerate(self.select_points):
                    s += "[Debug]: point %d: %.3f %.3f %.3f \n"%(idx, point[0], point[1], point[2])
                self.info_label.text = s

            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def align_to_oriention(self):
        name = self.geometry_combo_name
        if name in self.geometry_map.keys():

            if len(self.select_points)>3:
                target_axes = np.array([
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]
                ])

                plane_points = np.array(self.select_points)
                u, sigma, v = np.linalg.svd(plane_points)
                points_axes = v.T

                target_axes_normal = target_axes - kabsch_rmsd.centroid(target_axes)
                points_axes_normal = points_axes - kabsch_rmsd.centroid(points_axes)

                rot_mat = kabsch_rmsd.kabsch(P=points_axes_normal, Q=target_axes_normal)

                visible = self.geometry_map[name]['visible']
                if visible:
                    self.widget3d.scene.remove_geometry(name)

                geometry = self.geometry_map[name]['geometry']
                geometry = geometry.rotate(rot_mat, geometry.get_center())

                is_pcd = self.geometry_map[name]['is_pcd']
                if is_pcd:
                    self.widget3d.scene.add_geometry(name, geometry, self.materials[AppWindow.LIT])
                else:
                    self.widget3d.scene.add_model(name, geometry)

            else:
                self.info_label.text = "Select Points Are Not Enough"
        else:
            self.info_label.text = "No Geometry Find: Please select correct geometry combo"

    def move_to_origin(self):
        name = self.geometry_combo_name
        if name in self.geometry_map.keys():

            visible = self.geometry_map[name]['visible']
            if visible:
                self.widget3d.scene.remove_geometry(name)

            geometry = self.geometry_map[name]['geometry']
            vec = geometry.get_center()
            vec = -vec
            geometry = geometry.translate(vec)

            is_pcd = self.geometry_map[name]['is_pcd']
            if is_pcd:
                self.widget3d.scene.add_geometry(name, geometry, self.materials[AppWindow.LIT])
            else:
                self.widget3d.scene.add_model(name, geometry)

        else:
            self.info_label.text = "No Geometry Find: Please select correct geometry combo"

def main():
    args = parse_args()

    app = gui.Application.instance
    app.initialize()

    window = CustomWindow(args)

    app.run()

if __name__ == '__main__':
    main()
