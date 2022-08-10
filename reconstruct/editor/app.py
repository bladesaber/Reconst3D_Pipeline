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
from copy import deepcopy, copy

from  reconstruct.editor.custom_gui import InfoCheckbox
from reconstruct.utils.utils import eulerAngles_to_rotationMat_scipy
import reconstruct.utils.rmsd_kabsch as kabsch_rmsd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, help="",default=1280)
    parser.add_argument("--height", type=int, help="", default=960)
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

        # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0, 0, 0])
        # self.widget3d.scene.add_model(
        #     'axes', axes
        # )

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
        hlayout = gui.Horiz(self.spacing, self.margins)
        hlayout.add_child(gui.Label("Select: "))
        self.geometry_combo = gui.Combobox()
        self.geometry_combo.add_item("Hello World ...")
        self.geometry_combo.set_on_selection_changed(self.geometry_combo_select_change)
        hlayout.add_child(self.geometry_combo)
        self.analysis_layout.add_child(hlayout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        geometry_delete_btn = gui.Button("Delete")
        geometry_delete_btn.set_on_clicked(self.geometry_delete_on_click)
        hlayout.add_child(geometry_delete_btn)
        geometry_output_btn = gui.Button("Output")
        geometry_output_btn.set_on_clicked(self.geometry_output_on_click)
        hlayout.add_child(geometry_output_btn)
        # geometry_mesh_btn = gui.Button("Mesh")
        # geometry_mesh_btn.set_on_clicked(self.geometry_mesh_on_click)
        # hlayout.add_child(geometry_mesh_btn)
        self.analysis_layout.add_child(hlayout)

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

        panel_width = 20 * em
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
                self.geometry_map[base_name] = {'geometry': geometry, 'visible': True, 'is_pcd': False}
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
            self.load_process(base_name)

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
        if name in self.geometry_map.keys():
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
            self.delete_process(name)
        else:
            self.info_label.text = "No Geometry Find: Please select correct geometry combo"

    def geometry_output_on_click(self):
        name = self.geometry_combo_name
        if name in self.geometry_map.keys():
            dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose path to output", self.window.theme)
            dlg.add_filter(".ply", "Polygon files (.ply)")
            dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
            # dlg.add_filter(".stl", "Stereolithography files (.stl)")
            dlg.set_on_cancel(self.window.close_dialog)
            dlg.set_on_done(self._on_export_dialog_done)
            self.window.show_dialog(dlg)
        else:
            self.info_label.text = "No Geometry Find: Please select correct geometry combo"

    def geometry_mesh_on_click(self):
        name = self.geometry_combo_name
        if name in self.geometry_map.keys():
            if self.geometry_map[name]['is_pcd']:
                geometry = self.geometry_map[name]['geometry']
                mesh = geometry.voxel_down_sample(1.0)
                mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(mesh, depth=9)

                print(mesh)

                new_vis = o3d.visualization.O3DVisualizer("Mesh Vis")
                new_vis.add_geometry(name, mesh)
                new_vis.reset_camera_to_default()
                bounds = self.widget3d.scene.bounding_box
                new_vis.setup_camera(60, bounds.get_center(), bounds.get_center() + [0, 0, -3], [0, -1, 0])
                o3d.visualization.gui.Application.instance.add_window(new_vis)
                new_vis.os_frame = o3d.visualization.gui.Rect(50, 50, new_vis.os_frame.width, new_vis.os_frame.height)

        else:
            self.info_label.text = "No Geometry Find: Please select correct geometry combo"

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        name = self.geometry_combo_name
        if name in self.geometry_map.keys():
            is_pcd = self.geometry_map[name]['is_pcd']
            geometry = self.geometry_map[name]['geometry']
            if is_pcd:
                o3d.io.write_point_cloud(filename, geometry, print_progress=True)
            else:
                if filename.split('.')[-1] == 'stl':
                    geometry.compute_triangle_normals()
                    o3d.io.write_triangle_mesh(filename, geometry)
                else:
                    o3d.io.write_triangle_mesh(filename, geometry)

    def show_axel_switch(self, is_on):
        self.widget3d.scene.show_axes(is_on)
        # if is_on:

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

    def load_process(self, name):
        raise NotImplemented

    def delete_process(self, name):
        raise NotImplemented

class CustomWindow(AppWindow):
    select_points = []

    register_fix_pcd_name = ''
    register_flex_pcd_name = ''
    register_method = ''

    def __init__(self, args):
        super(CustomWindow, self).__init__(args=args)

        ### ---------------------------------------------------------
        ### translate layout
        translate_layout = gui.Vert(self.spacing, self.margins)

        hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("x:"))
        self.x_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.x_translate.set_preferred_width(40.0)
        self.x_translate.decimal_precision = 1
        sub_hlayout.add_child(self.x_translate)
        hlayout.add_child(sub_hlayout)

        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("y:"))
        self.y_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.y_translate.set_preferred_width(40.0)
        self.y_translate.decimal_precision = 1
        sub_hlayout.add_child(self.y_translate)
        hlayout.add_child(sub_hlayout)

        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("z:"))
        self.z_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.z_translate.set_preferred_width(40.0)
        self.z_translate.decimal_precision = 1
        sub_hlayout.add_child(self.z_translate)
        hlayout.add_child(sub_hlayout)
        translate_layout.add_child(hlayout)
        translate_layout.add_stretch()

        hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("angelX:"))
        self.x_angel_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.x_angel_translate.set_preferred_width(40.0)
        self.x_angel_translate.decimal_precision = 1
        sub_hlayout.add_child(self.x_angel_translate)
        hlayout.add_child(sub_hlayout)

        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("angelY:"))
        self.y_angel_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.y_angel_translate.set_preferred_width(40.0)
        self.y_angel_translate.decimal_precision = 1
        sub_hlayout.add_child(self.y_angel_translate)
        hlayout.add_child(sub_hlayout)

        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("angelZ:"))
        self.z_angel_translate = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.z_angel_translate.set_preferred_width(40.0)
        self.z_angel_translate.decimal_precision = 1
        sub_hlayout.add_child(self.z_angel_translate)
        hlayout.add_child(sub_hlayout)
        translate_layout.add_child(hlayout)
        translate_layout.add_stretch()

        hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("Scale:"))
        self.geometry_scale = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.geometry_scale.set_preferred_width(40.0)
        self.geometry_scale.decimal_precision = 1
        sub_hlayout.add_child(self.geometry_scale)
        hlayout.add_child(sub_hlayout)

        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("Voxel:"))
        self.geometry_voxel = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.geometry_voxel.set_preferred_width(40.0)
        self.geometry_voxel.decimal_precision = 1
        sub_hlayout.add_child(self.geometry_voxel)
        hlayout.add_child(sub_hlayout)
        translate_layout.add_child(hlayout)
        translate_layout.add_stretch()

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

        ### ----------------------------------------------------------
        # self.widget3d.set_on_key(self._on_key_widget3d)

        register_layout = gui.CollapsableVert("register", self.spacing, self.margins)

        hlayout = gui.Horiz()
        hlayout.add_child(gui.Label("Fix"))
        self.register_fix_combo = gui.Combobox()
        self.register_fix_combo.set_on_selection_changed(self.register_fix_combo_change)
        self.register_fix_combo.add_item("Hello World ...")
        hlayout.add_child(self.register_fix_combo)
        register_layout.add_child(hlayout)
        register_layout.add_stretch()

        hlayout = gui.Horiz()
        hlayout.add_child(gui.Label("Flex"))
        self.register_flex_combo = gui.Combobox()
        self.register_flex_combo.set_on_selection_changed(self.register_flex_combo_change)
        self.register_flex_combo.add_item("Hello World ...")
        hlayout.add_child(self.register_flex_combo)
        register_layout.add_child(hlayout)
        register_layout.add_stretch()

        hlayout = gui.Horiz()
        hlayout.add_child(gui.Label("Method: "))
        self.register_method_combo = gui.Combobox()
        self.register_method_combo.add_item("point_to_point")
        self.register_method_combo.add_item("point_to_plane")
        self.register_method_combo.add_item("color")
        self.register_method_combo.add_item("generalized")
        self.register_method_combo.add_item("fpfh_feature_match")
        self.register_method_combo.add_item("fpfh_ransac_match")
        self.register_method_combo.set_on_selection_changed(self.register_method_change)
        hlayout.add_child(self.register_method_combo)
        register_layout.add_child(hlayout)
        register_layout.add_stretch()

        hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("Voxel:"))
        self.register_voxel = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.register_voxel.set_preferred_width(40.0)
        self.register_voxel.decimal_precision = 1
        sub_hlayout.add_child(self.register_voxel)
        hlayout.add_child(sub_hlayout)
        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("Max Iter:"))
        self.register_max_iter = gui.NumberEdit(gui.NumberEdit.INT)
        sub_hlayout.add_child(self.register_max_iter)
        hlayout.add_child(sub_hlayout)
        register_layout.add_child(hlayout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("Max Dis:"))
        self.register_max_dist = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.register_max_dist.set_preferred_width(80.0)
        self.register_max_dist.decimal_precision = 1
        sub_hlayout.add_child(self.register_max_dist)
        hlayout.add_child(sub_hlayout)

        sub_hlayout = gui.Horiz(self.spacing, self.margins)
        sub_hlayout.add_child(gui.Label("Radius:"))
        self.register_radius = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.register_radius.set_preferred_width(40.0)
        self.register_radius.decimal_precision = 1
        sub_hlayout.add_child(self.register_radius)
        hlayout.add_child(sub_hlayout)
        register_layout.add_child(hlayout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        move_origin_btn = gui.Button("To Orign")
        move_origin_btn.set_on_clicked(self.move_to_origin)
        hlayout.add_child(move_origin_btn)
        register_btn = gui.Button("Register")
        register_btn.set_on_clicked(self.register_pcd)
        hlayout.add_child(register_btn)
        merge_btn = gui.Button("Merge")
        merge_btn.set_on_clicked(self.merge_pcd)
        hlayout.add_child(merge_btn)

        register_layout.add_child(hlayout)

        self.analysis_layout.add_child(register_layout)

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

            if voxel>0:
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
                self.info_label.text = "Voxel Size Should be Larger Than 0"

        else:
            self.info_label.text = "No Geometry Find: Please select correct geometry combo"

    # def _on_key_widget3d(self, event:gui.KeyEvent):
    #     if event.type == gui.KeyEvent.DOWN:
    #         if event.key == ord('s'):
    #             pass
    #         return gui.Widget.EventCallbackResult.HANDLED
    #
    #     return gui.Widget.EventCallbackResult.IGNORED

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

    def register_fix_combo_change(self, val, idx):
        self.register_fix_pcd_name = val

    def register_flex_combo_change(self, val, idx):
        self.register_flex_pcd_name = val

    def register_method_change(self, val, idx):
        self.register_method = val

    def register_pcd(self):
        fix_name = self.register_fix_pcd_name
        flex_name = self.register_flex_pcd_name

        if (fix_name not in self.geometry_map.keys()) or (flex_name not in self.geometry_map.keys()):
            self.info_label.text = "No Geometry Find: Please select correct geometry combo"
            return

        if fix_name == flex_name:
            self.info_label.text = "Fix And Flex Is The Same"
            return

        if self.geometry_map[fix_name]['visible']:
            self.widget3d.scene.remove_geometry(fix_name)
        if self.geometry_map[flex_name]['visible']:
            self.widget3d.scene.remove_geometry(flex_name)

        fix_geometry = self.geometry_map[fix_name]['geometry']
        flex_geometry = self.geometry_map[flex_name]['geometry']

        voxel_size = self.register_voxel.double_value
        if voxel_size>0:
            fix_pcd_temp = fix_geometry.voxel_down_sample(voxel_size=voxel_size)
            flex_pcd_temp = flex_geometry.voxel_down_sample(voxel_size=voxel_size)
        else:
            fix_pcd_temp = copy(fix_geometry)
            flex_pcd_temp = copy(flex_geometry)

        max_iter = self.register_max_iter.int_value
        icp_method = self.register_method
        distance_threshold = self.register_max_dist.double_value

        result_icp = self.icp(
            source=flex_pcd_temp, target=fix_pcd_temp,
            max_iter=max_iter,
            distance_threshold=distance_threshold,
            icp_method=icp_method,
            init_transformation=np.identity(4),
            radius=voxel_size * 2.0,
            max_correspondence_distance=voxel_size * 1.4
        )

        if result_icp is None:
            self.info_label.text = "Non Support ICP Method"
            return

        # print('[DEBUG]: Info')
        # print(result_icp)
        self.info_label.text = str(result_icp)

        T = result_icp.transformation
        flex_geometry = flex_geometry.transform(T)

        is_pcd = self.geometry_map[fix_name]['is_pcd']
        if is_pcd:
            self.widget3d.scene.add_geometry(fix_name, fix_geometry, self.materials[AppWindow.LIT])
        else:
            self.widget3d.scene.add_model(fix_name, fix_geometry)

        is_pcd = self.geometry_map[flex_name]['is_pcd']
        if is_pcd:
            self.widget3d.scene.add_geometry(flex_name, flex_geometry, self.materials[AppWindow.LIT])
        else:
            self.widget3d.scene.add_model(flex_name, flex_geometry)

    def icp(self,
            source:o3d.geometry.PointCloud, target:o3d.geometry.PointCloud,
            max_iter, distance_threshold,
            icp_method='color',
            init_transformation=np.identity(4),
            radius=0.02,
            max_correspondence_distance=0.01,
            ):
        result_icp = None

        if icp_method == "point_to_point":
            result_icp = o3d.pipelines.registration.registration_icp(
                source, target,
                distance_threshold,
                init_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))

        else:
            source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
            target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

            if icp_method == "point_to_plane":
                result_icp = o3d.pipelines.registration.registration_icp(
                    source, target,
                    distance_threshold,
                    init_transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
                )

            elif icp_method == "color":
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source, target,
                    max_correspondence_distance,
                    init_transformation,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=max_iter)
                )

            elif icp_method == "generalized":
                result_icp = o3d.pipelines.registration.registration_generalized_icp(
                    source, target,
                    distance_threshold,
                    init_transformation,
                    o3d.pipelines.registration.
                        TransformationEstimationForGeneralizedICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=max_iter)
                )

            elif icp_method == 'fpfh_feature_match':
                # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

                source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_iter)
                )
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_iter)
                )

                result_icp = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                    source, target, source_fpfh, target_fpfh,
                    o3d.pipelines.registration.FastGlobalRegistrationOption(
                        maximum_correspondence_distance=distance_threshold
                    )
                )

            elif icp_method == 'fpfh_ransac_match':
                source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_iter)
                )
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_iter)
                )

                result_icp = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                    source, target, source_fpfh, target_fpfh,
                    o3d.pipelines.registration.FastGlobalRegistrationOption(
                        maximum_correspondence_distance=distance_threshold
                    )
                )

        return result_icp

    def load_process(self, name):
        self.register_fix_combo.add_item(name)
        self.register_flex_combo.add_item(name)

    def delete_process(self, name):
        self.register_fix_combo.remove_item(name)
        self.register_flex_combo.remove_item(name)

    def merge_pcd(self):
        fix_name = self.register_fix_pcd_name
        flex_name = self.register_flex_pcd_name

        if (fix_name not in self.geometry_map.keys()) or (flex_name not in self.geometry_map.keys()):
            self.info_label.text = "No Geometry Find: Please select correct geometry combo"
            return

        if fix_name == flex_name:
            self.info_label.text = "Fix And Flex Is The Same"
            return

        if self.geometry_map[fix_name]['visible']:
            self.widget3d.scene.remove_geometry(fix_name)
        if self.geometry_map[flex_name]['visible']:
            self.widget3d.scene.remove_geometry(flex_name)

        fix_geometry:o3d.geometry.PointCloud = self.geometry_map[fix_name]['geometry']
        flex_geometry:o3d.geometry.PointCloud = self.geometry_map[flex_name]['geometry']

        merge_geometry = o3d.geometry.PointCloud()
        merge_geometry.points = o3d.utility.Vector3dVector(np.concatenate([
            np.asarray(fix_geometry.points), np.asarray(flex_geometry.points)
        ], axis=0))
        merge_geometry.colors = o3d.utility.Vector3dVector(np.concatenate([
            np.asarray(fix_geometry.colors), np.asarray(flex_geometry.colors)
        ], axis=0))

        del self.geometry_map[fix_name]
        del self.geometry_map[flex_name]
        self.geometry_combo.remove_item(fix_name)
        self.geometry_combo.remove_item(flex_name)
        self.register_fix_combo.remove_item(fix_name)
        self.register_fix_combo.remove_item(flex_name)
        self.register_flex_combo.remove_item(fix_name)
        self.register_flex_combo.remove_item(flex_name)

        new_name = fix_name+'/'+flex_name
        self.widget3d.scene.add_geometry(new_name, merge_geometry, self.materials[AppWindow.LIT])
        self.geometry_map[new_name] = {'geometry': merge_geometry, 'visible':True, 'is_pcd':True}
        self.geometry_combo.add_item(new_name)
        self.register_fix_combo.add_item(new_name)
        self.register_flex_combo.add_item(new_name)

        checkbox = gui.Checkbox(new_name)
        checkbox.checked = True
        checkbox.set_on_checked(partial(self.geometry_on_click, name=new_name))
        self.geometry_layout.add_child(checkbox)

def main():
    args = parse_args()

    app = gui.Application.instance
    app.initialize()

    window = CustomWindow(args)

    app.run()

if __name__ == '__main__':
    main()
