import open3d as o3d
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, help="",default=960)
    parser.add_argument("--height", type=int, help="", default=720)
    parser.add_argument("--plys", nargs='+', help="",
                        default=[
                            # '/home/psdz/Desktop/model/model1_fuse.ply',
                            # '/home/psdz/Desktop/model/model2_fuse.ply',
                            # '/home/psdz/Desktop/model/model4_fuse.ply'
                            '/home/psdz/Desktop/model/cropped_1.ply'
                        ])
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    s = "Demo for manual geometry cropping \n"
    s += "1) Press 'K' to lock screen and to switch to selection mode \n"
    s += "2) Drag for rectangle selection, or use ctrl + left click for polygon selection \n"
    s += "3) Press 'C' to get a selected geometry \n"
    s += "4) Press 'S' to save the selected geometry \n"
    s += "5) Press 'K' again to switch to freeview mode"
    print(s)

    for ply_path in args.plys:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(height=720, width=960)
        pcd = o3d.io.read_point_cloud(ply_path)
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

if __name__ == '__main__':
    main()
