import time

import numpy as np
import open3d as o3d
import struct
import copy

import customtkinter
import tkinter as tk
from RangeSlider.RangeSlider import RangeSliderH


def o3d_visualization(pcl, labels, colors=True, config=None, name=f'point_cloud', bboxes=None, flow=None, pcl_t1=None,
                      labels_t1=None):

    assert len(pcl.shape) == 2 and len(labels.shape) == 1 and pcl.shape[1] == 3

    xyz = pcl[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors:
        rgb = np.zeros((len(pcl), 3))
        if config is not None:
            for i in range(len(pcl)):
                rgb[i, :] = config['color_map'][int(labels[i])][::-1]
                rgb[i, :] /= 255
        else:
            for i in range(len(pcl)):
                if labels[i] == 1:
                    rgb[i, :] = [0, 1, 0]
                else:
                    rgb[i, :] = [1, 0, 0]
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(f"{name}.ply", pcd)

    vis_components = []
    cloud = o3d.io.read_point_cloud(f"{name}.ply")  # Read the point cloud
    vis_components.append(cloud)

    if bboxes is not None:
        for bbox in bboxes:
            center = np.array([bbox['center']['x'], bbox['center']['y'], bbox['center']['z']])
            heading = bbox['heading']
            r = np.array([[np.cos(heading), -np.sin(heading), 0],
                          [np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
            size = np.array([bbox['length'], bbox['width'], bbox['height']])
            sample_3d = o3d.geometry.OrientedBoundingBox(center, r, size)
            sample_3d.color = [1, 0, 0]
            vis_components.append(copy.deepcopy(sample_3d))

    if flow is not None:
        points = []
        lines = []
        for i in range(len(flow)):
            points.append(pcl[i, :].tolist())
            points.append((pcl[i, :] + flow[i, :]).tolist())
            lines.append([2 * i, 2 * i + 1])
        colors = [[0.2, 0.2, 0.2] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis_components.append(line_set)

    if pcl_t1 is not None:
        xyz_t1 = pcl_t1[:, 0:3]
        pcd_t1 = o3d.geometry.PointCloud()
        pcd_t1.points = o3d.utility.Vector3dVector(xyz_t1)
        rgb_t1 = np.zeros((len(pcl_t1), 3))
        if labels_t1 is not None:
            for i in range(len(pcl_t1)):
                if labels_t1[i] == 1:
                    rgb_t1[i, :] = [0, 1, 1]
                else:
                    rgb_t1[i, :] = [0.5, 0, 0]
        pcd_t1.colors = o3d.utility.Vector3dVector(rgb_t1)
        vis_components.append(pcd_t1)

    o3d.visualization.draw_geometries(vis_components,
                                      zoom=0.7,
                                      front=[0.5439, 0.2333, 0.8060],
                                      lookat=[2.4615, 2.1331, 1.338],
                                      up=[-0.1781, -0.9708, 0.1608])


def ply_visualization(pcl, labels, config=None, file_name='point_cloud'):

    assert len(pcl.shape) == 2 and len(labels.shape) == 1 and pcl.shape[1] == 3, f'shape of pcl, or labels are not correct'

    assert pcl.shape[0] == labels.shape[0], f'num of points does not match with num of labels'

    rgb = np.ones(pcl.shape)

    if config is None:
        import colorsys
        max_label = np.max(labels)
        for i in range(len(labels)):
            if labels[i] == 0:
                rgb[i, :] = np.array([0, 0, 0])
            else:
                color = colorsys.hsv_to_rgb((labels[i] / max_label), 1, 1)
                rgb[i, :] = np.array([color[2], color[1], color[0]]) * 255
    else:
        for i in range(len(labels)):
            rgb[i, :] = config['color_map'][labels[i]]

    rgb = rgb.astype(np.uint8)

    fid = open(f'{file_name}.ply', 'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n' % pcl.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(pcl.shape[0]):
        fid.write(bytearray(struct.pack("fffccc", pcl[i, 0], pcl[i, 1], pcl[i, 2],
                                        rgb[i, 2].tostring(), rgb[i, 1].tostring(),
                                        rgb[i, 0].tostring())))
    fid.close()


def o3d_visualization_interactive(pcl, labels, colors=True, config=None, name=f'point_cloud', background_points=None,
                                  background_labels=None):

    assert len(pcl[0].shape) == 2 and len(labels[0].shape) == 1 and pcl[0].shape[1] == 3

    combine_pcl = pcl[0]
    max_points = (0, 0)
    for i in range(1, len(pcl)):
        combine_pcl = np.vstack((combine_pcl, pcl[i]))
        if pcl[i].shape[0] > max_points[0]:
            max_points = (pcl[i].shape[0], i)

    instance_xyz = combine_pcl
    all_xyz = np.vstack((instance_xyz, background_points))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_xyz)
    if colors:
        instance_rgb = np.zeros((combine_pcl.shape[0], 3))
        instance_rgb[:, 1] = 1
        background_rgb = np.zeros((background_labels.shape[0], 3))
        all_rgb = np.vstack((instance_rgb, background_rgb))
        pcd.colors = o3d.utility.Vector3dVector(all_rgb)
    o3d.io.write_point_cloud(f"{name}.ply", pcd)

    cloud = o3d.io.read_point_cloud(f"{name}.ply")  # Read the point cloud

    pcd_single = o3d.geometry.PointCloud()
    pcd_single.points = o3d.utility.Vector3dVector(pcl[max_points[1]])
    pcd_single.colors = o3d.utility.Vector3dVector(np.zeros((pcl[max_points[1]].shape[0], 3)))

    vis_combine = o3d.visualization.Visualizer()
    vis_combine.create_window(window_name='Combine visualization')
    vis_single = o3d.visualization.Visualizer()
    vis_single.create_window(window_name='Single visualization', width=500, height=750)

    done_labeling = False
    start_sequence = 0
    end_sequence = len(pcl)
    while not done_labeling:

        vis_combine.remove_geometry(cloud)
        vis_combine.clear_geometries()
        vis_combine.add_geometry(cloud)
        vis_combine.poll_events()
        vis_combine.update_renderer()
        vis_single.remove_geometry(pcd_single)
        vis_single.clear_geometries()
        vis_single.add_geometry(pcd_single)
        vis_single.reset_view_point(True)
        vis_single.poll_events()
        vis_single.update_renderer()

        done = False
        while not done:
            inp = input(f'Enter start time frame (max is {len(pcl)})\nQ for quit\nL for label')
            if inp in ['Q', 'L']:
                done_labeling=True
                done = True
            else:
                try:
                    start_sequence = int(inp)
                    if 0 <= start_sequence < len(pcl):
                        done = True
                except:
                    print('Wrong input')
        if done_labeling:
            break
        done = False
        while not done:
            inp = input(f'Enter end time frame (max is {len(pcl)})')

            try:
                end_sequence = int(inp)
                if start_sequence <= end_sequence < len(pcl):
                    done = True
            except:
                print('Wrong input')


        max_points = (0, 0)
        instance_rgb = None
        for i in range(len(pcl)):
            frame_rgb = np.zeros((pcl[i].shape[0], 3))
            if start_sequence <= i <= end_sequence:
                frame_rgb[:, 1] = 1
                if pcl[i].shape[0] > max_points[0]:
                    max_points = (pcl[i].shape[0], i)
            else:
                frame_rgb[:, 0] = 1
            if instance_rgb is None:
                instance_rgb = frame_rgb
            else:
                instance_rgb = np.vstack((instance_rgb, frame_rgb))

        instance_xyz = combine_pcl
        all_xyz = np.vstack((instance_xyz, background_points))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_xyz)
        if colors:
            background_rgb = np.zeros((background_labels.shape[0], 3))
            all_rgb = np.vstack((instance_rgb, background_rgb))
            pcd.colors = o3d.utility.Vector3dVector(all_rgb)
        o3d.io.write_point_cloud(f"{name}.ply", pcd)

        cloud = o3d.io.read_point_cloud(f"{name}.ply")  # Read the point cloud

        pcd_single = o3d.geometry.PointCloud()
        pcd_single.points = o3d.utility.Vector3dVector(pcl[max_points[1]] - np.mean(pcl[start_sequence], axis=0))
        pcd_single.colors = o3d.utility.Vector3dVector(np.zeros((pcl[max_points[1]].shape[0], 3)))

    vis_combine.destroy_window()
    vis_single.destroy_window()


if __name__ == '__main__':
    import glob
    import yaml

    with open(f'waymo.yaml', 'r') as f:
        config = yaml.safe_load(f)

    sequence = 497

    data_path = f'dataset/waymo_MF_instances/{sequence:04d}'

    frames_paths = sorted(glob.glob(f'{data_path}/velodyne/*.bin'))

    for i, velodyne_path in enumerate(frames_paths):

        pcl = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)

        label_path = velodyne_path.replace('velodyne', 'labels').replace('.bin', '.label')

        labels = np.fromfile(label_path, dtype=np.int32).reshape(-1)

        bbox_path = velodyne_path.replace('velodyne', 'bboxes_motion').replace('.bin', '.npz')
        bboxes = np.load(bbox_path, allow_pickle=True)
        bboxes = bboxes['bboxes']

        o3d_visualization(pcl[:, :3], labels, config=config, name=f'point_cloud', bboxes=bboxes)
