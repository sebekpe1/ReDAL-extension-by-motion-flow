import glob
import yaml
import visualization
import sys

from gui import *


classes = ["Undefined", "Car", "Truck", "Bus", "Other vehicle", "Motorcyclist", "Bicyclist", "Pedestrian", "Sign",
           "Traffic-light", "Pole", "Construction cone", "Bicycle", "Motorcycle", "Building", "Vegetation", "Tree trunk",
           "Curb", "Road", "Lane marker", "Other ground", "Walkable", "Sidewalk"]


if __name__ == '__main__':

    with open(f'colors.yaml', 'r') as f:
        config = yaml.safe_load(f)

    single_background = True
    instance_folder = 'MF_instances_back'

    if len(sys.argv) == 1:
        sequence = 497
        sample_instance_index = 10
    elif len(sys.argv) == 2:
        example_1 = int(sys.argv[1]) == 1
        if example_1:
            sequence = 497
            sample_instance_index = 10
        else:
            sequence = 1
            sample_instance_index = 20
    else:
        if int(sys.argv[1]) == 1:
            sequence = 1
        else:
            sequence = 497
        sample_instance_index = int(sys.argv[2])

    data_path = f'dataset/waymo_MF_instances/{sequence:04d}'

    frames_paths = sorted(glob.glob(f'{data_path}/velodyne/*.bin'))

    t_pose = np.load(frames_paths[0].replace('velodyne', 'poses').replace('.bin', '.npy'))
    t_pose = np.linalg.inv(t_pose)

    combine_instance = []
    combine_labels = []

    for i, velodyne_path in enumerate(frames_paths):

        pcl = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)

        pose = np.load(velodyne_path.replace('velodyne', 'poses').replace('.bin', '.npy'))

        pcl[:, 3] = 1

        t_matrix = t_pose @ pose

        pcl = pcl @ t_matrix.T
        pcl = pcl[:, :3]

        instances = np.load(velodyne_path.replace('velodyne', instance_folder).replace('.bin', '.npy'))

        sample_mask = instances == sample_instance_index
        if np.count_nonzero(sample_mask) != 0:
            print(f'Number of points: {np.count_nonzero(sample_mask)}')

        sample_points = pcl[sample_mask]

        if single_background:
            if np.count_nonzero(sample_mask) > 0:
                combine_instance.append(sample_points)
                combine_labels.append(np.ones(sample_points.shape[0]))

            if i == len(frames_paths) - 1:
                background_points = pcl[~sample_mask]
                background_labels = np.zeros(background_points.shape[0])

                vis = App(classes, 0, len(combine_instance), pcl=combine_instance,
                          labels=combine_labels, config=config, background_points=background_points,
                          background_labels=background_labels)
                print(vis.start())

        else:
            combine_instance.append(sample_points)
            combine_labels.append(np.ones(sample_points.shape[0]))

            visualization.o3d_visualization(combine_instance, combine_labels, config=config)
