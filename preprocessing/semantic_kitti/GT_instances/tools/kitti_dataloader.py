import numpy as np
import glob


class KittiLoader():
    def __init__(self, dataset_path, sequences=['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']):

        self.dataset_path = dataset_path
        self.velodyne_paths = []

        for seq in sequences:
            sequence_velodynes = sorted(glob.glob(f'{dataset_path}/sequences/{seq}/velodyne/*.bin'))
            if len(sequence_velodynes) == 0:
                print(f'Sequence {seq} is empty.')
            else:
                self.velodyne_paths = self.velodyne_paths + sequence_velodynes

        self.velo_2_cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                                    [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                                    [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                                    [0, 0, 0, 1]])
        self.my_calib = np.array([[0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 0, 1]])

    def __len__(self):
        return len(self.velodyne_paths)

    def __getitem__(self, index):

        velodyne = np.fromfile(self.velodyne_paths[index], dtype=np.float32).reshape(-1, 4)

        label_path = self.velodyne_paths[index].replace('velodyne', 'labels').replace('.bin', '.label')
        all_labels = np.fromfile(label_path, dtype=np.int32).reshape(-1)
        labels = all_labels & 0xFFFF
        instances = all_labels >> 16

        supervoxel_path = self.velodyne_paths[index].replace('velodyne', 'supervoxel')
        supervoxels = np.fromfile(supervoxel_path, dtype=np.int32)

        frame_idx = self.velodyne_paths[index].split('/')[-1].split('.')[0]
        sequence_idx = self.velodyne_paths[index].split('/')[-3]

        poses = np.loadtxt(f'{self.dataset_path}/sequences/{sequence_idx}/poses.txt')
        transform_matrix = self.create_transform_matrix(poses=poses, frame_idx=int(frame_idx))

        return velodyne, labels, instances, transform_matrix, sequence_idx, frame_idx, supervoxels

    def create_transform_matrix(self, poses, frame_idx):
        pose = poses[frame_idx].reshape(3, 4)
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))

        transform_matrix = np.dot(pose, self.velo_2_cam)
        return np.dot(np.linalg.inv(self.my_calib), transform_matrix)

