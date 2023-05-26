import os
import numpy as np
import glob
from sklearn.cluster import DBSCAN
from multiprocessing import Pool


class FrameCell():
    def __init__(self, t0_frame, t1_frame):
        self.t0_frame_name = t0_frame
        self.t1_frame_name = t1_frame


class SuperCuprWaymo():
    def __init__(self, data_path, return_mask=False):

        self.data_path = data_path
        self.return_mask = return_mask
        self.frames_list = []

        data = np.load(f'dataset_split.npz', allow_pickle=True)
        val_sequences = data['val_sequences']

        sequences_directories = sorted(glob.glob(f'{data_path}/*/'))

        for s_dir in sequences_directories:

            sequence = s_dir.split('/')[-2]

            if sequence == 'MF_output_with_classes':
                continue
            if int(sequence) in val_sequences:
                continue

            frames = sorted(glob.glob(f'{s_dir}velodyne/*.bin'))

            last_frame_idx = len(glob.glob(f'{s_dir}velodyne/*_000.bin')) - 1

            for i in range(len(frames)):
                frame_name = frames[i].split('/')[-1].split('.')[0]

                if int(frame_name.split('_')[0]) >= last_frame_idx:
                    continue

                self.frames_list.append(FrameCell(f"{sequence}#{frames[i].split('/')[-1].split('.')[0]}",
                                                  f"{sequence}#{frames[i+1].split('/')[-1].split('.')[0]}"))

    def __len__(self):
        """
        For each dataset should be separetly written.
        Returns:
            length of the dataset
        """
        return len(self.frames_list)

    def _get_point_cloud_pair(self, index):
        """
        For each dataset should be separetly written. Returns two consecutive point clouds.
        Args:
            index:

        Returns:
            t0_frame: pointcloud in shape [N, features]
            t1_frame: pointcloud in shape [N, features]
        """
        t0_frame_name = self.frames_list[index].t0_frame_name
        t1_frame_name = self.frames_list[index].t1_frame_name

        t0_frame_path = f"{self.data_path}/{t0_frame_name.split('#')[0]}/velodyne/{t0_frame_name.split('#')[1]}.bin"
        t1_frame_path = f"{self.data_path}/{t1_frame_name.split('#')[0]}/velodyne/{t1_frame_name.split('#')[1]}.bin"

        t0_frame = np.fromfile(t0_frame_path, dtype=np.float32).reshape(-1, 4)
        t1_frame = np.fromfile(t1_frame_path, dtype=np.float32).reshape(-1, 4)

        t0_ground_mask = np.load(t0_frame_path.replace('velodyne', 'ground_mask').replace('.bin', '.npy'))
        t1_ground_mask = np.load(t1_frame_path.replace('velodyne', 'ground_mask').replace('.bin', '.npy'))

        # combine mask ground and crop frame (square 70 x 70, center in ego)
        t0_mask = ~t0_ground_mask * (abs(t0_frame[:, 0]) < 35.) * (abs(t0_frame[:, 1]) < 35.)
        t1_mask = ~t1_ground_mask * (abs(t1_frame[:, 0]) < 35.) * (abs(t1_frame[:, 1]) < 35.)

        if self.return_mask:
            return (t0_frame, t0_mask), (t1_frame, t1_mask)
        else:
            t0_frame = t0_frame[t0_mask]
            t1_frame = t1_frame[t1_mask]
            return t0_frame[:, :3], t1_frame[:, :3]

    def _get_pose_transform(self, index):
        """
        For each dataset should be separetly written. Returns transforamtion from t0 to t1
        Returns:
            t0_to_t1: in shape [4, 4]
        """
        t0_frame_name = self.frames_list[index].t0_frame_name
        t1_frame_name = self.frames_list[index].t1_frame_name

        t0_pose = np.load(f"{self.data_path}/{t0_frame_name.split('#')[0]}/poses/{t0_frame_name.split('#')[1]}.npy")
        t1_pose = np.load(f"{self.data_path}/{t1_frame_name.split('#')[0]}/poses/{t1_frame_name.split('#')[1]}.npy")

        t0_to_t1 = t0_pose @ np.linalg.inv(t1_pose)
        return t0_to_t1


def create_clustering_labels(i):

    print(i)
    file_name_t0 = dataloader.frames_list[i].t0_frame_name
    file_name_t1 = dataloader.frames_list[i].t1_frame_name

    file_sequence_t0 = file_name_t0.split('#')[0]
    file_sequence_t1 = file_name_t1.split('#')[0]
    assert file_sequence_t0 == file_sequence_t1, f'sequence of t0 frame and t1 frame are not equal, frame {i}'
    os.makedirs(f'{data_path}/{file_sequence_t0}/clusters', exist_ok=True)

    file_frame_t0 = file_name_t0.split('#')[1]
    file_frame_t1 = file_name_t1.split('#')[1]

    save_name_t0 = f'{data_path}/{file_sequence_t0}/clusters/{file_frame_t0}.npy'
    save_name_t1 = f'{data_path}/{file_sequence_t0}/clusters/{file_frame_t1}.npy'

    if (not os.path.exists(save_name_t0)) or (not os.path.exists(save_name_t1)):

        t0, t1 = dataloader._get_point_cloud_pair(i)

        if not os.path.exists(save_name_t0):

            xyz = t0[:, :3]

            clustering_t0 = DBSCAN(eps=0.75, min_samples=20).fit(xyz)

            np.save(save_name_t0, clustering_t0.labels_)

        if not os.path.exists(save_name_t1):

            xyz = t1[:, :3]

            clustering_t1 = DBSCAN(eps=0.75, min_samples=20).fit(xyz)

            np.save(save_name_t1, clustering_t1.labels_)


if __name__ == '__main__':
    data_path = ''
    dataloader = SuperCuprWaymo(data_path)

    frames_indexes = np.arange(len(dataloader))

    frames_indexes = frames_indexes[:int(len(frames_indexes)/2)]

    work = frames_indexes.tolist()

    cpus = 20
    p = Pool(cpus)
    p.map(create_clustering_labels, work)
