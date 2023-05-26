import os
import numpy as np
import glob


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


if __name__ == '__main__':
    data_path = ''
    dataloader = SuperCuprWaymo(f'{data_path}/test/')

    for i in range(len(dataloader)):
        frame_name_t0 = dataloader.frames_list[i].t0_frame_name
        sequence = int(frame_name_t0.split('#')[0])
        frame_name = frame_name_t0.split('#')[1]

        os.makedirs(f'{data_path}/test/{sequence:04d}/MF', exist_ok=True)
        os.system(f'cp {data_path}/MF_output_with_classes/{i:06d}.npz {data_path}/test/{sequence:04d}/MF/{frame_name}.npz')