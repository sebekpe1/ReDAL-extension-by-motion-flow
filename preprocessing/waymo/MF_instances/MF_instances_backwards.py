import numpy as np
import os
import sys
import glob
import yaml
import time
import scipy.spatial as spatial
from multiprocessing import Pool
import tools.visualization as visualization


class FrameCell():
    def __init__(self, t0_frame, t1_frame):
        self.t0_frame_name = t0_frame
        self.t1_frame_name = t1_frame


class SuperCuprWaymo():
    def __init__(self, data_path, sequence='0000', return_mask=False, return_clusters=False):

        self.data_path = data_path
        self.return_mask = return_mask
        self.return_clusters = return_clusters
        self.frames_list = []

        frames = sorted(glob.glob(f'{data_path}/{sequence}/velodyne/*.bin'))

        last_frame_idx = len(glob.glob(f'{data_path}/{sequence}/velodyne/*_000.bin')) - 1

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

        t0_clusters = np.load(t0_frame_path.replace('velodyne', 'clusters').replace('.bin', '.npy'))
        t1_clusters = np.load(t1_frame_path.replace('velodyne', 'clusters').replace('.bin', '.npy'))

        if self.return_mask and self.return_clusters:
            return (t0_frame[:, :3], t0_mask, t0_clusters), (t1_frame[:, :3], t1_mask, t1_clusters)
        elif self.return_mask:
            return (t0_frame[:, :3], t0_mask), (t1_frame[:, :3], t1_mask)
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

        t0_to_t1 = np.linalg.inv(t1_pose) @ t0_pose
        return t0_to_t1


class DynamicInstanceIdexes():
    def __init__(self, instance_idx, dbscan_idx):
        self.instances_idx = instance_idx
        self.dbscan_idx = dbscan_idx


def process_sequence(dataloader, data_path, vis=False):
    if vis:
        with open(f'colors.yaml', 'r') as f:
            config = yaml.safe_load(f)

    for i in range(len(dataloader) - 1, -1, -1):

        frame_name_t0 = dataloader.frames_list[i].t0_frame_name
        print(frame_name_t0)
        sequence = frame_name_t0.split('#')[0]
        frame_name_t0 = frame_name_t0.split('#')[1]

        frame_name_t1 = dataloader.frames_list[i].t1_frame_name
        assert sequence == frame_name_t1.split('#')[0], f'Sequences are not equal'
        frame_name_t1 = frame_name_t1.split('#')[1]

        (t0_pcl, t0_mask, t0_clusters), (t1_pcl, t1_mask, t1_clusters) = dataloader._get_point_cloud_pair(i)
        t0_pcl = t0_pcl[t0_mask]
        t1_pcl = t1_pcl[t1_mask]

        t0_instances = np.load(f'{data_path}/{sequence}/MF_instances/{frame_name_t0}.npy')
        if os.path.exists(f'{data_path}/{sequence}/MF_instances_back/{frame_name_t1}.npy'):
            t1_instances = np.load(f'{data_path}/{sequence}/MF_instances_back/{frame_name_t1}.npy')
            print('New t1 instances')
        else:
            t1_instances = np.load(f'{data_path}/{sequence}/MF_instances/{frame_name_t1}.npy')
        t0_instances = t0_instances[t0_mask]
        t1_instances = t1_instances[t1_mask]

        dyn_inst_miss = []
        t0_dyn_inst = np.unique(t0_instances)
        t1_dyn_inst = np.unique(t1_instances)
        for t1_inst in t1_dyn_inst:
            if t1_inst not in t0_dyn_inst:
                dyn_inst_miss.append(t1_inst)

        if len(dyn_inst_miss) == 0:
            os.makedirs(f'{data_path}/{sequence}/MF_instances_back', exist_ok=True)
            save_t0_instances = np.zeros(t0_mask.shape[0])
            save_t0_instances[t0_mask] = t0_instances
            np.save(f'{data_path}/{sequence}/MF_instances_back/{frame_name_t0}.npy', save_t0_instances)
            if i == len(dataloader) - 1:
                save_t1_instances = np.zeros(t1_mask.shape[0])
                save_t1_instances[t1_mask] = t1_instances
                np.save(f'{data_path}/{sequence}/MF_instances_back/{frame_name_t1}.npy', save_t1_instances)
            continue

        if vis and i == 0:
            visualization.o3d_visualization(t1_pcl, np.where(np.isin(t1_instances, dyn_inst_miss), 1, 0))
            visualization.o3d_visualization(t1_pcl, t1_instances % 35, config=config)
            visualization.o3d_visualization(t0_pcl, t0_instances % 35, config=config)

        MF_output = np.load(f'{data_path}/{sequence}/MF/{frame_name_t0}.npz', allow_pickle=True)
        flow = MF_output['aggregated_flow'][0]

        t0_pcl_ex = np.hstack((t0_pcl + flow, t0_clusters.reshape(-1, 1) * 1e-10))

        radius = 0.1
        point_tree = spatial.cKDTree(t0_pcl_ex)

        for inst_idx in dyn_inst_miss:

            instance_t1_mask = t1_instances == inst_idx

            all_neighbors = np.array([])

            timeout_time = time.time() + 10

            for pts_xyz in np.hstack((t1_pcl, np.zeros((t1_pcl.shape[0], 1))))[instance_t1_mask]:

                neighbors = point_tree.query_ball_point(pts_xyz, radius)

                all_neighbors = np.append(all_neighbors, neighbors)

                if time.time() > timeout_time:
                    print(f'TIMEOUT: frame={frame_name_t0} n_points={np.count_nonzero(instance_t1_mask)}')
                    break

            all_neighbors = np.unique(all_neighbors).astype(int)
            if len(all_neighbors) == 0:
                # instances_t0 = np.where(instances_t0 == inst_idx, 0, instances_t0)
                continue
            dbscan_t0_idxs = t0_pcl_ex[all_neighbors, 3] * 1e10

            n_i, i_c = np.unique(dbscan_t0_idxs, return_counts=True)

            if n_i[np.argmax(i_c)].astype(int) == -1:
                i_c[np.argmax(i_c)] = -1
                dbscan_next = n_i[np.argmax(i_c)].astype(int)
                if dbscan_next == -1:
                    continue
            else:
                dbscan_next = n_i[np.argmax(i_c)].astype(int)

            instance_t0_mask = t0_clusters == dbscan_next

            if vis and i == 0:
                visualization.o3d_visualization(np.vstack((t0_pcl, t1_pcl)), np.append(instance_t0_mask.astype(int), instance_t1_mask.astype(int) * 2), config=config)


            if abs(np.count_nonzero(instance_t0_mask) - np.count_nonzero(instance_t1_mask)) > 0.4 * np.count_nonzero(
                    instance_t0_mask):
                continue

            print(f'successful match')
            t0_instances[instance_t0_mask] = inst_idx

        if vis and i==0:
            visualization.o3d_visualization(np.vstack((t0_pcl, t1_pcl)), np.append(t0_instances, t1_instances)
                                            % 30, config=config, flow=flow)

        os.makedirs(f'{data_path}/{sequence}/MF_instances_back', exist_ok=True)
        save_t0_instances = np.zeros(t0_mask.shape[0])
        save_t0_instances[t0_mask] = t0_instances
        np.save(f'{data_path}/{sequence}/MF_instances_back/{frame_name_t0}.npy', save_t0_instances)
        if i == len(dataloader) - 1:
            save_t1_instances = np.zeros(t1_mask.shape[0])
            save_t1_instances[t1_mask] = t1_instances
            np.save(f'{data_path}/{sequence}/MF_instances_back/{frame_name_t1}.npy', save_t1_instances)


def prepare_sequence(seq):
    vis = False
    data_path = ''

    dataloader = SuperCuprWaymo(data_path, sequence=f'{seq:04d}', return_mask=True, return_clusters=True)

    if len(dataloader) != len(glob.glob(f'{data_path}/{seq:04d}/ground_mask/*.npy')) - 1:
        print(f'Sequence {seq:04d} is not ready yet')
        return 0
    if len(dataloader) == len(glob.glob(f'{data_path}/{seq:04d}/MF_instances_back/*.npy')) - 1:
        return 0

    process_sequence(dataloader, data_path, vis=vis)


if __name__ == '__main__':
    mult_proc = True

    train_sequences = np.arange(798)

    data = np.load(f'dataset_split.npz', allow_pickle=True)
    val_sequences = data['val_sequences']

    train_sequences = np.delete(train_sequences, val_sequences)

    if sys.argv[1] == 'lower':
        train_sequences = train_sequences[:int(len(train_sequences)/2)]
    elif sys.argv[1] == 'upper':
        train_sequences = train_sequences[int(len(train_sequences)/2):]
    elif sys.argv[1] == 'all':
        pass
    elif sys.argv[1] == 'single':
        train_sequences = np.array([int(sys.argv[2])])
    elif sys.argv[1] == 'custom':
        train_sequences = np.array([306, 307, 497, 498, 499, 501, 502, 585, 586, 587, 588, 589, 590])
    else:
        assert False, f'Wrong sequence specification: {sys.argv[1]}'

    work = train_sequences.tolist()

    if mult_proc:
        cpus = min(20, len(work))

        p = Pool(cpus)
        p.map(prepare_sequence, work)
    else:
        for i in work:
            prepare_sequence(i)
