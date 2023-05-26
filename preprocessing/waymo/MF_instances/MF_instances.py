import numpy as np
import os
import sys
import glob
import yaml
import copy
import tools.visualization as visualization
import scipy.spatial as spatial
from multiprocessing import Pool
import time


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


def process_sequence(dataloader, data_path, vis=False, vis_create=False):
    if vis or vis_create:
        with open(f'colors.yaml', 'r') as f:
            config = yaml.safe_load(f)

    now = False

    n_instances_idx = 1

    dyn_inst_idxs_t0 = []

    for i in range(len(dataloader)):

        frame_name_t0 = dataloader.frames_list[i].t0_frame_name
        print(frame_name_t0)
        sequence = frame_name_t0.split('#')[0]
        frame_name_t0 = frame_name_t0.split('#')[1]

        (t0_pcl, t0_mask, t0_clusters), (t1_pcl, t1_mask, t1_clusters) = dataloader._get_point_cloud_pair(i)

        t0_pcl = t0_pcl[t0_mask]
        t1_pcl = t1_pcl[t1_mask]

        MF_output = np.load(f'{data_path}/{sequence}/MF/{frame_name_t0}.npz', allow_pickle=True)
        flow = MF_output['aggregated_flow'][0]
        classes_oh = MF_output['classes_oh'][0]
        dynamic_prediction = classes_oh[:, 1]

        if vis and now:
            visualization.o3d_visualization(t0_pcl[:, :3], dynamic_prediction.astype(int), flow=flow)
            t0_labels = np.where(dynamic_prediction, 34, 1)
            visualization.o3d_visualization(np.vstack((t0_pcl[:, :3], t1_pcl[:, :3])), np.append(t0_labels, np.ones(t1_pcl.shape[0]) * 26),
                                            config=config, flow=flow)

        if vis_create:
            visualization.ply_visualization(t0_pcl[:, :3], np.where(dynamic_prediction, 34, 1), config=config,
                                            file_name=f'vis_output/{frame_name_t0}-prediction')

        if vis and now:

            visualization.o3d_visualization(t0_pcl[:, :3], (t0_clusters.astype(int)) % 35, config=config)
        if vis_create:
            visualization.ply_visualization(t0_pcl[:, :3], (t0_clusters.astype(int)) % 35, config=config,
                                            file_name=f'vis_output/{frame_name_t0}-dbscan')

        dynamic_mask = np.zeros(t0_pcl.shape[0])
        instances_t0 = np.zeros(t0_pcl.shape[0])

        for inst_idx in np.unique(t0_clusters):
            if inst_idx == -1:
                continue

            instance_mask = t0_clusters == inst_idx

            from_previous_frame = False
            for idx_cell in dyn_inst_idxs_t0:
                if inst_idx == idx_cell.dbscan_idx:
                    dynamic_mask[instance_mask] = 2
                    instances_t0[instance_mask] = idx_cell.instances_idx
                    from_previous_frame = True
                    break
            if from_previous_frame:
                continue

            if np.count_nonzero(dynamic_prediction * instance_mask) >= np.count_nonzero(instance_mask) / 2:
                dynamic_mask[instance_mask] = 1
                instances_t0[instance_mask] = n_instances_idx

                indexes_cell = DynamicInstanceIdexes(instance_idx=n_instances_idx, dbscan_idx=inst_idx)
                if indexes_cell not in dyn_inst_idxs_t0:
                    dyn_inst_idxs_t0.append(indexes_cell)
                n_instances_idx += 1

        if vis and now:
            visualization.o3d_visualization(t0_pcl[:, :3], dynamic_mask.astype(int), config=config, flow=flow)
        if vis_create:
            visualization.ply_visualization(t0_pcl[:, :3], dynamic_mask.astype(int), config=config,
                                            file_name=f'vis_output/{frame_name_t0}-dyn-mask')

        t1_pcl_ex = np.hstack((t1_pcl, t1_clusters.reshape(-1, 1) * 1e-10))

        radius = 0.1
        point_tree = spatial.cKDTree(t1_pcl_ex)

        instances_t1 = np.zeros(t1_pcl.shape[0])

        dyn_inst_idxs_t1 = []

        for inst_idx in dyn_inst_idxs_t0:

            instance_t0_mask = instances_t0 == inst_idx.instances_idx
            if vis and now:
                visualization.o3d_visualization(t0_pcl[:, :3], instance_t0_mask.astype(int))

            all_neighbors = np.array([])

            timeout_time = time.time() + 10

            for (pts_xyz, pts_flow) in zip(np.hstack((t0_pcl, np.zeros((t0_pcl.shape[0], 1))))[instance_t0_mask],
                                           np.hstack((flow, np.zeros((flow.shape[0], 1))))[instance_t0_mask]):

                neighbors = point_tree.query_ball_point(pts_xyz + pts_flow, radius)

                all_neighbors = np.append(all_neighbors, neighbors)

                if time.time() > timeout_time:
                    print(f'TIMEOUT: frame={frame_name_t0} n_points={np.count_nonzero(instance_t0_mask)}')
                    break

            all_neighbors = np.unique(all_neighbors).astype(int)
            if len(all_neighbors) == 0:
                instances_t0 = np.where(instances_t0 == inst_idx.instances_idx, 0, instances_t0)
                continue
            dbscan_t1_idxs = t1_pcl_ex[all_neighbors, 3] * 1e10

            n_i, i_c = np.unique(dbscan_t1_idxs, return_counts=True)

            if n_i[np.argmax(i_c)].astype(int) == -1:
                i_c[np.argmax(i_c)] = -1
                dbscan_next = n_i[np.argmax(i_c)].astype(int)
                if dbscan_next == -1:
                    continue
            else:
                dbscan_next = n_i[np.argmax(i_c)].astype(int)

            instance_t1_mask = t1_clusters == dbscan_next

            if abs(np.count_nonzero(instance_t0_mask) - np.count_nonzero(instance_t1_mask)) > 0.4 * np.count_nonzero(
                    instance_t0_mask):
                continue

            instances_t1[instance_t1_mask] = inst_idx.instances_idx
            dyn_inst_idxs_t1.append(DynamicInstanceIdexes(instance_idx=inst_idx.instances_idx, dbscan_idx=dbscan_next))

        if vis and now:
            visualization.o3d_visualization(np.vstack((t0_pcl, t1_pcl[:, :3])), np.append(instances_t0, instances_t1)
                                            % 30, config=config, flow=flow)
        if vis_create:
            visualization.ply_visualization(np.vstack((t0_pcl, t1_pcl[:, :3])), np.append(instances_t0, instances_t1)
                                            % 35, config=config, file_name=f'vis_output/{frame_name_t0}-match')

        dyn_inst_idxs_t0 = copy.deepcopy(dyn_inst_idxs_t1)
        os.makedirs(f'{data_path}/{sequence}/MF_instances', exist_ok=True)
        save_t0_instances = np.zeros(t0_mask.shape[0])
        save_t0_instances[t0_mask] = instances_t0
        np.save(f'{data_path}/{sequence}/MF_instances/{frame_name_t0}.npy', save_t0_instances)
        if i == len(dataloader) - 1:
            frame_name_t1 = dataloader.frames_list[i].t1_frame_name
            print(frame_name_t1)
            sequence = frame_name_t1.split('#')[0]
            frame_name_t1 = frame_name_t1.split('#')[1]
            save_t1_instances = np.zeros(t1_mask.shape[0])
            save_t1_instances[t1_mask] = instances_t1
            np.save(f'{data_path}/{sequence}/MF_instances/{frame_name_t1}.npy', save_t1_instances)


def prepare_sequence(seq):
    vis = False
    vis_create = False
    data_path = ''

    dataloader = SuperCuprWaymo(data_path, sequence=f'{seq:04d}', return_mask=True, return_clusters=True)

    process_sequence(dataloader, data_path, vis=vis, vis_create=vis_create)


if __name__ == '__main__':

    mutl_proc = True

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
        train_sequences = np.array([307, 308, 310, 312, 498, 499])
    else:
        assert False, f'Wrong sequence specification: {sys.argv[1]}'

    work = train_sequences.tolist()
    print(work)

    if mutl_proc:
        cpus = min(20, len(work))

        p = Pool(cpus)
        p.map(prepare_sequence, work)
    else:
        for i in work:
            prepare_sequence(i)

