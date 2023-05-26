import numpy as np
import os
import sys
import glob
import yaml
import copy
import json
import tools.visualization as visualization
from multiprocessing import Pool


class InstanceHandler():
    def __init__(self):
        self.active_dataset_indexes = []
        self.active_motion_flow_indexes = []
        self.next_instance_idx = 1

    def dataset2motion_index(self, dataset_index, sequence):
        mf_index = None
        new = False
        if dataset_index in self.active_dataset_indexes:
            mf_index = self.active_motion_flow_indexes[self.active_dataset_indexes.index(dataset_index)]
        else:
            mf_index = self.next_instance_idx + (1000 + sequence) * 10000
            self.next_instance_idx += 1
            assert self.next_instance_idx < 10000, f'Too many instances in sequence, please number pool'
            self.active_dataset_indexes.append(dataset_index)
            self.active_motion_flow_indexes.append(mf_index)
            new = True
        return mf_index, new

    def active_check(self, active_idxs):

        for idx in self.active_dataset_indexes:
            if idx not in active_idxs:
                item_index = self.active_dataset_indexes.index(idx)
                self.active_dataset_indexes.pop(item_index)
                self.active_motion_flow_indexes.pop(item_index)

    def new_sequence(self):
        self.active_dataset_indexes = []
        self.active_motion_flow_indexes = []
        self.next_instance_idx = 1


def process_sequence(data_path, seq, dictionary, vis=False, vis_create=False):
    if vis or vis_create:
        with open(f'colors.yaml', 'r') as f:
            config = yaml.safe_load(f)

    i_handler = InstanceHandler()

    instances_paths = sorted(glob.glob(f'{data_path}/{seq:04d}/MF_instances_back/*.npy'))

    for instance_path in instances_paths:

        frame = instance_path.split('/')[-1].split('.')[0]

        label_path = instance_path.replace('MF_instances_back', 'labels').replace('.npy', '.label')

        MF_instances = np.load(instance_path)

        all_labels = np.fromfile(label_path, dtype=np.int32).reshape(-1, 1)
        labels_ = all_labels & 0xFFFF

        annotator_instances = copy.deepcopy(MF_instances)
        annotator_labels = copy.deepcopy(labels_)

        if vis:
            velodyne_path = instance_path.replace('MF_instances_back', 'velodyne').replace('.npy', '.bin')
            pcl = np.fromfile(velodyne_path, dtype=np.float32).reshape((-1, 4))
            destroied_instances = np.zeros(MF_instances.shape[0])

        frame_dynamic_instances = []

        for i_idx in np.unique(MF_instances):
            if i_idx == 0:
                continue

            sample_mask = MF_instances == i_idx

            sample_labels = labels_[sample_mask]

            sample_label_list, sample_label_count = np.unique(sample_labels, return_counts=True)

            if np.max(sample_label_count) >= 0.9 * sample_labels.shape[0] and sample_labels.shape[0] - np.max(sample_label_count) < 100:
                annotator_labels[sample_mask] = sample_label_list[np.argmax(sample_label_count)]
            else:
                annotator_instances[sample_mask] = 0
                if vis:
                    destroied_instances[sample_mask] = 1
                continue

            frame_dynamic_instances.append(i_idx)

            dictionary_idx, new = i_handler.dataset2motion_index(i_idx, int(seq))

            annotator_instances[sample_mask] = dictionary_idx

            if new:
                dictionary[dictionary_idx] = [f'{seq:04d}_{frame}']
            else:
                dictionary[dictionary_idx].append(f'{seq:04d}_{frame}')

        if vis:
            visualization.o3d_visualization(pcl[:, :3], destroied_instances, config=config)
            visualization.o3d_visualization(pcl[:, :3], labels_.flatten(), config=config)
            visualization.o3d_visualization(pcl[:, :3], MF_instances % 35, config=config)
            visualization.o3d_visualization(pcl[:, :3], annotator_instances % 35, config=config)

        assert len(np.unique(annotator_instances)) - 1 == len(
            frame_dynamic_instances), f'Something goes wrong {seq}/{frame}'

        i_handler.active_check(frame_dynamic_instances)

        os.makedirs(f'{data_path}/{seq:04d}/annotator_instances', exist_ok=True)
        save_instances = open(f'{data_path}/{seq:04d}/annotator_instances/{frame}.label', 'wb')
        save_instances.write(annotator_instances.astype(np.uint32))
        save_instances.close()
        os.makedirs(f'{data_path}/{seq:04d}/annotator_labels', exist_ok=True)
        save_labels = open(f'{data_path}/{seq:04d}/annotator_labels/{frame}.label', 'wb')
        save_labels.write(annotator_labels.astype(np.uint32))
        save_labels.close()

    return dictionary


def prepare_sequence(seq, dictionary):
    vis = False
    vis_create = False
    data_path = 'Waymo/parsed_waymo'

    dictionary = process_sequence(data_path, seq, dictionary, vis=vis, vis_create=vis_create)

    return dictionary


if __name__ == '__main__':

    mutl_proc = False

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

    dictionary = dict()

    if mutl_proc:
        cpus = min(20, len(work))

        p = Pool(cpus)
        p.map(prepare_sequence, work)
    else:
        for i in work:
            print(i)
            dictionary = prepare_sequence(i, dictionary)

        os.makedirs('output/', exist_ok=True)
        with open("output/MF_dynamic_instances_waymo.txt", "w") as fp:
            json.dump(dictionary, fp)
        print('\rDictionary was saved.')

