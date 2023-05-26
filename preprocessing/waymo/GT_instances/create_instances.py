import glob
import numpy as np
import os
import json
from tools.bbox_crop import cut_bounding_box


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
            mf_index = self.next_instance_idx + (1000 + sequence) * 1000
            self.next_instance_idx += 1
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


if __name__ == '__main__':

    dataset_path = 'Waymo/parsed_waymo'

    dictionary = dict()

    i_handler = InstanceHandler()

    for seq_idx in range(798):

        print(f'\rStarting new sequence {seq_idx:04d}. Directory contains {len(dictionary)} instances')
        i_handler.new_sequence()

        velodyne_paths = sorted(glob.glob(f'{dataset_path}/{seq_idx:04d}/velodyne/*.bin'))

        for velodyne_path in velodyne_paths:

            frame = velodyne_path.split('/')[-1].split('.')[0]

            bboxes_path = velodyne_path.replace('velodyne', 'bboxes_motion').replace('.bin', '.npz')
            label_path = velodyne_path.replace('velodyne', 'labels').replace('.bin', '.label')

            velodyne = np.fromfile(velodyne_path, dtype=np.float32).reshape((-1, 4))

            all_labels = np.fromfile(label_path, dtype=np.int32).reshape(-1, 1)
            labels_ = all_labels & 0xFFFF

            velodyne = np.hstack((velodyne, labels_))

            bboxes = np.load(bboxes_path, allow_pickle=True)
            bboxes = bboxes['bboxes']

            dynamic_instances = np.zeros(velodyne.shape[0])

            frame_dynamic_instances = []

            for bbox in bboxes:

                if bbox['static']:
                    continue
                if bbox['class'] in ['UNKNOWN', 'SIGN']:
                    continue

                sample_mask = cut_bounding_box(velodyne, bbox, return_mask=True)

                if not (sample_mask).any():
                    print(f'Dynamic object with zero points')
                    continue

                sample_points = dynamic_instances[sample_mask]
                free_points = sample_points.shape[0] - np.count_nonzero(sample_points)
                if bbox['class'] == 'VEHICLE' and min(np.count_nonzero(sample_mask), free_points) < 40:
                    print(f'Small number of points for vehicle')
                    continue
                if bbox['class'] in ['PEDESTRIAN', 'CYCLIST'] and min(np.count_nonzero(sample_mask), free_points) < 20:
                    print(f'Small number of points for pedestrian/cyclist')
                    continue

                frame_dynamic_instances.append(bbox['ID'])

                dictionary_idx, new = i_handler.dataset2motion_index(bbox['ID'], int(seq_idx))

                free_points_mask = (dynamic_instances == 0)

                dynamic_instances = np.where(sample_mask * free_points_mask, dictionary_idx, dynamic_instances)

                if new:
                    dictionary[dictionary_idx] = [f'{seq_idx:04d}_{frame}']
                else:
                    dictionary[dictionary_idx].append(f'{seq_idx:04d}_{frame}')

                assert len(np.unique(dynamic_instances)) - 1 == len(frame_dynamic_instances), f'Something goes wrong {seq_idx}/{frame}'

            i_handler.active_check(frame_dynamic_instances)

            os.makedirs(f'{dataset_path}/{seq_idx:04d}/instances_GT', exist_ok=True)
            save_labels = open(f'{dataset_path}/{seq_idx:04d}/instances_GT/{frame}.label', 'wb')
            save_labels.write(dynamic_instances.astype(np.uint32))
            save_labels.close()

    os.makedirs('output/', exist_ok=True)
    with open("output/dynamic_instances_waymo_GT.txt", "w") as fp:
        json.dump(dictionary, fp)
    print('\rDictionary was saved.')
