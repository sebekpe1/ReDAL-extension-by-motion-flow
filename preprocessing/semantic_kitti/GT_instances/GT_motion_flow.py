import numpy as np
import os
import json

from tools.kitti_dataloader import KittiLoader


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
            mf_index = self.next_instance_idx + (10 + sequence) * 100000
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

    dataset_path = 'SemanticKitti/dataset'

    dataloader = KittiLoader(dataset_path)

    print(f'Number of files in dataset {len(dataloader)}')

    dictionary = dict()

    i_handler = InstanceHandler()

    old_sequence = '-1'

    for idx, (pcl, labels, instances, t_matrix, sequence, frame, _) in enumerate(dataloader):

        if old_sequence != sequence:
            old_sequence = sequence
            print(f'\rStarting new sequence. Directory contains {len(dictionary)} instances')
            i_handler.new_sequence()

        dataset_instances, dataset_instances_counts = np.unique(instances, return_counts=True)

        print(f'\r{idx}\t/{len(dataloader)} --≥ {idx / len(dataloader) * 100:.01f}%', end='')

        dataset_instances = dataset_instances.tolist()
        dataset_instances.remove(0)

        dynamic_instances = np.zeros_like(instances)

        frame_dynamic_instances = []

        for dataset_ins in dataset_instances:

            objects_labels = labels[np.where(instances == dataset_ins)]

            objects_labels = np.unique(objects_labels)

            for object_label in objects_labels:

                if object_label == 30:
                    pass
                if object_label < 250:
                    continue

                mask = (labels == object_label) * (instances == dataset_ins)

                frame_dynamic_instances.append(dataset_ins + object_label * 10000)

                dictionary_idx, new = i_handler.dataset2motion_index(dataset_ins + object_label * 10000, int(sequence))

                dynamic_instances = np.where(mask, dictionary_idx, dynamic_instances)

                if new:
                    dictionary[dictionary_idx] = [f'{sequence}_{frame}']
                else:
                    dictionary[dictionary_idx].append(f'{sequence}_{frame}')

        i_handler.active_check(frame_dynamic_instances)

        os.makedirs(f'{dataset_path}/sequences/{sequence}/instances_GT', exist_ok=True)
        save_labels = open(f'{dataset_path}/sequences/{sequence}/instances_GT/{frame}.label', 'wb')
        save_labels.write(dynamic_instances.astype(np.uint32))
        save_labels.close()

    os.makedirs('output/', exist_ok=True)
    with open("output/dynamic_instances.txt", "w") as fp:
        json.dump(dictionary, fp)
    print('Dictionary was saved.')
