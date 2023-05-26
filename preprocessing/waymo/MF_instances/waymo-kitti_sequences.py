import copy
import os
import numpy as np
import glob
import json


def create_mapping_dict(txt_path):
    mapping = dict()
    done = False
    txt_mapping = open(f'{txt_path}', 'r')
    _ = txt_mapping.readline()

    while not done:
        line = txt_mapping.readline()

        if len(line) == 0:
            done = True
            continue

        line = line.replace('\n', '')
        line_elements = line.split(' ')
        mapping[line_elements[0]] = line_elements[2]

    txt_mapping.close()
    return mapping


def convert_dynamic_dict(instances_dict, mapping):
    new_dict = copy.deepcopy(instances_dict)
    for key in instances_dict.keys():
        for supvox in instances_dict[key]:
            waymo_seq = supvox.split('#')[0].split('_')[0]
            waymo_frame = supvox.split('#')[0].split('_')[1]
            supvox_id = supvox.split('#')[1]
            kitti_position = mapping[f'{waymo_seq}/{waymo_frame}']
            kitti_seq = kitti_position.split('/')[0]
            kitti_frame = kitti_position.split('/')[1]
            new_dict[key].append(f'{kitti_seq}_{kitti_frame}#{int(float(supvox_id))}')
            new_dict[key].remove(supvox)
    return new_dict


def convert_scene_dict(scene_instances, mapping):
    new_dict = dict()
    for key in scene_instances.keys():
        waymo_seq = key.split('#')[0].split('_')[0]
        waymo_frame = key.split('#')[0].split('_')[1]
        kitti_position = mapping[f'{waymo_seq}/{waymo_frame}']
        kitti_seq = kitti_position.split('/')[0]
        kitti_frame = kitti_position.split('/')[1]
        new_dict[f'{kitti_seq}_{kitti_frame}'] = scene_instances[key]

    return new_dict


if __name__ == '__main__':

    supervoxel_supfix = '_MF'

    source_path = 'Waymo/parsed_waymo'
    destination_path = 'waymo/sequences'

    if os.path.exists(f'{destination_path}/mapping.json'):
        with open(f'{destination_path}/mapping.json', 'r') as f:
            mapping = json.load(f)

    else:
        mapping = create_mapping_dict(f'{destination_path}/mapping.txt')
        with open(f"{destination_path}/mapping.json", 'w') as f:
            json.dump(mapping, f)

    train_sequences = np.arange(798)

    data = np.load(f'dataset_split.npz', allow_pickle=True)
    val_sequences = data['val_sequences']

    train_sequences = np.delete(train_sequences, val_sequences)

    work = train_sequences.tolist()

    for seq in work:

        labels_paths = sorted(glob.glob(f'{source_path}/{seq:04d}/annotator_labels/*.label'))

        for i, label_path in enumerate(labels_paths):

            print(f'\r{i:02d}/{len(labels_paths)} --> {seq}/798 --> {100 * seq / 798:.01f}%', end='')

            waymo_frame = label_path.split('/')[-1].split('.')[0]

            kitti_position = mapping[f'{seq:04d}/{waymo_frame}']
            kitti_seq = kitti_position.split('/')[0]
            kitti_frame = kitti_position.split('/')[1]

            os.makedirs(f'{destination_path}/{kitti_seq}/labels{supervoxel_supfix}', exist_ok=True)
            os.system(f'cp {label_path} {destination_path}/{kitti_seq}/labels{supervoxel_supfix}/{kitti_frame}.label')
