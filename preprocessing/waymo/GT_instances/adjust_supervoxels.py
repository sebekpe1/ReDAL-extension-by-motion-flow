import glob
import numpy as np
import os
import json


def remove_missing_supvox(supervoxels, added_supvox_ids):
    while not np.max(supervoxels) == len(np.unique(supervoxels)) - 1:
        for i in range(np.max(supervoxels)):
            if i not in np.unique(supervoxels):
                supervoxels = np.where(supervoxels > i, supervoxels - 1, supervoxels)
                added_supvox_ids = np.where(added_supvox_ids > i, added_supvox_ids - 1, added_supvox_ids)
                break
    assert (np.unique(supervoxels) == np.arange(np.max(supervoxels)+1)).all(), f'Something goes wrong in function' \
                                                                               f'remove_missing_supvox'
    return supervoxels, added_supvox_ids


if __name__ == '__main__':

    with open("output/dynamic_instances_waymo_GT.txt", "r") as f:
        dynamic_instances_dict = json.load(f)

    single_instances = []
    for key in dynamic_instances_dict.keys():
        if len(dynamic_instances_dict[key]) == 1:
            single_instances.append(key)
    for key in single_instances:
        dynamic_instances_dict.pop(key)

    os.makedirs('output/', exist_ok=True)

    scene_instances_dict = dict()
    for key in dynamic_instances_dict.keys():
        for frame in dynamic_instances_dict[key]:
            if frame in scene_instances_dict:
                scene_instances_dict[frame].append(key)
            else:
                scene_instances_dict[frame] = [key]

    dataset_path = 'Waymo/parsed_waymo'

    for seq_idx in range(798):

        print(f'\r{seq_idx}/797 --> {100 * seq_idx/797:.01f}%', end='')

        instances_paths = sorted(glob.glob(f'{dataset_path}/{seq_idx:04d}/instances_GT/*.label'))

        os.makedirs(f'{dataset_path}/{seq_idx:04d}/supervoxel_GT/', exist_ok=True)

        for instances_path in instances_paths:

            frame = instances_path.split('/')[-1].split('.')[0]

            dynamic_instances = np.fromfile(instances_path, dtype=np.int32)
            mask = np.isin(dynamic_instances, single_instances)
            dynamic_instances = np.where(mask, 0, dynamic_instances)

            supervoxel_path = instances_path.replace('instances_GT', 'supervoxel').replace('.label', '.bin')
            supervoxels = np.fromfile(supervoxel_path, dtype=np.int32)

            if len(np.unique(dynamic_instances)) == 1:
                assert f'{seq_idx:04d}_{frame}' not in scene_instances_dict, f'nothing in dynamic instances ' \
                                                                             f'but not in dictionary'

                save_supervoxel = open(f'{dataset_path}/{seq_idx:04d}/supervoxel_GT/{frame}.bin', 'wb')
                save_supervoxel.write(supervoxels.astype(np.uint32))
                save_supervoxel.close()
                continue

            assert len(np.unique(dynamic_instances)) - 1 == len(scene_instances_dict[f'{seq_idx:04d}_{frame}']), \
                f'different number of dynamic instances in mask and dictionary'

            added_instances = []
            added_instances_supvox_id = np.array([])
            for inst in scene_instances_dict[f'{seq_idx:04d}_{frame}']:
                assert int(inst) in np.unique(dynamic_instances), f'dictionary instance is not in dynamic instance mask'

                new_supervoxel_index = np.max(supervoxels) + 1
                supervoxels = np.where(dynamic_instances == int(inst), new_supervoxel_index, supervoxels)

                added_instances.append(inst)
                added_instances_supvox_id = np.append(added_instances_supvox_id, new_supervoxel_index)

            supervoxels, added_instances_supvox_id = remove_missing_supvox(supervoxels, added_instances_supvox_id)

            added_instances_supvox_id = added_instances_supvox_id.tolist()
            for inst, supvox_id in zip(added_instances, added_instances_supvox_id):
                dynamic_instances_dict[inst].remove(f'{seq_idx:04d}_{frame}')
                dynamic_instances_dict[inst].append(f'{seq_idx:04d}_{frame}#{supvox_id}')

            save_supervoxel = open(f'{dataset_path}/{seq_idx:04d}/supervoxel_GT/{frame}.bin', 'wb')
            save_supervoxel.write(supervoxels.astype(np.uint32))
            save_supervoxel.close()

    with open('output/dynamic_instances_GT.txt', 'w') as f:
        json.dump(dynamic_instances_dict, f)
    print(f'Dictionary with dynamic instances and their supervoxels (dynamic_instances) was saved')

    with open('output/scene_instances_GT.txt', 'w') as f:
        json.dump(scene_instances_dict, f)
    print(f'Dictionary with scenes and their dynamic instances (scene_instances) was saved')
