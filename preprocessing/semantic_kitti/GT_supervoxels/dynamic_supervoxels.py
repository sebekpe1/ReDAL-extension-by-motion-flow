import copy
import numpy as np
import os
import json

from tools.kitti_dataloader import KittiLoader


if __name__ == '__main__':

    with open("../instances_GT/output/dynamic_instances.txt", "r") as f:
        dynamic_instances_dict = json.load(f)

    single_instances = []
    for key in dynamic_instances_dict.keys():
        if len(dynamic_instances_dict[key]) == 1:
            single_instances.append(key)
    for key in single_instances:
        dynamic_instances_dict.pop(key)

    scene_instances_dict = dict()
    for key in dynamic_instances_dict.keys():
        for frame in dynamic_instances_dict[key]:
            if frame in scene_instances_dict:
                scene_instances_dict[frame].append(key)
            else:
                scene_instances_dict[frame] = [key]

    with open("../../ReDAL-main/dataloader/semantic_kitti/init_data/semkitti_large_pts.json", 'r') as f:
        supvox_pts = json.load(f)

    with open('../../ReDAL-main/dataloader/semantic_kitti/init_data/init_label_large_region.json') as f:
        labeled_regions = json.load(f)

    with open('../../ReDAL-main/dataloader/semantic_kitti/init_data/init_ulabel_large_region.json') as f:
        unlabeled_regions = json.load(f)

    dataset_path = 'SemanticKitti/dataset'

    dataloader = KittiLoader(dataset_path)

    print(f'Number of files in dataset {len(dataloader)}')

    for idx, (pcl, labels, instances, _, sequence, frame, supervoxels) in enumerate(dataloader):

        print(f'\r{sequence}_{frame} --> {idx}/{len(dataloader)} --> {100 * idx/len(dataloader):.01f}%', end='')

        dynamic_instances = np.fromfile(f'{dataset_path}/sequences/{sequence}/instances_GT/{frame}.label', dtype=np.int32)
        mask = np.isin(dynamic_instances, single_instances)
        dynamic_instances = np.where(mask, 0, dynamic_instances)

        new_supervoxels = copy.deepcopy(supervoxels)

        if len(np.unique(dynamic_instances)) == 1:
            assert not f'{sequence}_{frame}' in scene_instances_dict, f'nothing in dynamic instances ' \
                                                                      f'but not in dictionary'

            os.makedirs(f'{dataset_path}/sequences/{sequence}/supervoxel_GT/', exist_ok=True)
            save_supervoxel = open(f'{dataset_path}/sequences/{sequence}/supervoxel_GT/{frame}.bin', 'wb')
            save_supervoxel.write(new_supervoxels.astype(np.uint32))
            save_supervoxel.close()
            continue

        assert len(np.unique(dynamic_instances)) - 1 == len(scene_instances_dict[f'{sequence}_{frame}']), f'different number of dynamic instances in mask and dictionary'

        labeled_section = False
        if f'{sequence}_{frame}' in labeled_regions:
            supervoxel_idx_list = labeled_regions[f'{sequence}_{frame}']
            labeled_section = True

        elif f'{sequence}_{frame}' in unlabeled_regions:
            supervoxel_idx_list = unlabeled_regions[f'{sequence}_{frame}']

        else:
            assert False, 'Frame is not in any init json'

        assert np.max(supervoxels) == np.max(supervoxel_idx_list), f'number of supervoxels does not match ' \
                                                                   f'in json vs supervoxel file'

        for i in supervoxel_idx_list:
            supposed_n_points = supvox_pts[f'{sequence}/velodyne/{frame}.bin#{i}']
            real_n_points = len(supervoxels[supervoxels == i])
            assert supposed_n_points == real_n_points, f'different number of points in ' \
                                                       f'supervoxel json vs supervoxel file'

        for inst in scene_instances_dict[f'{sequence}_{frame}']:
            assert int(inst) in np.unique(dynamic_instances), f'dictionary instance is not in dynamic instance mask'

            if labeled_section:
                new_supervoxel_index = max(labeled_regions[f'{sequence}_{frame}']) + 1
                labeled_regions[f'{sequence}_{frame}'].append(new_supervoxel_index)
            else:
                new_supervoxel_index = max(unlabeled_regions[f'{sequence}_{frame}']) + 1
                unlabeled_regions[f'{sequence}_{frame}'].append(new_supervoxel_index)

            supvox_pts[f'{sequence}/velodyne/{frame}.bin#{new_supervoxel_index}'] = 0

            new_supervoxels = np.where(dynamic_instances == int(inst), new_supervoxel_index, new_supervoxels)

            dynamic_instances_dict[inst].remove(f'{sequence}_{frame}')
            dynamic_instances_dict[inst].append(f'{sequence}_{frame}#{new_supervoxel_index}')

        if labeled_section:
            supervoxel_idx_list = labeled_regions[f'{sequence}_{frame}']

        else:
            supervoxel_idx_list = unlabeled_regions[f'{sequence}_{frame}']

        for i in supervoxel_idx_list:
            num_points = len(new_supervoxels[new_supervoxels == i])
            if num_points == 0:
                assert f'{sequence}/velodyne/{frame}.bin#{i}' in supvox_pts, f'Supervoxel[{sequence}/velodyne/{frame}.bin#{i}] not in supvox_pts'
                supvox_pts.pop(f'{sequence}/velodyne/{frame}.bin#{i}')
                if labeled_section:
                    assert i in labeled_regions[f'{sequence}_{frame}'], f'Supervoxel {i} not in labeled_regions[{sequence}_{frame}]'
                    labeled_regions[f'{sequence}_{frame}'].remove(i)
                else:
                    assert i in unlabeled_regions[f'{sequence}_{frame}'], f'Supervoxel {i} not in unlabeled_regions[{sequence}_{frame}]'
                    unlabeled_regions[f'{sequence}_{frame}'].remove(i)
            else:
                supvox_pts[f'{sequence}/velodyne/{frame}.bin#{i}'] = num_points

        os.makedirs(f'{dataset_path}/sequences/{sequence}/supervoxel_GT/', exist_ok=True)
        save_supervoxel = open(f'{dataset_path}/sequences/{sequence}/supervoxel_GT/{frame}.bin', 'wb')
        save_supervoxel.write(new_supervoxels.astype(np.uint32))
        save_supervoxel.close()

    labeled_regions_copy = copy.deepcopy(labeled_regions)
    for key in labeled_regions_copy.keys():
        if key in scene_instances_dict:
            dynamic_objects = scene_instances_dict[key]

            for dyn_obj in dynamic_objects:
                check = False
                for dyn_obj_supvox in dynamic_instances_dict[dyn_obj]:
                    frame = dyn_obj_supvox.split('#')[0]
                    supvox = int(dyn_obj_supvox.split('#')[1])
                    if frame in labeled_regions and supvox in labeled_regions[frame]:
                        check = True
                    else:
                        if frame not in labeled_regions:
                            labeled_regions[frame] = [supvox]
                        else:
                            labeled_regions[frame].append(supvox)

                        assert frame in unlabeled_regions, f'frame is not in unlabeled regions'
                        assert supvox in unlabeled_regions[frame], f'supvox in not in unlabeled frame'

                        unlabeled_regions[frame].remove(supvox)
                        if len(unlabeled_regions[frame]) == 0:
                            unlabeled_regions.pop(frame)

                assert check, f'any dynamic supvoxel is not in labeled regions'
    os.makedirs('output/', exist_ok=True)

    with open("output/semkitti_large_pts_GT.json", 'w') as f:
        json.dump(supvox_pts, f)
    print(f'\rDictionary with number of points (semkitti_large_pts) was saved')

    with open('output/init_label_large_region_GT_wo_dyn_objects.json', 'w') as f:
        json.dump(labeled_regions, f)
    print(f'Dictionary with initially labeled supervoxels (init_label_large_region) was saved')

    with open('output/init_ulabel_large_region_GT_wo_dyn_objects.json', 'w') as f:
        json.dump(unlabeled_regions, f)
    print(f'Dictionary with initially unlabeled supervoxels (init_ulabel_large_region) was saved')

    with open('output/dynamic_instances_GT.txt', 'w') as f:
        json.dump(dynamic_instances_dict, f)
    print(f'Dictionary with dynamic instances and their supervoxels (dynamic_instances) was saved')

    with open('output/scene_instances_GT.txt', 'w') as f:
        json.dump(scene_instances_dict, f)
    print(f'Dictionary with scenes and their dynamic instances (scene_instances) was saved')


