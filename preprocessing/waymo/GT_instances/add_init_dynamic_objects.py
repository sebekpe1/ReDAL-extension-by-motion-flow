import copy
import os
import json


if __name__ == '__main__':
    file_supfix = '_GT'

    print(os.path.exists(f'output/init_label_large_region{file_supfix}_wo_dyn_objects.json'))
    with open(f'output/init_label_large_region{file_supfix}_wo_dyn_objects.json', 'r') as f:
        labeled_regions = json.load(f)

    with open(f'output/init_ulabel_large_region{file_supfix}_wo_dyn_objects.json', 'r') as f:
        unlabeled_regions = json.load(f)

    with open(f'../waymo-kitti_sequences/output/dynamic_instances{file_supfix}.json', 'r') as f:
        dynamic_instances_dict = json.load(f)
    with open(f'../waymo-kitti_sequences/output/scene_instances{file_supfix}.json', 'r') as f:
        scene_instances_dict = json.load(f)

    new_labeled_regions = copy.deepcopy(labeled_regions)
    new_ulabeled_regions = copy.deepcopy(unlabeled_regions)
    for key in labeled_regions.keys():
        if key in scene_instances_dict.keys():
            list_of_dynamic_instances = scene_instances_dict[key]
            for d_i in list_of_dynamic_instances:
                for supvox in dynamic_instances_dict[d_i]:
                    frame = supvox.split('#')[0]
                    supvox_id = int(supvox.split('#')[1])
                    if frame in new_labeled_regions.keys():
                        new_labeled_regions[frame].append(supvox_id)
                    else:
                        new_labeled_regions[frame] = [supvox_id]
                    if frame in new_ulabeled_regions.keys():
                        new_ulabeled_regions[frame].remove(supvox_id)
                        if len(new_ulabeled_regions[frame]) == 0:
                            new_ulabeled_regions.pop(frame)

    with open(f"output/init_label_large_region{file_supfix}.json", 'w') as f:
        json.dump(new_labeled_regions, f)
    with open(f"output/init_ulabel_large_region{file_supfix}.json", 'w') as f:
        json.dump(new_ulabeled_regions, f)
    with open(f"output/dynamic_instances{file_supfix}.json", 'w') as f:
        json.dump(dynamic_instances_dict, f)
    with open(f"output/scene_instances{file_supfix}.json", 'w') as f:
        json.dump(scene_instances_dict, f)
