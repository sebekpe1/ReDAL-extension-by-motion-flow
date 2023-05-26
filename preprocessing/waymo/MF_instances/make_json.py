import os
import numpy as np
import glob
import json

if __name__ == '__main__':

    with open('output/init_frames.json', 'r') as f:
        labeled_frames = json.load(f)

    data_path = 'waymo/sequences'
    sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    file_supfix = '_MF'

    supvox_pts = dict()
    labeled_regions = dict()
    unlabeled_regions = dict()

    for seq in sequences:

        velodyne_paths = sorted(glob.glob(f'{data_path}/{seq}/velodyne/*.bin'))

        for vel_path in velodyne_paths:

            frame = vel_path.split('/')[-1].split('.')[0]

            print(f'\r{seq}/{frame}', end='')

            supvox_file = vel_path.replace('velodyne', f'supervoxel{file_supfix}')

            supervoxels = np.fromfile(supvox_file, dtype=np.int32)

            supervoxel_idx, supervoxels_counts = np.unique(supervoxels, return_counts=True)
            supervoxel_idx = supervoxel_idx.tolist()
            supervoxels_counts = supervoxels_counts.tolist()

            if 0 in supervoxel_idx:
                indx = supervoxel_idx.index(0)
                supervoxel_idx.pop(indx)
                supervoxels_counts.pop(indx)

            if f'{seq}_{frame}' in labeled_frames.keys():
                labeled_regions[f'{seq}_{frame}'] = supervoxel_idx
            else:
                unlabeled_regions[f'{seq}_{frame}'] = supervoxel_idx

            for s_idx, s_count in zip(supervoxel_idx, supervoxels_counts):
                supvox_pts[f'{seq}/velodyne/{frame}.bin#{s_idx}'] = s_count

    with open(f"output/init_label_large_region{file_supfix}_wo_dyn_objects.json", 'w') as f:
        json.dump(labeled_regions, f)
    with open(f"output/init_ulabel_large_region{file_supfix}_wo_dyn_objects.json", 'w') as f:
        json.dump(unlabeled_regions, f)
    with open(f"output/waymo_large_pts{file_supfix}.json", 'w') as f:
        json.dump(supvox_pts, f)
