import glob
import os
import numpy as np


def get_ID_global_position(anno, pose):
    position = np.array([anno['center']['x'], anno['center']['y'], anno['center']['z'], 1]).T
    global_position = (pose @ position).T
    ID = anno['ID']
    return ID, global_position


if __name__ == "__main__":
    data_path = 'Waymo/parsed_waymo'

    classes = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

    for seq in range(0, 798):

        print(f'\r{100*seq/798:.1f}%; {seq:3d}/798', end="")

        ID_pos_array = np.array([])

        annotations_paths = glob.glob(f'{data_path}/{seq:04d}/bboxes/*.npz')
        annotations_paths.sort()
        os.makedirs(f'{data_path}/{seq:04d}/bboxes_motion', exist_ok=True)

        if len(annotations_paths) == 0:
            continue

        for anno_path in annotations_paths:
            pose_path = anno_path.replace('bboxes', 'poses').replace('.npz', '.npy')

            anno = np.load(anno_path, allow_pickle=True)
            anno = anno['bboxes']

            pose = np.load(pose_path).reshape(4, 4)

            for a in anno:
                ID, global_position = get_ID_global_position(a, pose)

                idx = np.where(ID_pos_array == ID)

                assert len(idx[0]) < 2, f'Multiple finds of same ID in list'

                if len(idx[0]) == 0:
                    if len(ID_pos_array) == 0:
                        ID_pos_array = np.array([[ID, global_position, True]])
                    else:
                        ID_pos_array = np.vstack((ID_pos_array, np.array([ID, global_position, True])))
                else:
                    idx = idx[0][0]
                    assert type(ID_pos_array[idx][1]) == type(np.array([])), f'Error in type of position list {ID_pos_array[idx][1]}'
                    assert type(global_position) == type(np.array([])), f'Error in type of global position {global_position}'
                    if np.linalg.norm((ID_pos_array[idx][1]-global_position)[:2]) > 0.5:
                        ID_pos_array[idx][2] = False

        for anno_path in annotations_paths:
            anno = np.load(anno_path, allow_pickle=True)
            anno = anno['bboxes']
            for a in anno:
                idx = np.where(ID_pos_array == a['ID'])[0][0]
                a['class'] = classes[a['class']]
                a['static'] = ID_pos_array[idx][2]

            anno_path = anno_path.replace('bboxes', 'bboxes_motion')
            np.savez(anno_path, bboxes=anno)



