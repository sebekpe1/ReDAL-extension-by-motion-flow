import numpy as np
import glob
import yaml

if __name__ == '__main__':

    with open('../waymo.yaml') as f:
        waymo_config = yaml.load(f)

    data_path = 'Waymo/parsed_waymo'

    waymo_sequences = sorted(glob.glob(f'{data_path}/*/'))
    assert len(waymo_sequences) == 798, f'Number of waymo sequences is not equal to 798'

    classes_distribution = np.zeros((798, 19))
    number_of_files = np.zeros(798, )

    for idx, waymo_seq in enumerate(waymo_sequences):
        print(f'Sequence {waymo_seq.split("/")[-2]} --> {idx+1}/{len(waymo_sequences)} = {100*(idx+1)/len(waymo_sequences):.01f}%')

        velodyne_paths = sorted(glob.glob(f'{waymo_seq}velodyne/*.bin'))

        sequence_distribution = np.zeros((1, 19))

        number_of_files[idx] = len(velodyne_paths)

        for velodyne_path in velodyne_paths:
            seq = velodyne_path.split('/')[-3]
            frame = velodyne_path.split('/')[-1].split('.')[0]

            label_path = velodyne_path.replace('velodyne', 'labels').replace('.bin', '.label')

            all_labels = np.fromfile(label_path, dtype=np.int32).reshape(-1)
            labels_ = all_labels & 0xFFFF

            train_labels = np.vectorize(waymo_config['learning_map'].__getitem__)(labels_).astype(np.uint8)

            labels, counts = np.unique(train_labels, return_counts=True)

            for l, c in zip(labels, counts):
                sequence_distribution[0, l] += c

        classes_distribution[idx] = sequence_distribution

    np.save(f'sequences_class_distribution', classes_distribution)
    np.save(f'sequences_num_velodyne', number_of_files)
