import os
import glob
import numpy as np

if __name__ == '__main__':
    source_path = ''
    destination_path = 'Waymo/parsed_waymo'

    train_sequences = np.arange(798)

    data = np.load(f'dataset_split.npz', allow_pickle=True)
    val_sequences = data['val_sequences']

    train_sequences = np.delete(train_sequences, val_sequences)

    work = train_sequences.tolist()

    for seq in work:
        print(seq)
        if os.path.exists(f'{destination_path}/{seq:04d}/MF_instances_back/'):
            continue
        else:
            os.makedirs(f'{destination_path}/{seq:04d}/MF_instances_back/', exist_ok=True)
            MF_instances_paths = sorted(glob.glob(f'{source_path}/{seq:04d}/MF_instances_back/*_000.npy'))

            for instance_path in MF_instances_paths:
                frame = instance_path.split('/')[-1].split('_')[0]

                os.system(f'cp {instance_path} {destination_path}/{seq:04d}/MF_instances_back/{frame}.npy')