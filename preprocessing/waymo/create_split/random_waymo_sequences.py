import os.path
import numpy as np


if __name__ == '__main__':
    sequences_class_points_num = np.load('sequences_class_distribution.npy')
    print(f'Total number of points = {np.sum(sequences_class_points_num)}')
    sequences_num_velodyne = np.load('sequences_num_velodyne.npy')

    desire_val_velodyne = 4150

    if os.path.exists(f'best_split.npz'):
        data = np.load(f'best_split.npz', allow_pickle=True)
        best_val_sequence = data['val_sequences']
        best_score = data['score'].item()
    else:
        best_val_sequence = np.array([])
        best_score = np.infty

    print(f'Starting best score is {best_score}')

    sequence_idx = np.arange(798)

    max_iters = 2000000

    for it in range(max_iters):
        print(f'\r{it}/{max_iters} --> {100*it/max_iters:.01f}%', end='')
        np.random.shuffle(sequence_idx)

        current_val_velodyne = 0
        current_sequences = np.array([])
        last_index = 0
        for i, idx in enumerate(sequence_idx):
            if current_val_velodyne >= desire_val_velodyne:
                last_index = i
                break
            else:
                current_val_velodyne += sequences_num_velodyne[idx]
                current_sequences = np.append(current_sequences, idx)

        current_sequences_points = sequences_class_points_num[sequence_idx[:last_index].astype(int), :]
        current_sequences_distribution = np.sum(current_sequences_points, axis=0) / np.sum(current_sequences_points)
        train_sequences_points = sequences_class_points_num[sequence_idx[last_index:].astype(int), :]
        train_sequences_distribution = np.sum(train_sequences_points, axis=0) / np.sum(train_sequences_points)
        devision = current_sequences_distribution / train_sequences_distribution
        score = np.sum(np.abs(np.log(devision)))

        if score < best_score:
            print(f'\rNew best score: {score}')
            best_score = score
            best_val_sequence = current_sequences.astype(int)
            np.savez('best_split_new', score=best_score, val_sequences=best_val_sequence)


        