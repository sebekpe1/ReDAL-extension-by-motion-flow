import numpy as np
from scipy.spatial.transform import Rotation as R
from tools.visualization import o3d_visualization


def cut_bounding_box(point_cloud, annotation, annotation_move=[0, 0, 0], annotation_label=None, return_mask=False):
    """
    Function, which cuts bounding box from point-cloud.
    :param point_cloud: numpy 2D array, original point-cloud
    :param annotation: dictionary, annotation of bounding box which should be cut out
    :param annotation_move: numpy 1D array, row translation vector between annotation and LiDAR
    :return: numpy 2D array, annotations point-cloud
    """
    xc = annotation['center']['x'] - annotation_move[0]
    yc = annotation['center']['y'] - annotation_move[1]
    zc = annotation['center']['z'] - annotation_move[2]
    z_rotation = annotation['heading']
    length = annotation['length'] + 0.2
    width = annotation['width'] + 0.2
    height = annotation['height'] + 0.2

    rot_matrix = np.array([[np.cos(z_rotation), -1 * np.sin(z_rotation), 0],
                                [np.sin(z_rotation), np.cos(z_rotation), 0],
                                [0, 0, 1]])

    mask_1 = (rot_matrix[0][0] * point_cloud[:, 0] + rot_matrix[1][0] * point_cloud[:, 1] + rot_matrix[2][
        0] * point_cloud[:,
             2] >
              rot_matrix[0][0] * (xc + rot_matrix[0][0] * length / 2) + rot_matrix[1][0] * (
                      yc + rot_matrix[1][0] * length / 2) + rot_matrix[2][0] *
              (zc + rot_matrix[2][0] * length / 2))  # FRONT
    mask_2 = (rot_matrix[0][0] * point_cloud[:, 0] + rot_matrix[1][0] * point_cloud[:, 1] + rot_matrix[2][
        0] * point_cloud[:, 2] <
              rot_matrix[0][0] * (xc - rot_matrix[0][0] * length / 2) + rot_matrix[1][0] * (
                      yc - rot_matrix[1][0] * length / 2) + rot_matrix[2][0] *
              (zc - rot_matrix[2][0] * length / 2))  # BACK
    mask_3 = (rot_matrix[0][1] * point_cloud[:, 0] + rot_matrix[1][1] * point_cloud[:, 1] + rot_matrix[2][
        1] * point_cloud[:, 2] >
              rot_matrix[0][1] * (xc + rot_matrix[0][1] * width / 2) + rot_matrix[1][1] * (
                      yc + rot_matrix[1][1] * width / 2) + rot_matrix[2][1] *
              (zc + rot_matrix[2][1] * width / 2))  # SIDE
    mask_4 = (rot_matrix[0][1] * point_cloud[:, 0] + rot_matrix[1][1] * point_cloud[:, 1] + rot_matrix[2][
        1] * point_cloud[:, 2] <
              rot_matrix[0][1] * (xc - rot_matrix[0][1] * width / 2) + rot_matrix[1][1] * (
                      yc - rot_matrix[1][1] * width / 2) + rot_matrix[2][1] *
              (zc - rot_matrix[2][1] * width / 2))  # SIDE
    mask_5 = (rot_matrix[0][2] * point_cloud[:, 0] + rot_matrix[1][2] * point_cloud[:, 1] + rot_matrix[2][
        2] * point_cloud[:, 2] >
              rot_matrix[0][2] * (xc + rot_matrix[0][2] * height / 2) + rot_matrix[1][2] * (
                      yc + rot_matrix[1][2] * height / 2) + rot_matrix[2][2] *
              (zc + rot_matrix[2][2] * height / 2))
    mask_6 = (rot_matrix[0][2] * point_cloud[:, 0] + rot_matrix[1][2] * point_cloud[:, 1] + rot_matrix[2][
        2] * point_cloud[:, 2] <
              rot_matrix[0][2] * (xc - rot_matrix[0][2] * height / 2) + rot_matrix[1][2] * (
                      yc - rot_matrix[1][2] * height / 2) + rot_matrix[2][2] *
              (zc - rot_matrix[2][2] * height / 2))

    position_mask = mask_1
    position_mask = np.ma.mask_or(position_mask, mask_2)
    position_mask = np.ma.mask_or(position_mask, mask_3)
    position_mask = np.ma.mask_or(position_mask, mask_4)
    position_mask = np.ma.mask_or(position_mask, mask_5)
    position_mask = np.ma.mask_or(position_mask, mask_6)

    if annotation_label is None:
        if annotation['class'] == 'VEHICLE':
            label_mask = np.isin(point_cloud[:, 4], [1, 2, 3, 4])
        elif annotation['class'] == 'PEDESTRIAN':
            label_mask = np.isin(point_cloud[:, 4], [7])
        elif annotation['class'] == 'CYCLIST':
            label_mask = np.isin(point_cloud[:, 4], [5, 6, 12, 13])
        else:
            label_mask = np.ones(point_cloud.shape[0], dtype=bool)
    else:
        label_mask = np.isin(point_cloud[:, 4], [annotation_label])

    final_mask = ~position_mask * label_mask

    if return_mask:
        if not final_mask.any():
            print(f'Position OK points = {np.count_nonzero(~position_mask)}, Label OK points = {np.count_nonzero(label_mask)}')
        return final_mask
    else:
        scene = point_cloud[~final_mask]
        bbox = point_cloud[final_mask]
        return scene, bbox


def separate_bbox(point_cloud, annotation, annotation_move=[0, 0, 0], visualization=False, annotation_label=None, return_mask=False):

    xc = annotation['center']['x'] - annotation_move[0]
    yc = annotation['center']['y'] - annotation_move[1]
    zc = annotation['center']['z'] - annotation_move[2]
    z_rotation = annotation['heading']
    length = annotation['length'] + 0.2
    width = annotation['width'] + 0.2
    height = annotation['height'] + 0.2

    rotation_matrix = np.array([[np.cos(z_rotation), -1 * np.sin(z_rotation), 0],
                                [np.sin(z_rotation), np.cos(z_rotation), 0],
                                [0, 0, 1]])

    r = R.from_matrix(rotation_matrix)
    rot_matrix = r.as_matrix()

    mask_1 = (rot_matrix[0][0] * point_cloud[:, 0] + rot_matrix[1][0] * point_cloud[:, 1] + rot_matrix[2][
            0] * point_cloud[:,
                 2] >
        rot_matrix[0][0] * (xc + rot_matrix[0][0] * length / 2) + rot_matrix[1][0] * (
                yc + rot_matrix[1][0] * length / 2) + rot_matrix[2][0] *
              (zc + rot_matrix[2][0] * length / 2))     # FRONT
    mask_2 = (rot_matrix[0][0] * point_cloud[:, 0] + rot_matrix[1][0] * point_cloud[:, 1] + rot_matrix[2][0] * point_cloud[:, 2] <
              rot_matrix[0][0] * (xc - rot_matrix[0][0] * length / 2) + rot_matrix[1][0] * (
                yc - rot_matrix[1][0] * length / 2) + rot_matrix[2][0] *
              (zc - rot_matrix[2][0] * length / 2))     # BACK
    mask_3 = (rot_matrix[0][1] * point_cloud[:, 0] + rot_matrix[1][1] * point_cloud[:, 1] + rot_matrix[2][1] * point_cloud[:, 2] >
              rot_matrix[0][1] * (xc + rot_matrix[0][1] * width / 2) + rot_matrix[1][1] * (
                yc + rot_matrix[1][1] * width / 2) + rot_matrix[2][1] *
              (zc + rot_matrix[2][1] * width / 2))      # SIDE
    mask_4 = (rot_matrix[0][1] * point_cloud[:, 0] + rot_matrix[1][1] * point_cloud[:, 1] + rot_matrix[2][1] * point_cloud[:, 2] <
              rot_matrix[0][1] * (xc - rot_matrix[0][1] * width / 2) + rot_matrix[1][1] * (
                yc - rot_matrix[1][1] * width / 2) + rot_matrix[2][1] *
              (zc - rot_matrix[2][1] * width / 2))      # SIDE
    mask_5 = (rot_matrix[0][2] * point_cloud[:, 0] + rot_matrix[1][2] * point_cloud[:, 1] + rot_matrix[2][2] * point_cloud[:, 2] >
              rot_matrix[0][2] * (xc + rot_matrix[0][2] * height / 2) + rot_matrix[1][2] * (
                yc + rot_matrix[1][2] * height / 2) + rot_matrix[2][2] *
              (zc + rot_matrix[2][2] * height / 2))
    mask_6 = (rot_matrix[0][2] * point_cloud[:, 0] + rot_matrix[1][2] * point_cloud[:, 1] + rot_matrix[2][2] * point_cloud[:, 2] <
              rot_matrix[0][2] * (xc - rot_matrix[0][2] * height / 2) + rot_matrix[1][2] * (
                yc - rot_matrix[1][2] * height / 2) + rot_matrix[2][2] *
              (zc - rot_matrix[2][2] * height / 2))

    if visualization:
        position_mask = mask_1

        scene = point_cloud[position_mask]
        bbox = point_cloud[position_mask == False]

        o3d_visualization(np.vstack((scene[:, :3], bbox[:, :3])),
                          np.append(np.zeros(scene.shape[0]), np.ones(bbox.shape[0])), bboxes=[annotation])

        position_mask = np.ma.mask_or(mask_1, mask_2)

        scene = point_cloud[position_mask]
        bbox = point_cloud[position_mask == False]

        o3d_visualization(np.vstack((scene[:, :3], bbox[:, :3])),
                          np.append(np.zeros(scene.shape[0]), np.ones(bbox.shape[0])), bboxes=[annotation])

        position_mask = np.ma.mask_or(position_mask, mask_3)
        scene = point_cloud[position_mask]
        bbox = point_cloud[position_mask == False]

        o3d_visualization(np.vstack((scene[:, :3], bbox[:, :3])),
                          np.append(np.zeros(scene.shape[0]), np.ones(bbox.shape[0])), bboxes=[annotation])

        position_mask = np.ma.mask_or(position_mask, mask_4)

        scene = point_cloud[position_mask]
        bbox = point_cloud[position_mask == False]

        o3d_visualization(np.vstack((scene[:, :3], bbox[:, :3])),
                          np.append(np.zeros(scene.shape[0]), np.ones(bbox.shape[0])), bboxes=[annotation])

        position_mask = np.ma.mask_or(position_mask, mask_5)

        scene = point_cloud[position_mask]
        bbox = point_cloud[position_mask == False]

        o3d_visualization(np.vstack((scene[:, :3], bbox[:, :3])),
                          np.append(np.zeros(scene.shape[0]), np.ones(bbox.shape[0])), bboxes=[annotation])

        position_mask = np.ma.mask_or(position_mask, mask_6)

        scene = point_cloud[position_mask]
        bbox = point_cloud[position_mask == False]

        o3d_visualization(np.vstack((scene[:, :3], bbox[:, :3])),
                          np.append(np.zeros(scene.shape[0]), np.ones(bbox.shape[0])), bboxes=[annotation])

    else:
        position_mask = mask_1
        position_mask = np.ma.mask_or(position_mask, mask_2)
        position_mask = np.ma.mask_or(position_mask, mask_3)
        position_mask = np.ma.mask_or(position_mask, mask_4)
        position_mask = np.ma.mask_or(position_mask, mask_5)
        position_mask = np.ma.mask_or(position_mask, mask_6)

    if annotation_label is None:
        if annotation['class'] == 'VEHICLE':
            label_mask = np.isin(point_cloud[:, 4], [1, 2, 3, 4])
        elif annotation['class'] == 'PEDESTRIAN':
            label_mask = np.isin(point_cloud[:, 4], [7])
        elif annotation['class'] == 'CYCLIST':
            label_mask = np.isin(point_cloud[:, 4], [5, 6, 12, 13])
        else:
            label_mask = np.ones(point_cloud.shape[0], dtype=bool)
    else:
        label_mask = np.isin(point_cloud[:, 4], [annotation_label])

    final_mask = ~position_mask * label_mask
    scene = point_cloud[~final_mask]
    bbox = point_cloud[final_mask]

    if return_mask:
        return final_mask
    else:
        return scene, bbox