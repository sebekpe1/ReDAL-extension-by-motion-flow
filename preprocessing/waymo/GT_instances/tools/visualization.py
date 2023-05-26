import numpy as np
import open3d as o3d
import struct


def o3d_visualization(pcl, labels, colors=True, config=None, name=f'point_cloud'):

    assert len(pcl.shape) == 2 and len(labels.shape) == 2 and pcl.shape[1] == 3

    xyz = pcl[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors:
        rgb = np.ones((len(pcl), 3))
        for i in range(len(pcl)):
            rgb[i, :] = config['color_map'][int(labels[i, 0])]
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(f"{name}.ply", pcd)
    cloud = o3d.io.read_point_cloud(f"{name}.ply")  # Read the point cloud
    vis = o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud


def ply_visualization(pcl, labels, config=None, file_name='point_cloud'):

    assert len(pcl.shape) == 2 and len(labels.shape) == 1 and pcl.shape[1] == 3, f'shape of pcl, or labels are not correct'

    assert pcl.shape[0] == labels.shape[0], f'num of points does not match with num of labels'

    rgb = np.ones(pcl.shape)

    if config is None:
        rgb[:, 2] = 255
    else:
        for i in range(len(labels)):
            rgb[i, :] = config['color_map'][labels[i]]

    rgb = rgb.astype(np.uint8)

    fid = open(f'{file_name}.ply', 'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n' % pcl.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(pcl.shape[0]):
        fid.write(bytearray(struct.pack("fffccc", pcl[i, 0], pcl[i, 1], pcl[i, 2],
                                        rgb[i, 2].tostring(), rgb[i, 1].tostring(),
                                        rgb[i, 0].tostring())))
    fid.close()
