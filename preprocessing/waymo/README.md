# Waymo Open Dataset data preprocessing

## Dataset split

For finding split of sequences to training and validating you can use [random_waymo_sequences.py](create_split/random_waymo_sequences.py) or use [our split](create_split/best_split.npz), which we find by meantioned algorithm.

## Data preparation based on ground truth

- launch [static_dynamic.py](GT_instances/static_dynamic.py) to determine if bounding box belong to static or dynamic object.
- launch [create_instances.py](GT_instances/create_instances.py) to create instances mask for dynamic objects.
- run [adjust_supervoxels.py](GT_instances/adjust_supervoxels.py) to adjust supervoxels, which were originaly created by VCCS and link them.
- run [make_json.py](GT_instances/make_json.py) and then [add_init_dynamic_objects.py](GT_instances/add_init_dynamic_objects.py) to create init jsons.

## Data preparation based on motion flow

- create DBSCANs of point clouds by [clustering.py](MF_instances/clustering.py)
- run [MF_instances.py](MF_instances/MF_instances.py) and then [MF_instances_backwards.py](MF_instances/MF_instances_backwards.py) for creating instances mask, which are conected by whole sequence.
- launch [MF_dictionary_anotator.py](MF_instances/MF_dictionary_anotator.py) to simulare human anotator on linked regions.
- launch [MF_adjust_supervoxels.py](MF_instances/MF_adjust_supervoxels.py) to adjust supervocels, which were originaly created by VCCS and link them.
- run [make_json.py](MF_instances/make_json.py) and then [add_init_dynamic_objects.py](MF_instances/add_init_dynamic_objects.py) to create init jsons.
