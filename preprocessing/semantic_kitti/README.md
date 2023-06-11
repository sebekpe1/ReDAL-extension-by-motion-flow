# SemanticKITTI data preprocessing

At first lauch [GT_motion_flow.py](.GT_instances/GT_motion_flow.py) to create instance mask just for dynamic objects,
Secondly run [dynamic_supervoxels.py](.GT_supervoxels/dynamic_supervoxels.py) to adjust supervoxels, which were originaly created by VCCS and create links between supervoxels, which belongs to same dynamic object.
Finaly, run [add_init_dynamic_objects.py](.json_creating/add_init_dynamic_objects.py) to add to init labeled json (and remove to init unlabeled json) region, which wold be anotated automatically.
