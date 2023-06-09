# ReDAL extension by motion flow

This extension creates semanticaly pure regions for dynamic objects and link them through time, therefore it is possible to propagate annotations of these objects to entire sequence.

The full master thesis can be found at [CTU DSpace](https://dspace.cvut.cz/handle/10467/108586)

## Data Preparation

This repo supports SemanticKITTI and Waymo Open Dataset datasets.

At first create data preparation, which is provided in original implementation of the ReDAL method please see [this documentation](ReDAL-extension/w_pseudolabeling/data_preparation) in case of Waymo Open Dataset (which is not supported by authors of ReDAL) follow preparation steps of SemanticKITTI.

In next step follow instruction in README for [SemanticKITTI](preprocessing/semantic_kitti) or [Waymo Open Dataset](preprocessing/waymo)

## Neural network training

For training neural network please follow README for [training with pseudolabeling](ReDAL-extension/w_pseudolabeling) or [training without pseudolabeling](ReDAL-extension/wo_pseudolabeling)

## Annotator interface

Annotator interface is launched by [annotate_instance.py](annotator_gui/annotate_instance.py).
