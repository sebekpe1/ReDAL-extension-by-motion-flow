from typing import Any, Dict
import numpy as np
import torch
import torch.distributed as dist


class ConfusionMatrix():
    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou',
                 distributed: bool = True) -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.distributed = distributed

    def _before_epoch(self) -> None:
        self.cm = np.zeros((self.num_classes, self.num_classes))

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]
        if type(outputs) != np.ndarray:
            for i_tar in range(self.num_classes):
                for i_out in range(self.num_classes):
                    self.cm[i_tar, i_out] += torch.sum((targets == i_tar) & (outputs == i_out)).item()
        else:
            for i_tar in range(self.num_classes):
                for i_out in range(self.num_classes):
                    self.cm[i_tar, i_out] += np.sum((targets == i_tar) & (outputs == i_out))

    def _after_epoch(self):
        return self.cm
