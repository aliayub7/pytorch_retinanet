import torch

from pytorch_retinanet.model import retinanet_dataset

class ListDataset(retinanet_dataset.ListDataset):
    def __getitem__(self, idx):
        """Return a COCO-formated bounding box target"""

        img, boxes, labels = super().__getitem__(idx)

        num_objs = len(boxes)
        boxes = boxes.float()
        image_id = torch.tensor([idx])
        if num_objs > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor([])

        # Perform checks on bounding boxes
        for xmin, ymin, xmax, ymax in boxes:
            assert (xmax - xmin > 5.0), 'xmax is less than xmin'
            assert (ymax - ymin > 5.0), 'ymax is less than ymin'

        # Suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        return img, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))
