import torch

from pytorch_retinanet.model import retinanet_dataset
import pytorch_retinanet.config.fasterrcnn as config

class ListDataset(retinanet_dataset.ListDataset):
    def __init__(self,
                 img_dir=config.img_dir,
                 list_filename=config.train_list_filename,
                 label_map_filename=config.label_map_filename,
                 input_size=config.img_res,
                 **kwargs):
        super().__init__(
            img_dir=img_dir,
            list_filename=list_filename,
            label_map_filename=label_map_filename,
            input_size=input_size,
            **kwargs
       )

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
            # We add this margin to capture near-zero widths/heights.
            xmsg = '{} (xmax) <= {} (xmin)'.format(xmax, xmin)
            assert (xmax - xmin > 0.1), xmsg
            ymsg = '{} (ymax) <= {} (ymin)'.format(ymax, ymin)
            assert (ymax - ymin > 0.1), ymsg

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
