import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import matplotlib.pyplot as plt
import utils


class TextDataset(Dataset):

    def __init__(self, file_names: list, image_dir: str, annotation_dir: str, train: bool = True, transforms = None):
        super().__init__()

        self.image_ids = file_names
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.train = train


    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.train:
            records = utils.annotation_filereader(f'{self.annotation_dir}/{image_id}.txt')
            target = utils.bounding_box(records).bbox_dict

            # boxes = records[['x', 'y', 'w', 'h']].values
            # boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            # boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            #
            # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # area = torch.as_tensor(area, dtype=torch.float32)
            #
            # # there is only one class
            # labels = torch.ones((records.shape[0],), dtype=torch.int64)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

            # target = {}
            # target['boxes'] = boxes
            # target['labels'] = labels
            # target['masks'] = None
            target['image_id'] = torch.tensor([index])
            # target['area'] = area
            target['iscrowd'] = iscrowd

        if self.transforms:
            if not self.train:
                sample = {
                    'image': image,
                }
            else:
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': target['labels']
                }
            # print(sample['image'].shape)
            # print(sample['image'])
            image1 = self.transforms(sample['image'])
            # image = sample['image']


            # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        if not self.train:
            return image1, image_id
        else:
            return image1, target, image_id

    def __len__(self) -> int:
        return len(self.image_ids)

    def plot_image(self,index: int,bbox=True):
        image_id = self.image_ids[index]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if bbox:
            records = utils.annotation_filereader(f'{self.annotation_dir}/{image_id}.txt')
            target = utils.bounding_box(records).bbox_dict
            for i in range(len(target['boxes'],)):
                cv2.rectangle(image, target['boxes'][i][0], target['boxes'][i][1], color=(0, 255, 0),
                              thickness=1)  # Draw Rectangle with the coordinates
                # cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
        plt.figure(figsize=(20, 30))  # display the output image
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.show()


