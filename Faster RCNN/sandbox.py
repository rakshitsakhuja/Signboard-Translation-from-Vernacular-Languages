from collections import defaultdict
from os.path import basename

import numpy as np
import torch
import pandas as pd
from utils import side


def annotation_filereader(filepath):
    with open(filepath, "r") as file:  # Use file to refer to the file object
        line = file.readline()
        bbox_cnt = 1
        bbox = []
        bbox_dict = defaultdict()
        while line:
            # print("Line {}: {}".format(bbox_cnt, line.strip()))
            language = line.split("::")[-1].strip()
            # print(language)
            # print(line.split("::")[0])
            x1, x2, x3, x4, y1, y2, y3, y4, ground_truth = line.split("::")[0].split(",")
            print(x1, x2, x3, x4, y1, y2, y3, y4, ground_truth, language)
            bbox.append([basename(filepath), int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4), ground_truth, language])
            bbox_cnt += 1
            line = file.readline()
    return pd.DataFrame(bbox, columns=['filename', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'ground_truth',
                                       'language'])


bbox = annotation_filereader(
    "/home/rakshit/PycharmProjects/Signboard-Translation-from-Vernacular-Languages/Datasets/real_Image_dataset_Detection/Annotation/0.txt")
print(bbox)

#######Use case of Permute
# print('permute Use case')
# x = torch.randn(2, 3, 5)
# print(x.size())
# print(x)
# print(x.permute(2, 0, 1))
# print(x.permute(2, 0, 1).size())


filename, x1, x2, x3, x4, y1, y2, y3, y4, ground_truth, language = bbox['filename'], \
                                                                   bbox['x1'], bbox['x2'], bbox['x3'], \
                                                                   bbox['x4'], bbox['y1'], bbox['y2'], \
                                                                   bbox['y3'], bbox['y4'], \
                                                                   bbox['ground_truth'], bbox['language']

top_left = pd.concat([x1, y1], axis=1)
top_right = pd.concat([x2, y2], axis=1)
print(top_left.iloc[:,0])


#
# tl=bbox['x1'].str.cat(bbox['x2'], sep =" ")
# print(tl,tl[:,0].split()[0],tl[:,0].split()[1])
class bounding_box:

    def __init__(self, bbox):
        filename, x1, x2, x3, x4, y1, y2, y3, y4, ground_truth, language = bbox['filename'], \
                                                                           bbox['x1'], bbox['x2'], bbox['x3'], \
                                                                           bbox['x4'], bbox['y1'], bbox['y2'], \
                                                                           bbox['y3'], bbox['y4'], \
                                                                           bbox['ground_truth'], bbox['language']
        self.top_left = pd.concat([x1, y1], axis=1)
        self.top_right = pd.concat([x2, y2], axis=1)
        self.bottom_right = pd.concat([x3, y3], axis=1)
        self.bottom_left = pd.concat([x4, y4], axis=1)
        self.side1 = side(self.top_right, self.top_left)
        self.side2 = side(self.top_right, self.bottom_right)
        self.side3 = side(self.bottom_left, self.bottom_right)
        self.side4 = side(self.top_left, self.bottom_left)
        self.label = ground_truth
        self.height = pd.concat([self.side2, self.side4],axis=1).max(axis=1)
        self.width = pd.concat([self.side1, self.side3],axis=1).max(axis=1)
        self.corners = [self.top_left,
                        self.top_right,
                        self.bottom_right,
                        self.bottom_left]

    def perimeter(self):
        """
        returns perimeter of bbox
           """
        p = self.side1 + self.side2 + self.side3 + self.side4
        # print("perimeter=", p)
        return p

    def bbox_area(self):
        """
        returns area of bounding box
        """
        n = len(self.corners)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.corners[i].iloc[:,0] * self.corners[j].iloc[:,1]
            area -= self.corners[j].iloc[:,0] * self.corners[i].iloc[:,1]
        area = abs(area) / 2.0
        return area

    @property
    def bbox_dict(self):
        bb_dict = {}
        bb_dict['boxes'] = pd.concat([self.top_left,self.bottom_right],axis=1).values
        bb_dict['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*bb_dict['boxes'])))).permute(1, 0)
        labels = torch.ones((bb_dict['boxes'].shape[0],), dtype=torch.int64)
        area = torch.as_tensor(self.bbox_area().values, dtype=torch.float32)
        bb_dict['labels'] = labels
        bb_dict['area'] = area
        return bb_dict

# boxessss = list(zip(*[[122, 294, 295, 126, 207, 214, 277, 270, '## HINDI'],[122, 294, 295, 126, 207, 214, 277, 270, '## HINDI'],[122, 294, 295, 126, 207, 214, 277, 270, '## HINDI'],[122, 294, 295, 126, 207, 214, 277, 270, '## HINDI'],[122, 294, 295, 126, 207, 214, 277, 270, '## HINDI'],[122, 294, 295, 126, 207, 214, 277, 270, '## HINDI'],[122, 294, 295, 126, 207, 214, 277, 270, '## HINDI']]))
# boxessss
box = bounding_box(bbox)
print(box.bbox_area())
print(box.perimeter())
print(box.height)
print(box.width)
print(box.side1, box.side2, box.side3, box.side4)
print(bounding_box(bbox).bbox_dict)