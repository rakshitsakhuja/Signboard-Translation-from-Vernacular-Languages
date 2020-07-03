import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split

from Dataset import TextDataset
from config import parameters
from tranformation import get_transform


def collate_fn(batch):
    return tuple(zip(*batch))


# file_names = [basename(i) for i in glob(parameters['image_dir'])]
def get_custom_dataset(file_names, frac):
    complete_dataset = TextDataset(file_names,
                                   parameters['image_dir'],
                                   parameters['annotation_dir'],
                                   train=True,
                                   transforms = get_transform())
    print(len(complete_dataset))

    train_size = int(frac * len(complete_dataset))
    test_size = len(complete_dataset) - train_size
    train_data, validation_data = random_split(complete_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=parameters['batch_len'],
                                               pin_memory=True,
                                               num_workers=parameters['num_workers'],
                                               shuffle=True,
                                               collate_fn=collate_fn)

    validation_loader = torch.utils.data.DataLoader(validation_data,
                                                    batch_size=parameters['batch_len'],
                                                    pin_memory=True,
                                                    num_workers=parameters['num_workers'],
                                                    shuffle=True,
                                                    collate_fn=collate_fn)
    return train_loader, validation_loader

# indices = torch.randperm(len(train_dataset)).tolist()
#
# train_data_loader = DataLoader(
#     train_dataset,
#     batch_size=8,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=collate_fn
# )
#
# valid_data_loader = DataLoader(
#     valid_dataset,
#     batch_size=8,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=collate_fn
# )
