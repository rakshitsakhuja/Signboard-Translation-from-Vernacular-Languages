import torchvision
import torchvision.transforms as T


def get_transform(train=True):
    custom_transforms = []

    if train:

        # custom_transforms.append(torchvision.transforms.RandomHorizontalFlip(.5))
        custom_transforms.append(torchvision.transforms.ToTensor())
        print('Transformation Done')
    else:

        custom_transforms.append(torchvision.transforms.ToTensor())

    return torchvision.transforms.Compose(custom_transforms)



