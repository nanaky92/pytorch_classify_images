from torchvision import datasets, models, transforms
import torch


def get_image_datasets(data_dir):
    if data_dir[-1] != "/":
        data_dir = "{}/".format(data_dir)

    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(45),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "validation": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "validation": datasets.ImageFolder(valid_dir, transform=data_transforms["validation"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"])
    }

    return image_datasets

def get_dataloaders(image_datasets):
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=64, shuffle=True),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64, shuffle=True)
    }
    
    return dataloaders