from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

custom_transforms = {
    'train' : transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(80),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

    'eval' : transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(80),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

class FogDataset(Dataset):
    def __init__(self, data_dict, transforms=None):

        self.transforms = transforms
        self.images = data_dict['images']
        self.targets = data_dict['targets']
        self.filepaths = data_dict['filepaths']
        self.meteo = data_dict['meteo']

    def __getitem__(self, index):
        image = self.images[index]

        if self.transforms != None:
            image = self.transforms(image)

        target = self.targets[index]
        filepath = self.filepaths[index]
        meteo = self.meteo[index]

        return (image, target, index, filepath, meteo)


    def __len__(self):
        return len(self.targets)

def create_loader(dataset, type, batch_size):
    '''
    Fetches the DataLoader object for each type in types from data_dir.

    :param dataset: numpy dict containing the data arrays
    :param type: type of dataset [train, validation or test]
    :param batch_size: mini-batch size to use in experiment
    :return: dataloader object
    '''

    if type == 'train':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif type == 'validation':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif type == 'test':
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    else:
        raise ValueError("Specify either 'train', 'validation' or 'test' to create dataloader, not '{}'".format(type)) 


    return dataloader
