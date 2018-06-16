from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
    '''
    Custom dataset class used for creating the KNMI/highway datasets.

    '''
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
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    if type == 'train':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif type == 'eval':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    return dataloader
