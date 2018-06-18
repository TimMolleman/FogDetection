import argparse
import numpy as np
from torchvision import models
from models.data_loader import *
from models.model_loading import get_model
from train.train_utils import *

parser = argparse.ArgumentParser(description='Pytorch ML')
# Learning
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate [default: 0.0001]')
parser.add_argument('--epochs', type=int, default=200, help='default number of epochs for the training [default 100]')
parser.add_argument('--batch_size', type=int, default=164, help='batch size for training [default: 164]')
parser.add_argument('--from_conv_layer', type=int, default=False, help='train convolutional layers or only fc layer [default: False]')
parser.add_argument('--include_meteo', type=int, default=False, help='include meteo in model training or not [default: False]')

# Model
parser.add_argument('--model_name', nargs="?", type=str, default='resnet18', help="Form of model, i.e resnet18, simple_CNN, etc.")
parser.add_argument('--num_classes', type=int, default=3, help='number of classes to predict [default: 3]')
parser.add_argument('--meteo_inputs', type=int, default=4, help='number of meteorological variables included [default: 4]')
parser.add_argument('--meteo_hidden_size', type=int, default=15, help='size of hidden layer for meteo net [default: 15]')
parser.add_argument('--meteo_outputs', type=int, default=3, help='number of outputs for meteo NN [default: 3')
#Device
parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
parser.add_argument('--save_path', type=str, default="/models/trained.pth.tar", help='Path where to dump model')
# Data locations
parser.add_argument('--train_data_path', type=str, default="data/processed/highway_train.npy", help='location of training numpy dictionary')
parser.add_argument('--val_data_path', type=str, default="data/processed/highway_val.npy", help='location of validation numpy array dictionary')
parser.add_argument('--test_data_path', type=str, default="data/processed/test.npy", help='location of test numpy array dictionary')

args = parser.parse_args()

if __name__ == '__main__':

	# Load dictionaries containing numpy arrays
	train_data, val_data = np.load(args.train_data_path)[()], np.load(args.val_data_path)[()]

	train_data['targets'][0:10] = 2
	train_data['targets'] = train_data['targets'][:300]
	train_data['images'] = train_data['images'][:300]
	train_data['meteo'] = train_data['meteo'][:300]
	train_data['filepaths'] = train_data['filepaths'][:300]

	val_data['targets'][0:10] = 2
	val_data['targets'] = val_data['targets'][:300]
	val_data['images'] = val_data['images'][:300]
	val_data['meteo'] = val_data['meteo'][:300]
	val_data['filepaths'] = val_data['filepaths'][:300]

	# Get loss weights
	loss_weights = calculate_loss_weights(train_data['targets'])

	# Create datasets
	train_dataset = FogDataset(train_data, custom_transforms['train'])
	validation_dataset = FogDataset(val_data, custom_transforms['eval'])

	# Create dataloaders and put in dictionary
	train_loader = create_loader(train_dataset, 'train', args.batch_size)
	validation_loader = create_loader(validation_dataset, 'eval', args.batch_size)

	# Loader dict
	loaders = {'train' : train_loader, 'validation': validation_loader}

	# Obtain model
	model = get_model(args)

	train_model = train_model(model, loaders, loss_weights, args)
	


	