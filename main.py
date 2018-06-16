import argparse
import numpy as np
from torchvision import models
from models.data_loader import *
from train.train_utils import *

parser = argparse.ArgumentParser(description='Pytorch ML')
# Learning
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate [default: 0.0001]')
parser.add_argument('--epochs', type=int, default=200, help='default number of epochs for the training [default 256]')
parser.add_argument('--batch_size', type=int, default=164, help='batch size for training [default: 64]')
parser.add_argument('--from_conv_layer', type=int, default=False, help='train convolutional layers or only fc layer [default: False]')
# Model
parser.add_argument('--model_name', nargs="?", type=str, default='CNN', help="Form of model, i.e dan, rnn, etc.")
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

	train_data['targets'][0] = 2

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

	model = models.resnet18(pretrained=True)

	trained_model = train_model(loaders, model, loss_weights, args)
	


	