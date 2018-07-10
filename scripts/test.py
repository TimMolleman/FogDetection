import argparse
import torch
import sys
sys.path.insert(0, '..')

from helpers.model_plotters import *
from models.model_utils import *
from train.test_utils import *
from helpers.data_loader import *

from torch import nn
from torch.autograd import Variable
from helpers.model_plotters import show_cm

parser = argparse.ArgumentParser(description='Testing models for fog detection')

# Paths
parser.add_argument('--test_data_path', type=str, default='../data/processed/test_IDW.npy', help='path to test data numpy array')
parser.add_argument('--checkpoint_path', type=str, default='../experiments/ex1/resnet18_completed', help='path to a trained model to be tested')
parser.add_argument('--img_path', type=str, default='../img', help='path to a trained model to be tested')

# Measure and model type
parser.add_argument('--state_performance_measure', type=str, default='avg', help='decide what performance measure to select model states on, either \'avg\' or \'f1\'')
parser.add_argument('--model_name', type=str, default='resnet18', help='resnet18, simple_CNN or merged_net based on which model architecture is tested')
parser.add_argument('--include_meteo', type=int, default=False, help='include meteo in model testing or not [default: False]')

# Plot saving
parser.add_argument('--save_confusion_matrix', type=bool, default=False, help='if True saves the confusion matrix on test dataset to img folders [default: False]')
parser.add_argument('--save_loss_curves', type=bool, default=False, help='if True plots and saves the loss curves for training/validation dataset [default: False]')

args = parser.parse_args()

# Get test data numpy dict
test_dict = np.load(args.test_data_path)[()]

# Get dataloader for testing
test_dataset = FogDataset(test_dict, transforms=custom_transforms['eval'])
test_loader = create_loader(test_dataset, 'test', len(test_dict['targets']))

# Get checkpoint and best model state
checkpoint = load_model(args.checkpoint_path)
model = checkpoint['best_model_{}'.format(args.state_performance_measure)]

if args.save_loss_curves:

	# Plot and save loss curves
	train_loss = checkpoint['train_loss']
	validation_loss = checkpoint['validation_loss']
	figure = plot_loss_curves(train_loss, validation_loss)
	figure.savefig('{}/LearnCurves/{}'.format(args.img_path, args.model_name))

# Test model and get confusion matrix
cm = test_model(model, test_loader, args)

if args.save_confusion_matrix:
	cm.savefig('{}/ConfusionMatrices/{}'.format(args.img_path, args.model_name))



