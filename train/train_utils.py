import torch
import numpy as np
import torchvision
import torch.nn as nn
import time
import copy

from torch.autograd import Variable
from sklearn.metrics import f1_score


def train_model(model, data_loaders, weights, args):
	'''
	Main function used for training the networks.

	:param model: Model to train
	:param data_loaders: Dictionary containing train dataloader and validation dataloader
	:param weights: Weights for the weighting of loss. Passed to criterion
	'''

	print('Start training of {} model:\n'.format(args.model_name))

	# Get weights to optimize
	optim_params = get_optim_params(model, args)

	# Enable cuda if available
	if args.cuda:
		model = model.cuda()
		weights = weights.cuda()

	# Define loss function, optimizer and start time
	criterion = nn.CrossEntropyLoss(reduce=False, weight=weights)
	optimizer = torch.optim.Adam(optim_params, lr=args.lr)
	start = time.time()

	# Dict for storing a number of important variables
	checkpoint_dict = {'train_loss':[], 'validation_loss':[], 'best_f1macro': 0.0, 'best_epoch_f1' : 0.0,
		'best_model_f1': model, 'best_avg_accuracy': 0.0, 'best_epoch_avg': 0, 'best_model_avg': model}

	for epoch in range(1, args.epochs+1):

		model, measures, epoch_loss_train, epoch_loss_val = run_epoch(model, data_loaders, optimizer, criterion, epoch, start, args)
		
		# Update other checkpoints
		create_checkpoint(checkpoint_dict, model, measures, epoch, epoch_loss_train, epoch_loss_val)

		# Save the checkpoint at specified path
		torch.save(checkpoint_dict, args.save_path)

	# Print the end status of the model
	elapsed_time = time.time() - start
	print('Training was completed in {:.0f}m {:.0f}s\n'.format(elapsed_time//60, elapsed_time%60))
	print('Best validation avg accuracy: {:.4f}% at epoch: {}'.format(checkpoint_dict['best_avg_accuracy'], checkpoint_dict['best_epoch_avg']))
	print('Best validation f1-macro: {:.4f} at epoch: {}'.format(checkpoint_dict['best_f1macro'], checkpoint_dict['best_epoch_f1']))

	return model

def create_checkpoint(checkpoint_dict, model, measures, epoch, epoch_loss_train, epoch_loss_val):
	'''
	Adjusts the checkpoint dictionary after each epoch. Returns this adjusted dictionary. 

	'''
	# Update variables if the model state is better than previous model states 
	if measures['f1_macro'] > checkpoint_dict['best_f1macro']:
		checkpoint_dict['best_f1macro'] = measures['f1_macro']
		checkpoint_dict['best_epoch_f1'] = epoch
		checkpoint_dict['best_model_f1'] = copy.deepcopy(model)
	if measures['avg_acc'] > checkpoint_dict['best_avg_accuracy']:
		checkpoint_dict['best_avg_accuracy'] = measures['avg_acc']
		checkpoint_dict['best_epoch_avg'] = epoch
		checkpoint_dict['best_model_avg'] = copy.deepcopy(model)

	# Extend arrays containing the loss
	checkpoint_dict['train_loss'].append(epoch_loss_train)
	checkpoint_dict['validation_loss'].append(epoch_loss_val)
	
	return checkpoint_dict


def run_epoch(model, loaders, optimizer, criterion, epoch, start, args):
	running_loss_train = 0.0
	running_correct_train = 0.0
	running_loss_val = 0.0
	running_correct_val = 0.0
	validation_targets = []
	validation_predictions = []

	for phase in ['train', 'validation']:

		if phase == 'train':
			model.train()
		else:
			model.eval()

		# Iterate over batches in loader
		for i, (image_tensor, label_tensor, image_index, filepaths, meteo) in enumerate(loaders[phase]):

			features = Variable(image_tensor)
			targets = Variable(label_tensor.view(-1))
			meteo_features = Variable(meteo.type(torch.FloatTensor))

			# If cuda is available, make variables cuda
			if args.cuda:
				features = features.cuda()
				targets = targets.cuda()
				meteo_features = meteo_features.cuda()
			print(features.size())
			if phase == 'train':
				optimizer.zero_grad()

			# Check if you want to train 
			if args.include_meteo:
				outputs = model(features, meteo_features)
			else:
				outputs = model(features)

			 # Get prediction index and no. correct predictions
			_, predictions = torch.max(outputs.data, 1)          
			correct = torch.sum(predictions == targets.data) 

			# Loss and optimization
			loss = criterion(outputs, targets)
			
			# Average the loss
			total_loss = torch.mean(loss)
			
			# Only do backpropagation if in the training phase
			if phase == 'train':
				total_loss.backward()
				optimizer.step()

			# Running loss and number of correct predictions
			if phase == 'train':
				running_loss_train += total_loss.data[0]
				running_correct_train += correct
			else:
				running_loss_val += total_loss.data[0]
				running_correct_val += correct
				validation_targets.extend(list(targets.data))
				validation_predictions.extend(list(predictions))

			# If model is in training phase, show loss every N iterations
			# if (i+1) % 2 == 0:
			if phase == 'train':
				print ('Epoch {}/{}, Iteration {}/{} Train Running Loss: {:.4f}'.format(epoch, args.epochs, i+1, 
																			len(loaders[phase].dataset)//args.batch_size + 1, 
																			running_loss_train / (i+1)))

	cur_time = time.time() - start

	# Gather epoch losses and evaluation metrics
	epoch_train_loss = running_loss_train / (len(loaders['train'].dataset)//args.batch_size)
	epoch_train_accuracy = (running_correct_train / (len(loaders['train'].dataset)) / args.batch_size * 100)
	epoch_val_loss = running_loss_val / (len(loaders['validation'].dataset) // args.batch_size)
	epoch_val_accuracy= (running_correct_val / (len(loaders['validation'].dataset) // args.batch_size)) / args.batch_size * 100
	f1_macro = f1_score(validation_targets, validation_predictions, average='macro')
	f1_micro = f1_score(validation_targets, validation_predictions, average='micro')
	average_accuracy = get_average_accuracy(validation_predictions, validation_targets)


	# Print the average epoch loss and the average prediction accuracy
	print('\nEpoch {}/{}, Train Time: {:.0f}m {:.0f}s\n Train Loss: {:.4f}, Train Overall Accuracy: {:.4f}%\n'
		  'Validation Loss: {:.4f}, Validation Overall Accuracy: {:.4f}%, Validation Avg Accuracy: {:.4f}% f1_macro: {:.4f}, f1_micro: {:.4f}\n'.format(epoch, 
																				args.epochs, cur_time//60, cur_time%60, epoch_train_loss, epoch_train_accuracy,
																				epoch_val_loss, epoch_val_accuracy, average_accuracy, f1_macro, f1_micro))                          

	# Save the model state selection measures
	selection_measures = {'f1_macro' : f1_macro, 'avg_acc' : average_accuracy}

	return model, selection_measures, running_loss_val, running_correct_val

def get_average_accuracy(predictions, targets):
	'''
	Calculate the average accuracy for an epoch.

	:param predictions: Numpy array containing predictions of model training epoch.
	:param targets: Numpy array containing targets of model training epoch.
	'''
  
	# Lists for holding corrects
	no_fog_correct = 0
	light_fog_correct = 0
	dense_fog_correct = 0

	for pred, target in zip(predictions, targets):
		if pred == 0 and target == 0:
			no_fog_correct += 1
		elif pred == 1 and target == 1:
			light_fog_correct += 1
		elif pred == 2 and target == 2:
			dense_fog_correct += 1

	# Validation counts
	total = np.bincount(targets)
	no_fog_total = total[0]
	light_fog_total = total[1]
	dense_fog_total = total[2]

	# Accuracy per class
	acc_no_fog = no_fog_correct / no_fog_total
	acc_light = light_fog_correct / light_fog_total
	acc_dense = dense_fog_correct / dense_fog_total

	average_acc = (acc_no_fog + acc_light + acc_dense) / 3 * 100

	return average_acc

def get_optim_params(model, args):
	'''
	Retrieve the weights that have to be optimized.

	:param model: Model class to be trained
	:param args: Parser arguments
	'''

	if args.model_name != 'shallow_CNN':

		# Per default, freeze all weights and unfreeze the FC layer
		for parameter in model.parameters():
			parameter.requires_grad = False

		for parameter in model.fc.parameters():
			parameter.requires_grad = True

		# Check to see if additional convolutional blocks should be unfrozen
		if args.from_conv_layer:

			# Check if meteo model or not
			if args.include_meteo:
				for parameter in model.meteo_net.parameters():
					parameter.requires_grad = True    

				for i, (name, child) in enumerate(model.resnet18_convblocks.named_children()):
					if i + 1 > args.from_conv_layer:
						for name2, params in child.named_parameters():
							params.requires_grad = True

			else:
				# Unfreeze layer until 'from_conv_layer'
				for i, (name, child) in enumerate(model.named_children()):

					if i + 1 > args.from_conv_layer:
						print(name)
						for name2, params in child.named_parameters():
							params.requires_grad = True

	# Print number of countable parameters
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Number of weights that will be trained: {}'.format(params))

	# Get the optimizer parameters
	optim_params = filter(lambda p: p.requires_grad, model.parameters())

	return optim_params

def calculate_loss_weights(train_targets):
	'''
	Calculates weights for the loss function. Should be used when training the model by feeding it to
	criterion.

	:param train_targets: Numpy array containing the train targets.
	'''

	class_counts = np.bincount(train_targets.astype(int))
	total = len(train_targets)

	proportion_0 = class_counts[0] / total
	proportion_1 = class_counts[1] / total
	proportion_2 = class_counts[2] / total
	proportions = [proportion_0, proportion_1, proportion_2]

	inverse_weights = 1 / torch.Tensor(proportions)

	return inverse_weights





