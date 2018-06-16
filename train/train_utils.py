import torch
import numpy as np

def train_model(data_loaders, model, weights, args):
	'''
	Main function used for training the networks.

	:param train_data: 
	'''

	# Get weights to optimize
	optim_params = get_optim_params(model, args.from_conv_layer)

	# Enable cuda if available
	if args.cuda:
		model = model.cuda()
		weights = weights.cuda()

	# Define loss function, optimizer and start time
	criterion = nn.CrossEntropyLoss(reduce=False, weight=inverse_weights)
	optimizer = torch.optim.Adam(optim_params, lr=args.lr)
	start = time.time()

	# Dict for storing a number of important variables
	checkpoint_dict = {'train_loss':[], 'validation_loss':[], 'best_f1macro': 0.0, 'best_epoch_f1' : 0,
		'best_model_f1': model, 'best_avg_accuracy': 0.0, 'best_epoch_avg': 0, 'best_model_avg': model}

	for epoch in range(1, args.epochs+1):

		model, measures, epoch_loss_train, epoch_loss_val = run_epoch(model, args)
		
		# Update other checkpoints
		checkpoint_dict = create_checkpoint(checkpoint_dict, model, measures, epoch_loss_train, epoch_loss_val)

		# Save the checkpoint at specified path
		torch.save(checkpoint_dict, args.save_path)

	# Print the end status of the model
	elapsed_time = time.time() - start
	print('Training was completed in {:.0f}m {:.0f}s\n'.format(elapsed_time//60, elapsed_time%60))
	print('Best validation avg accuracy: {:.4f}% at epoch: {}'.format(checkpoint_dict['best_avg_accuracy'], checkpoint_dict['best_epoch_avg']))
	print('Best validation f1-macro: {:.4f} at epoch: {}'.format(checkpoint_dict['best_accuracy'], checkpoint_dict['best_epoch_f1']))

	return model

def create_checkpoint(checkpoint_dict, model, measures, epoch_loss_train, epoch_loss_val):
	'''
	Adjusts the checkpoint dictionary after each epoch. Returns this adjusted dictionary. 

	'''
	# Update variables if the model state is better than previous model states 
	if measures['f1_macro'] > checkpoint_dict['best_f1macro']:
		checkpoint_dict['best_f1macro'] = measures['f1_macro']
		checkpoint_dict['best_epoch_f1'] = epoch
		checkpoint_dict['best_model_f1'] = copy.deepcopy(model)
	if measures['avg_acc'] > checkpoint_dict['best_avg_accuracy']:
		checkpoint_dict['best_avg_accuracy'] = average_accuracy
		checkpoint_dict['best_epoch_avg'] = epoch
		checkpoint_dict['best_model_avg'] = copy.deepcopy(model)


	# Extend arrays containing the loss
	checkpoint_dict['train_loss'] = train_loss.append(epoch_loss_train)
	checkpoint_dict['validation_loss'] = validation_loss.append(epoch_loss_val)

	return checkpoint_dict

def run_epoch(model, args):
	running_loss_train = 0.0
	running_correct_train = 0.0
	running_loss_val = 0.0
	running_correct_val = 0.0
	epoch_validation_targets = []
	epoch_validation_predictions = []

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
				epoch_validation_targets.extend(list(targets.data))
				epoch_validation_predictions.extend(list(predictions))

			 # If model is in training phase, show loss every N iterations
			if (i+1) % 50 == 0:
				if phase == 'train':
					print ('Epoch {}/{}, Iteration {}/{} Train Running Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, 
																				len(X_train)//BATCH_SIZE, 
																				running_loss_train / i))                          

	# Return measures to see if model is better than previous model
	measures = epoch_scores(running_loss_train, running_correct_val, running_loss_val, running_correct_val, epoch_validation_targets,
							epoch_validation_predictions, start)

	return model, measures, running_loss_val, running_correct_val

def get_average_accuracy(predictions, targets):
	'''
	Calculate the average accuracy.

	:param predictions: 
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
	total = np.bincount(validation_targets)
	no_fog_total = total[0]
	light_fog_total = total[1]
	dense_fog_total = total[2]

	# Accuracy per class
	acc_no_fog = no_fog_correct / no_fog_total
	acc_light = light_fog_correct / light_fog_total
	acc_dense = dense_fog_correct / dense_fog_total

	average_acc = (acc_no_fog + acc_light + acc_dense) / 3 * 100

	return average_acc

def epoch_scores(running_loss_train, running_correct_train, running_loss_val, 
				running_correct_val, validation_targets, validation_predictions, start,
				epoch):
	
	cur_time = time.time() - start

	# Epoch losses and epoch train accuracies
	epoch_train_loss = running_loss_train / (len(X_train)//BATCH_SIZE)
	epoch_train_accuracy = (running_correct_train / (len(X_train)//BATCH_SIZE)) / BATCH_SIZE * 100
	epoch_val_loss = running_loss_val / (len(X_validation) // BATCH_SIZE)
	epoch_val_accuracy= (running_correct_val / (len(X_validation) // BATCH_SIZE)) / BATCH_SIZE * 100

	f1_macro = f1_score(epoch_validation_targets, epoch_validation_predictions, average='macro')
	f1_micro = f1_score(epoch_validation_targets, epoch_validation_predictions, average='micro')
	
	# Average accuracy of epoch
	average_accuracy = get_average_accuracy(validation_predictions, validation_targets)

	 # Print the average epoch loss and the average prediction accuracy
	print('\nEpoch {}/{}, Train Time: {:.0f}m {:.0f}s\n Train Loss: {:.4f}, Train Overall Accuracy: {:.4f}%\n'
		  'Validation Loss: {:.4f}, Validation Overall Accuracy: {:.4f}%, Validation Avg Accuracy: {:.4f}% f1_macro: {:.4f}, f1_micro: {:.4f}\n'.format(epoch, 
																				num_epochs, cur_time//60, cur_time%60, epoch_train_loss, epoch_train_accuracy,
																				epoch_val_loss, epoch_val_accuracy, average_accuracy, f1_macro, f1_micro))

	# Return the measures that are used for selecting the best model state
	selection_measures = {'epoch_f1_macro' : f1_macro, 'avg_acc' : avg_accuracy}

	return selection_measures

def get_optim_params(model, from_conv_layer):

	if from_conv_layer:

		# Per default, freeze all weights
		for parameter in model.parameters():
			parameter.required_grad = False

		# Unfreeze layer until 'from_conv_layer'
		for i, (name, child) in enumerate(model.named_children):

			if i > from_conv_layer:
				for name2, params in child.named_parameters():
					params.required_grad = True

	else:

		# Get the optimizer parameters. Only FC if from_conv_layer == False
		optim_params = filter(lambda p: p.requires_grad, model.parameters())

	return optim_params

def calculate_loss_weights(train_targets):
	class_counts = np.bincount(train_targets.astype(int))
	total = len(train_targets)
	print(class_counts)
	proportion_0 = class_counts[0] / total
	proportion_1 = class_counts[1] / total
	proportion_2 = class_counts[2] / total
	proportions = [proportion_0, proportion_1, proportion_2]

	print('Class percentages:\nNo fog: {:.2f}%\nFog: {:.2f}%\nDense fog: {:.2f}%'.format(proportion_0 * 100,
																				  proportion_1 * 100, proportion_2 * 100))
	print(class_counts)

	inverse_weights = 1 / torch.Tensor(proportions)

	return inverse_weights

