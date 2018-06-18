import torchvision
import torch.nn as nn
from models import custom_models

def get_model(args):
	'''
	Retrieve the right model to pass to 'train_model' function. 

	:param args: Parser arguments
	'''

	if args.model_name == 'resnet18':

		model = torchvision.models.resnet18(pretrained=True)
		model.avgpool = nn.AdaptiveAvgPool2d(1)

		# Replace the fully connected layer
		num_features = model.fc.in_features
		model.fc = nn.Linear(num_features, args.num_classes)
		optim_params = model.fc.parameters()

		return model

	elif args.model_name == 'simple_CNN':

		model = custom_models.simple_CNN()

		return model

	elif args.model_name == 'resnet+meteo_NN':

		resnet_model = torchvision.models.resnet18(pretrained=True)
		meteo_NN = custom_models.meteo_NN(args.meteo_inputs, args.meteo_hidden_size, args.meteo_outputs)

		model = custom_models.resnet18_meteo(resnet_model, meteo_NN, args.num_classes)

		return model

	else:
		raise ValueError("Define one of 'resnet18', 'simple_CNN' or 'resnet+meteo_NN'")