import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


def show_cm(targets, predictions):
	'''
	Shows a confusion matrix for model testing.

	:param targets: Numpy array containing targets
	:param predictions: Numpy array containing corresponding predictions
	:return: figure object containing confusion matrix
	'''
	cm = confusion_matrix(y_target=targets, 
						y_predicted=predictions, 
						binary=False)

	fig, ax = plot_confusion_matrix(conf_mat=cm)
	plt.show(block=True)

	return fig


def plot_loss_curves(training_loss, validation_loss):
	"""
	Plots loss curves of trained model.
	
	:param training_loss: List with training loss for every epoch.
	:param validation_loss: List with validation loss for every epoch.
	:return: figure object containing loss curves
	"""
	fig = plt.figure(figsize= (8,8))
	train_plot, = plt.plot(training_loss, label='Training')
	val_plot, = plt.plot(validation_loss, label='Validation')
	plt.title('Loss curves (training/validation)')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(handles=[train_plot, val_plot])
	plt.show(block=True)

	return fig