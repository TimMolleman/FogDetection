import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def show_cm(targets, predictions):
	'''
	Shows a confusion matrix for model testing.

	:param targets: Numpy array containing targets
	:param predictions: Numpy array containing corresponding predictions
	'''
	cm = confusion_matrix(y_target=targets, 
						y_predicted=predictions, 
						binary=False)

	fig, ax = plot_confusion_matrix(conf_mat=cm)
	plt.show(block=True)

def plot_loss_curves(training_loss, validation_loss):
	"""
	Plots loss curves of trained model.
	
	:param training_loss: List with training loss for every epoch.
	:param validation_loss: List with validation loss for every epoch.
	"""
	train_plot, = plt.plot(training_loss, label='Training')
	val_plot, = plt.plot(validation_loss, label='Validation')
	plt.title('Loss curves (training/validation)')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(handles=[train_plot, val_plot])
	plt.show(block=True)