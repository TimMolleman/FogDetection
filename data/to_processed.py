import argparse
import numpy as np
import pandas as pd
import os
import re
import csv
import torch
from PIL import Image
from sklearn import preprocessing

parser = argparse.ArgumentParser(description='Processing of raw data')

# Filepaths
parser.add_argument('--processed_dir', type=str, default='processed/', help='path to directory for storing processed data')
parser.add_argument('--test_image_dir', type=str, default='/Volumes/TIMKNMI/KNMIPictures/RWS/', help='path to directory containing test images')
parser.add_argument('--knmi_relabel_path', type=str, default='helpers/knmichangelabels', help='path to file containing new labels for knmi images')
parser.add_argument('--highway_relabel_path', type=str, default='helpers/trainhighwaylabels', help='path to file containing new labels for highway images')
parser.add_argument('--highway_ommit_path', type=str, default='helpers/ommitancehighway', help='path to file with highway images to be removed')
parser.add_argument('--test_image_filename', type=str, default='TestImagesRefined.txt', help='path to file with highway images to be removed')
parser.add_argument('--semi_processed_dir', type=str, default='semi-processed', help='path to directory where dataframes are stored')
parser.add_argument('--main_dataframe', type=str, default='main_dataframe_IDW', help='filename of main dataframe')
parser.add_argument('--test_dataframe', type=str, default='test_df_IDW', help='filename of test dataframe')

# Meteo variable filling method
parser.add_argument('--meteo_method', type=str, default='IDW', help='method used for filling missing meteorological values, either IDW or harmonie')

# Standardize
parser.add_argument('--standardize_meteo', type=bool, default=False, help='decides if meteo variables should be standardized [default: False]')

args = parser.parse_args()

# Image parameters
IMG_SIZE = 100
CHANNELS = 3
KNMI_NAMES = ['De Bilt (260_A_a)', 'Cabauw (348)', 'BEEK airport', 'EELDE airport', 'ROTTERDAM airport', 'SCHIPHOL airport']


def df_to_numpy_train(df, dataset_name):
	'''
	Transforms training dataframe to numpy arrays. 

	:param df: highway train dataframe or KNMI dataframe
	:param dataset_name: either 'highway' or 'KNMI', depending on dataframe in first argument
	:return: four numpy arrays containing the images, targets, filepaths and meteorological variables
	'''

	image_list, target_list, filepath_list, meteo_list = [], [], [], []

	# Loop over passed dataframe
	for c, (index, row) in enumerate(df.iterrows()):
		# Get necessary information
		camera = row['camera_name']
		year_month = row['datetime'].strftime("%Y%m")  
		file = row['filepath']
		meteo = row[['wind_speed', 'rel_humidity', 'air_temp', 'dew_point']]

		# Different folder structures of KNMI and 
		if dataset_name == 'KNMI':
			folder = row['filepath'].split('-')[0]
			folder = folder.split('_')[0]
			filepath = '/Volumes/TIMPP/KNMIPictures/{}/{}/{}/{}'.format(folder, camera, year_month, file)

		elif dataset_name == 'highway':
			location = row['location_name'].split('-')
			filepath = '/Volumes/TIMKNMI/KNMIPictures/UnlabeledRWS/{}/{}/{}/{}/{}'.format(location[0], location[1], camera,
																				 year_month, file)
		else:
			raise ValueError("Specify either 'KNMI' or 'highway'") 

		# Not every image in the df is in folder and vice versa. We don't want to throw errors because of this
		try:
			# Here the actual image is loaded to an array
			img = Image.open(filepath)
			img = img.resize((IMG_SIZE, IMG_SIZE))
			image_list.append(np.asarray(img))
			img.close()

			# Other arrays
			target = df[df['filepath'] == file]['visibility']
			target_list.append(target.iloc[0])
			filepath_list.append(filepath)
			meteo_list.append(meteo)

		except:
			print('Could not load image: {}'.format(file))
			continue

		  # Print number of iterated df rows every 500 iterations
		if c % 500 == 0:
			print('Number of df rows iterated: {} of {}'.format(c, len(df)))

	print('Loaded all {} images'.format(dataset_name))

	return np.asarray(image_list), np.asarray(target_list), np.asarray(filepath_list), np.asarray(meteo_list)

def df_to_test_dict(test_df, test_dir, filename_test):
	'''
	Loads the testing data to a dictionary of numpy arrays (images, targets, filepaths, meteorological variables).

	:param test_df: test dataframe
	:param test_dir: test directory location
	:param filename_test: filename for the test txt file
	:return: dictionary containing the four numpy arrays
	'''

	# Used for storing filename:label pairs
	image_list_test, target_list_test, filepath_list_test, meteo_list_test = [], [], [], []

	# This opens all labeled files and finds the corresponding pictures and labels
	with open(test_dir + filename_test) as filestream:
	
	    for row in filestream:
	        row = row.strip().split(',')
	        filename = row[0]
	        label = row[1]
	        
	        datapoint_df = test_df[test_df['filepath'] == filename]
	        meteo = datapoint_df[['wind_speed', 'rel_humidity', 'air_temp', 'dew_point']]

	        if len(np.asarray(meteo)) > 0:
	    
		        # Regex necessary elements
		        highway = re.search(r'A\d*', filename).group(0)
		        ID = re.search(r'ID\d*', filename).group(0)
		        HM = re.search(r'HM\d*', filename).group(0)
		        year_month = re.search(r'_\d*_', filename).group(0)[1:7]

		        path = '{}{}/{}/{}/{}/{}'.format(test_dir, highway, HM, ID, year_month, filename)

		        # Get all variables, labels and filepaths for image
		        img = Image.open(path)
		        img = img.resize((IMG_SIZE, IMG_SIZE))
		        image_list_test.append(np.asarray(img))
		        target_list_test.append(np.asarray(label))
		        filepath_list_test.append(np.asarray(path))
		        meteo_list_test.append(np.asarray(meteo))

		        img.close()

	        else:
	        	print('Couldn\'t find meteorological variables for {}'.format(filename))
            
	# Transform lists to arrays and then put them in dictionary 
	test_features, test_targets = np.asarray(image_list_test), np.asarray(target_list_test).astype(int)
	test_filepaths, test_meteo = np.asarray(filepath_list_test), np.asarray(meteo_list_test)
	test_meteo = test_meteo.reshape(len(test_meteo), 4)
	test_dict = {'images': test_features, 'targets': test_targets, 'filepaths': test_filepaths, 'meteo': test_meteo}

	return test_dict


def change_labels(target_array, filepath_array, filepath_changes):
	'''
	Changes labels in a targets array. These labels were manually decided by inspection of images.

	:param target_array: array containing targets
	:param filepath_array: array containing filepaths for matching rows in file
	:param filepath_changes: filepath to csv containing the changed labels
	:return: changed target array
	'''

	with open(filepath_changes) as file:
		for row in file:
			path, target = row.split(',')
			path_idx = np.where(filepath_array == path.strip("'"))
			target_array[path_idx] = target
		file.close()

	print('Changed labels.')

	return target_array

def ommit_labels_KNMI(KNMI_images, KNMI_targets, KNMI_filepaths, KNMI_meteo):
	'''
	Ommits KNMI labels. Decided on by inspection of images.

	:param KNMI_images: KNMI image array
	:param KNMI_targets: KNMI targets array
	:param KNMI_filepaths: KNMI filepaths array
	:param KNMI_meteo: KNMI array containing meteorological variables
	:return: all input arrays with bad indices ommitted
	'''
	# Get indices to delete from KNMI numpy arrays
	del_knmi_idx = [i for i, v in enumerate(KNMI_filepaths) if 'BSRN-1' in v or 'Meetterrein_201606' in v or 'Meetterrein_201607' in v
				or 'Meetterrein_201608' in v]
	
	# Ommit bad images from numpy arrays
	KNMI_images = np.delete(KNMI_images, del_knmi_idx, 0)
	KNMI_targets = np.delete(KNMI_targets, del_knmi_idx, 0)
	KNMI_filepaths = np.delete(KNMI_filepaths, del_knmi_idx, 0)
	KNMI_meteo = np.delete(KNMI_meteo, del_knmi_idx, 0)

	print('Ommited KNMI labels.')

	return KNMI_images, KNMI_targets, KNMI_filepaths, KNMI_meteo


def ommit_labels_highway(highway_images, highway_targets, highway_filepaths, highway_meteo, ommitance_filepath):
	'''
	Ommits indices out of highway numpy arrays that are bad images for training/validation.


	:param highway_images: numpy array containing highway images
	:param highway_targets: numpy array containing highway targets
	:param highway_filepaths: numpy array containing highway filepaths
	:param highway_meteo: numpy array containing highway meteo variables
	:param ommitance_filepath: contains paths of images that have to be ommitted	
	:return: all input arrays with bad indices ommitted
	'''

	with open(ommitance_filepath) as file:

		indices = []

		for row in file:
			row = " ".join(row.split())
			path = row.strip("'")
			path_idx = np.where(highway_filepaths == row.strip("'"))
			indices.append(path_idx[0][0])
		file.close()

		highway_targets = np.delete(highway_targets, indices, 0)
		highway_images = np.delete(highway_images, indices, 0)
		highway_filepaths = np.delete(highway_filepaths, indices, 0)
		highway_meteo = np.delete(highway_meteo, indices, 0)

	print('Ommitted highway labels.')

	return highway_targets, highway_images, highway_filepaths, highway_meteo

def create_highway_df(main_df):
	'''
	Get all the highways camera's that are closest to Schiphol and Eelde airport.

	:param main_df: main dataframe containing all the images 
	:return: highway image dataframe with images that were taken within 7.500 meter distance
	of Eelde and Schiphol airport
	'''

	schiphol_df = main_df[main_df['MeteoStationLocationID'] == 542]
	schiphol_highways = schiphol_df[schiphol_df['location_name']!= 'SCHIPHOL airport']
	eelde_df = main_df[main_df['MeteoStationLocationID'] == 506]
	eelde_highways = eelde_df[eelde_df['location_name'] != 'EELDE airport']
	highway_df = pd.concat([eelde_highways, schiphol_highways])

	# Make sure to only use camera's within 7.500 meters of airport sensors
	highway_df = highway_df[highway_df['distanceInMeters'] < 7500]

	return highway_df

def split_highway(highway_images, highway_targets, highway_filepaths, highway_meteo, std_meteo):
	'''
	Splits the labeled highway dataset into two separate datasets: Train dataset and validation set.
	5 camera's are used for validation set. Other camera's are used for training.

	:param highway_images: numpy array with highway images
	:param highway_images: numpy array with target images
	:param highway_filepaths: numpy array with the filepathss
	:param highway_meteo: numpy array with the meteorological variables
	:param std_meteo: if True, values for meteorological variables are standardized
	:return: two dictionaries 1) contains train highway arrays 2) contains validation highway arrays
	'''

	# Normalize meteo variables over the whole dataset
	if std_meteo:
		highway_meteo = standardize_meteo(highway_meteo)

	# List of cameras to use as validation set
	camera_list = ['A28/HM1893', 'A4/HM103', 'A5/HM86', 'A9/HM302', 'A28/HM1966']

	# Get validation numpy arrays
	val_idx = [i for camera in camera_list for i, v in enumerate(highway_filepaths) if camera in v]
	X_validation, y_validation = highway_images[val_idx], highway_targets[val_idx].astype(int)
	paths_validation, meteo_validation = highway_filepaths[val_idx], highway_meteo[val_idx] 

	# Get the highway train dataset by deleting validation indices
	X_train_highway = np.delete(highway_images, val_idx, 0)
	y_train_highway = np.delete(highway_targets, val_idx, 0).astype(int)
	paths_train_highway = np.delete(highway_filepaths, val_idx, 0)
	meteo_train_highway = np.delete(highway_meteo, val_idx, 0)

	# Meteo to tensors
	meteo_train_highway = torch.Tensor(meteo_train_highway)
	meteo_validation = torch.Tensor(meteo_validation)

	# Make dictionaries from the numpy arrays and return
	highway_train_dict = {'images': X_train_highway, 'targets': y_train_highway, 'filepaths': paths_train_highway, 'meteo': meteo_train_highway}
	highway_val_dict = {'images': X_validation, 'targets': y_validation, 'filepaths': paths_validation, 'meteo': meteo_validation}

	return highway_train_dict, highway_val_dict

def standardize_meteo(meteo_np):
	'''
	Standardizes the values for the four meteorological variables. If standardizing highway dataset:
	Call right before splitting into train/validation sets.

	:param meteo_np: numpy array containing meteo variable values, each column being one variable
	:return: standardized meteo array
	'''

	standardize_df = pd.DataFrame(meteo_np, columns=[0, 1, 2, 3])
	standardize_df = standardize_df.dropna(axis=0)
	std_scale = preprocessing.StandardScaler().fit(standardize_df[[0, 1, 2, 3]])
	standardized_meteo = std_scale.transform(standardize_df[[0, 1, 2, 3]])

	return standardized_meteo

def delete_np_nan(images, targets, filepaths, meteo):
	'''
	Double check to make sure no NaN values for visibility are in the numpy arrays.

	:param images: images numpy array
	:param targets: targets numpy array
	:param filepaths: filepaths numpy array
	:param meteo: meteo numpy array
	:return: numpy arrays with indices in targets array containing NaN removed 
	'''

	indices = np.isnan(targets)

	images = images[~indices]
	targets = targets[~indices]
	filepaths = filepaths[~indices]
	meteo = meteo[~indices]

	return images, targets, filepaths, meteo

def main():

	# Read the semi-processed data
	main_df = pd.read_pickle('{}/{}'.format(args.semi_processed_dir, args.main_dataframe))
	test_df = pd.read_pickle('{}/{}'.format(args.semi_processed_dir, args.test_dataframe))

	# Get KNMI df and highway df
	KNMI_df = main_df[main_df['location_name'].isin(KNMI_NAMES)][:500]
	highway_df = create_highway_df(main_df)[:500]

	# Load KNMI df and highway df to numpy arrays
	KNMI_images, KNMI_targets, KNMI_filepaths, KNMI_meteo = df_to_numpy_train(KNMI_df, 'KNMI')
	highway_images, highway_targets, highway_filepaths, highway_meteo = df_to_numpy_train(highway_df, 'highway')

	highway_images = np.load('/Volumes/TIMPP/UnusedKNMI/numpyfiles/meteo_train/valtestinterpolated/highway/highway_images_IDWValTest.npy')
	highway_targets = np.load('/Volumes/TIMPP/UnusedKNMI/numpyfiles/meteo_train/valtestinterpolated/highway/highway_targets_IDWValTest.npy')
	highway_filepaths = np.load('/Volumes/TIMPP/UnusedKNMI/numpyfiles/meteo_train/valtestinterpolated/highway/highway_filepaths_IDWValTest.npy')
	highway_meteo = np.load('/Volumes/TIMPP/UnusedKNMI/numpyfiles/meteo_train/valtestinterpolated/highway/highway_meteo_IDWValTest.npy')

	# Make sure indices of NaN in targets array are removed from every np array
	highway_images, highway_targets, highway_filepaths, highway_meteo = delete_np_nan(highway_images, highway_targets, highway_filepaths, highway_meteo)
	KNMI_images, KNMI_targets, KNMI_filepaths, KNMI_meteo = delete_np_nan(KNMI_images, KNMI_targets, KNMI_filepaths, KNMI_meteo)

	highway_images = highway_images[:10000]
	highway_targets = highway_targets[:10000]
	highway_filepaths = highway_filepaths[:10000]
	highway_meteo = highway_meteo[:10000]

	# Relabel the KNMI and highway targets
	KNMI_targets = change_labels(KNMI_targets, KNMI_filepaths, args.knmi_relabel_path)
	highway_targets = change_labels(highway_targets, highway_filepaths, args.highway_relabel_path)

	# Ommit labels of KNMI and highway datasets
	# KNMI_images, KNMI_targets, KNMI_filepaths, KNMI_meteo = ommit_labels_KNMI(KNMI_images, KNMI_targets, KNMI_filepaths, KNMI_meteo)
	# highway_images, highway_targets, highway_filepaths, highway_meteo = ommit_labels_highway(highway_images, highway_targets, highway_filepaths, highway_meteo, args.highway_ommit_path)

	# Split the highway dataset into training and validation
	highway_train_dict, highway_val_dict = split_highway(highway_images, highway_targets, highway_filepaths, highway_meteo, args.standardize_meteo)

	# Create KNMI dict
	KNMI_dict = {'images': KNMI_images, 'targets': KNMI_targets, 'filepaths': KNMI_filepaths, 'meteo': torch.Tensor(KNMI_meteo)}

	# Load the test dictionaries
	test_dict = df_to_test_dict(test_df, args.test_image_dir, args.test_image_filename)

	# Save the arrays into numpy dictionaries
	np.save(args.processed_dir + 'KNMI_'+args.meteo_method+'.npy', KNMI_dict)
	np.save(args.processed_dir + 'highway_train_'+args.meteo_method+'.npy', highway_train_dict)
	np.save(args.processed_dir + 'highway_val_'+args.meteo_method+'.npy', highway_val_dict)
	np.save(args.processed_dir + 'test_'+args.meteo_method+'.npy', test_dict)

	print('Done! Processed the data to a dictionary of numpy arrays.')

if __name__ == '__main__':
	main()




