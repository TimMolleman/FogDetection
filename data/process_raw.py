import numpy as np
import pandas as pd
import os
import re
import csv
from PIL import Image
from sklearn import preprocessing

# Image parameters
IMG_SIZE = 100
CHANNELS = 3

def df_to_numpy_train(df, dataset_name):
	'''
	Transforms training dataframes to numpy arrays. 

	:param df: highway train dataframe or KNMI dataframe
	:param dataset_name: either 'highway' or 'KNMI', depending on dataframe in first argument
	:returns: four numpy arrays containing the images, targets, filepaths and meteorological variables
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

def df_to_test_dict(main_df, test_dir, filename_test):
	'''
	Loads the testing data to a dictionary of numpy arrays (images, targets, filepaths, meteorological variables).

	:param main_df: a main dataframe 
	:param test_dir: test directory location
	:param filename_test: Filename for the test array.
	:returns: dictionary containing the four numpy arrays
	'''

	# Used for storing filename:label pairs
	image_list_test, target_list_test, filepath_list_test, meteo_list_test = [], [], [], []

	# This opens all labeled files and finds the corresponding pictures and labels
	with open(test_dir + filename_test) as filestream:
	
	    for row in filestream:
	        row = row.strip().split(',')
	        filename = row[0]
	        label = row[1]
	        
	        datapoint_df = main_df[main_df['filepath'] == filename]
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
	test_features, test_targets = np.asarray(image_list_test), np.asarray(target_list_test)
	test_filepaths, test_meteo = np.asarray(filepath_list_test), np.asarray(meteo_list_test)
	test_meteo = test_meteo.reshape(len(test_meteo), 4)
	test_dict = {'images': test_features, 'targets': test_targets, 'filepaths': test_filepaths, 'meteo': test_meteo}

	return test_dict


def change_labels(target_array, filepath_array, filepath_changes):
	'''
	Changes labels in a targets array. These labels were manually decided by inspection of images.

	Returns: Changed target array.

	:param target_array: Array containing targets
	:param filepath_array: Array containing filepaths for matching rows in file
	:param filepath_changes: Filepath to csv containing the changed labels
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

	Returns: All input arrays with bad indices ommitted.

	:param KNMI_images: KNMI image array
	:param KNMI_targets: KNMI targets array
	:param KNMI_filepaths: KNMI filepaths array
	:param KNMI_meteo: KNMI array containing meteorological variables
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

	Returns: All input arrays with bad indices ommitted.

	:param highway_images: Numpy array containing highway images
	:param highway_targets: Numpy array containing highway targets
	:param highway_filepaths: Numpy array containing highway filepaths
	:param highway_meteo: Numpy array containing highway meteo variables
	:param ommitance_filepath: Contains paths of images that have to be ommitted	
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

def standardize_meteo(meteo_numpy):
	'''
	Standardizes meteorological variables. Centers values with mean 0 and standard deviation 1.
	Done independently for every variables.

	Returns: standardized meteorological variables numpy array of same shape as input array.

	:param meteo_numpy: Numpy array containing meteorological variables
	'''
	# Put numpy array into dataframe and drop rows containing NaNs
	standardize_df = pd.DataFrame(meteo_numpy, columns=[0, 1, 2, 3])
	standardize_df = standardize_df.dropna(axis=0)

	# Scale data with mean centered around 0 and standard deviation 1
	std_scale = preprocessing.StandardScaler().fit(standardize_df[[0, 1, 2, 3]])
	highway_meteo = std_scale.transform(standardize_df[[0, 1, 2, 3]])

	print('Standardized meteorological variables')

	return highway_meteo

def create_highway_df(main_df):
	'''
	Get all the highways camera's that are closest to Schiphol and Eelde airport.

	Returns: Highway image dataframe with images that were taken within 7.500 meter distance
	of Eelde and Schiphol airport.

	:param main_df: Main dataframe containing all the images 
	'''

	schiphol_df = main_df[main_df['MeteoStationLocationID'] == 542]
	schiphol_highways = schiphol_df[schiphol_df['location_name']!= 'SCHIPHOL airport']
	eelde_df = main_df[main_df['MeteoStationLocationID'] == 506]
	eelde_highways = eelde_df[eelde_df['location_name'] != 'EELDE airport']
	highway_df = pd.concat([eelde_highways, schiphol_highways])

	# Make sure to only use camera's within 7.500 meters of airport sensors
	highway_df = highway_df[highway_df['distanceInMeters'] < 7500]

	return highway_df

def split_highway(highway_images, highway_targets, highway_filepaths, highway_meteo):
	'''
	Splits the labeled highway dataset into two separate datasets: Train dataset and validation set.
	5 camera's are used for validation set. Other camera's are used for training.

	Returns: Two dictionaries: First contains train highway arrays, second contains validation highway arrays.

	:param highway_images: Numpy array with highway images
	:param highway_images: Numpy array with target images
	:param highway_filepaths: Numpy array with the filepathss
	:param highway_meteo: Numpy array with the meteorological variables
	'''

	# Before splitting, normalize meteo variables over the whole dataset
	highway_meteo = standardize_meteo(highway_meteo)

	# List of camera's to use as validation set
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

	# Make dictionaries from the numpy arrays and return
	highway_train_dict = {'images': X_train_highway, 'targets': y_train_highway, 'filepaths': paths_train_highway, 'meteo': meteo_train_highway}
	highway_val_dict = {'images': X_validation, 'targets': y_validation, 'filepaths': paths_validation, 'meteo': meteo_validation}

	return highway_train_dict, highway_val_dict

def standardize_meteo(meteo_np):
	'''
	Standardizes the values for the four meteorological variables. If standardizing highway dataset:
	Call right before splitting into train/validation sets.

	Retunrs: Standardized meteo array

	:param meteo_np: Numpy array containing meteo variable values. Each column being one variable.
	'''

	standardize_df = pd.DataFrame(meteo_np, columns=[0, 1, 2, 3])
	standardize_df = standardize_df.dropna(axis=0)
	std_scale = preprocessing.StandardScaler().fit(standardize_df[[0, 1, 2, 3]])
	standardized_meteo = std_scale.transform(standardize_df[[0, 1, 2, 3]])

	return standardized_meteo

def process_raw():

	# Necessary dirs and paths to files
	PROCESSED_DIR = 'processed/'
	TEST_IMAGE_DIR = '/Volumes/TIMKNMI/KNMIPictures/RWS/'
	KNMI_relabel_path = 'helpers/knmichangelabels'
	highway_relabel_path = 'helpers/trainhighwaylabels'
	highway_ommit_path = 'helpers/ommitancehighway'
	test_image_filename = 'TestImages.txt'

	# Read the semi-processed data
	main_df = pd.read_pickle('semi-processed/all_info_df')

	# Get KNMI df and highway df
	KNMI_names = ['De Bilt (260_A_a)', 'Cabauw (348)', 'BEEK airport', 'EELDE airport', 'ROTTERDAM airport', 'SCHIPHOL airport']
	KNMI_df = main_df[main_df['location_name'].isin(KNMI_names)][:500]
	highway_df = create_highway_df(main_df)[:500]

	# Load KNMI df and highway df to numpy arrays
	KNMI_images, KNMI_targets, KNMI_filepaths, KNMI_meteo = df_to_numpy_train(KNMI_df, 'KNMI')
	highway_images, highway_targets, highway_filepaths, highway_meteo = df_to_numpy_train(highway_df, 'highway')

	highway_images = np.load('/Volumes/TIMPP/UnusedKNMI/numpyfiles/meteo/highway/change/highway_images.npy')
	highway_targets = np.load('/Volumes/TIMPP/UnusedKNMI/numpyfiles/meteo/highway/change/highway_targets.npy')
	highway_filepaths = np.load('/Volumes/TIMPP/UnusedKNMI/numpyfiles/meteo/highway/change/highway_filepaths.npy')
	highway_meteo = np.load('/Volumes/TIMPP/UnusedKNMI/numpyfiles/meteo/highway/change/highway_meteo.npy')

	highway_images = highway_images[:10000]
	highway_targets = highway_targets[:10000]
	highway_filepaths = highway_filepaths[:10000]
	highway_meteo = highway_meteo[:10000]

	print(np.bincount(KNMI_targets.astype(int)))
	print(np.bincount(highway_targets.astype(int)))

	# Relabel the KNMI and highway targets
	KNMI_targets = change_labels(KNMI_targets, KNMI_filepaths, KNMI_relabel_path)
	highway_targets = change_labels(highway_targets, highway_filepaths, highway_relabel_path)

	print(np.bincount(KNMI_targets.astype(int)))
	print(np.bincount(highway_targets.astype(int)))

	# Ommit labels of KNMI and highway datasets
	# KNMI_images, KNMI_targets, KNMI_filepaths, KNMI_meteo = ommit_labels_KNMI(KNMI_images, KNMI_targets, KNMI_filepaths, KNMI_meteo)
	# highway_images, highway_targets, highway_filepaths, highway_meteo = ommit_labels_highway(highway_images, highway_targets, highway_filepaths, highway_meteo, highway_ommit_path)

	# Split the highway dataset into training and validation
	highway_train_dict, highway_val_dict = split_highway(highway_images, highway_targets, highway_filepaths, highway_meteo)

	# Create KNMI dict
	KNMI_dict = {'images': KNMI_images, 'targets': KNMI_targets, 'filepaths': KNMI_filepaths, 'meteo': KNMI_meteo}

	# Load the test dictionaries
	test_dict = df_to_test_dict(main_df, TEST_IMAGE_DIR, test_image_filename)

	# Save the arrays into numpy dictionaries
	np.save(PROCESSED_DIR + 'KNMI.npy', KNMI_dict)
	np.save(PROCESSED_DIR + 'highway_train.npy', highway_train_dict)
	np.save(PROCESSED_DIR + 'highway_val.npy', highway_val_dict)
	np.save(PROCESSED_DIR + 'test.npy', test_dict)

	print('Done! Processed the data to a dictionary of numpy arrays.')

process_raw


