import numpy as np
import pandas as pd
import seaborn as sns
import os
import torch
import torchvision
import torchsample
import psycopg2
import random
import re
import time
import csv
import copy
from functools import reduce 
from datetime import datetime

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import models
from torchvision import transforms
from torchsample import transforms as ts_transforms
from matplotlib import pyplot as plt
from PIL import Image
from geopy.distance import vincenty 
from scipy.ndimage import imread
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score

from mpl_toolkits.mplot3d import Axes3D
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

def adjust_filepath(filepath):
	'''
	Used for getting the right filepaths to match the filepath structure in directory.

	:param filepath: Any filepath in fog database pandas dataframe.
	'''

	# Changes filepaths of non-highway images
	regex = re.compile(r'\d[/].*$')
	search = re.search(regex, filepath)
	jpg_name = search.group(0)[2:]

	# Second search for filepaths of highways
	regex2 = re.compile(r'[A]\d*-.*')
	search2 = re.search(regex2, jpg_name)

	# Only do this if filepath is highway. So not being None
	if search2 != None:
		jpg_name = search2.group(0)
		return str(jpg_name)

	return jpg_name

def add_missing_windspeed(filepath, df_meteo):
	'''
	Adds missing windspeed values for Schiphol, Rotterdam and Eelde airport.

	:param filepath: Filepath to the helper file that contains missing windspeed values.
	'''

	with open(filepath, 'r') as file:

		# Skip over header
		csv_read = csv.reader(file)
		next(csv_read)

		c = 0

		for row in csv_read:
			if row[1] == '240_W_18Cm27':
				location_name = 'SCHIPHOL airport'
				meteo_id = 542
			elif row[1] == '344_W_06t':
				location_name = 'ROTTERDAM airport'
				meteo_id = 536
			elif row[1] == '380_W_04t':
				location_name = 'EELDE airport'
				meteo_id = 506

			# Right date format
			date = row[0][:15].replace('_', '')

			# Hour '24' has to change to '00' for the datetime function not to error
			if date[8:10] == '24':
				date = date[:8] + '00' + date[10:]

			d_time = datetime.strptime(date, '%Y%m%d%H%M%S')

			# Get dataframe rows that match the specific datetime and location
			indices = df_meteo[(df_meteo['datetime'] == d_time) &
							  (df_meteo['MeteoStationLocationID'] == meteo_id)].index

			for idx in indices:
				try:
					c += 1
					df_meteo.at[idx, 'wind_speed'] = row[2]
				except:
					print(location_name, d_time)

				if c % 10000 == 0:
					print('Added {} wind speed values'.format(c))

	return df_meteo

def add_missing_temp_hum(filepath, df_meteo):
	'''
	Adds the missing temperature and humidity values for Schiphol.

	:param filepath: Filepath to helper file that contains missing temperature and humidity values.
	'''

	with open(filepath, 'r') as file:
		print('ey wollah')
		# Skip header row
		csv_read = csv.reader(file)
		next(csv_read)

		for c, row in enumerate(csv_read):

			# Variables to fill and MeteoStation ID Schiphol
			air_temp = row[2]
			rel_hum = row[4]
			meteo_id = 542

			date = row[0][:15].replace('_', '')

			# Hour '24' has to change to '00' for the datetime function not to error
			if date[8:10] == '24':
				date = date[:8] + '00' + date[10:]

			d_time = datetime.strptime(date, '%Y%m%d%H%M%S')

			# Get dataframe rows that match the specific datetime and location
			indices = df_meteo[(df_meteo['datetime'] == d_time) &
										(df_meteo['MeteoStationLocationID'] == meteo_id)].index

			for idx in indices:

				try:
					df_meteo.at[idx, 'air_temp'] = air_temp
					df_meteo.at[idx, 'rel_humidity'] = rel_hum
				except:
					continue

				if c % 10000 == 0:
					print('Added {} temperature and humidity values'.format(c))

	return df_meteo

def ordinal_visibility(mor_vis):
	'''
	Changes the ratio variable of MOR-visibility to ordinal scale.

	:param mor_vis: Ratio MOR-visibility value.
	'''
	if mor_vis > 1000:
		return 0
	elif mor_vis < 250:
		return 2
	elif mor_vis >= 250 and mor_vis <= 1000:
		return 1

def db_connect(config_path):
	'''
	Establish connection to fog database. Returns cursor object.

	:param config_path: Filepath to configuration file for the database.
	'''


	# Opens config file for connecting to database
	with open(config_path, mode='r') as infile:
		reader = csv.reader(infile)
		config = {rows[0]:rows[1] for rows in reader}

		# Credentials fo AWS database
		dsn_database = config['database']
		dsn_hostname = "fogdb.cmse2tqlcvsl.eu-west-1.rds.amazonaws.com"
		dsn_port = "9418"
		dsn_uid = config['\ufeffusername']
		dsn_pwd = config ['password']

		try:
			conn_string = "host="+dsn_hostname+" port="+dsn_port+" dbname="+dsn_database+" user="+dsn_uid+" password="+dsn_pwd
			print("Connected to database:\n{}\n".format(conn_string))
			conn=psycopg2.connect(conn_string)
		except:
			print ("\nNo connection to the database could be established.")

		cursor = conn.cursor()

	# Return cursor object
	return cursor

def perform_IDW(IDW_df, main_df):
	'''
	Performs Inverse Distance Weighting.

	:param df: Should be the main dataframe after
	'''

	print('Starting Inverse Distance Weighting')

	unique_dates = IDW_df['datetime'].unique()
	num_dates = len(unique_dates)
	variable_list = ['air_temp', 'dew_point', 'rel_humidity', 'wind_speed']

	# List of optimized hyperparameters for the IDW. See notebook 'data manipulation' for this
	hyper_IDW = {'air_temp' : {'k' : 9, 'p' : 1.0}, 'dew_point' : {'k' : 5, 'p' : 1.5},
				'rel_humidity' : {'k' : 8, 'p' : 1.0}, 'wind_speed' : {'k' : 14, 'p' : 1.0}}

	for c, date in enumerate(unique_dates):

		for variable in variable_list:

			meteo_stations = IDW_df[IDW_df['datetime'] == date]
			meteo_stations = meteo_stations[meteo_stations[variable].notnull()]

			x = meteo_stations[['lat', 'lon']].as_matrix()
			y = meteo_stations[variable]

			date_df = main_df[main_df['datetime'] == date]
			date_df = date_df[date_df[variable].isnull()]

			for idx, row in date_df.iterrows():

				coords_camera = np.asarray(row[['lat', 'long']])
				interpolated = IDW(coords_camera, x, y, k=hyper_IDW[variable]['k'], p=hyper_IDW[variable]['p'])
				main_df.at[idx, variable] = interpolated

		if c % 1000 == 0:
			print('Change for {} of the {} unique dates'.format(c, num_dates))

def fetch_primary_dataframes(cursor, distance_filepath):
	'''
	This loads two primary dataframes. It returns 1. dataframe containing cameras and closest meteorological stations
	2. dataframe containing the meteorological variables and MOR visibility per images.

	:param cursor: Cursor object obtained by running db_connect function.
	:param distance_filepath: Filepath to csv containing distance between cameras and nearest meteorological stations.
	'''

	# Distance to nearest meteo stations for all cameras
	distance_df = pd.read_csv(distance_filepath)
	print('Loaded the distance df\n')
	# Get the images in dataframe
	cursor.execute("SELECT * FROM images WHERE day_phase = '1'")
	img_df = pd.DataFrame(cursor.fetchall(), columns=['img_id', 'camera_id', 'datetime', 'filepath', 'day_phase'])
	img_df['filepath'] = img_df['filepath'].apply(adjust_filepath, 1)
	print('Loaded images and adjusted filepaths\n')
	# Fetch all the camera/location id pairs and put into df
	cursor.execute("SELECT * FROM cameras")
	df_cameras = pd.DataFrame(cursor.fetchall(), columns=['camera_id', 'location_id', 'cam_description', 'camera_name'])
	print('Got cameras \n')
	# Get the meteorological features
	cursor.execute("SELECT * FROM meteo_features_copy")
	df_meteo_features = pd.DataFrame(cursor.fetchall(), columns=['key','MeteoStationLocationID', 'datetime',
																 'wind_speed', 'rel_humidity', 'air_temp', 'dew_point',
																'mor_vis'])
	print('beforemerge')
	# Merge image df with the cameras df and then with distance df
	merged_image_cameras = pd.merge(img_df, df_cameras, on='camera_id')
	merged_nearest = pd.merge(merged_image_cameras, distance_df, on='location_id')

	print('Succesfully fetched the primary dataframes\n')

	return merged_nearest, df_meteo_features

def create_main_df(df_meteo, merged_nearest, cursor):
	'''
	Creates the main dataframe from the two primary dataframes.

	:param df_meteo: Primary dataframe containing.
	'''
	# Load locations df
	cursor.execute("SELECT * FROM locations")
	locations_df = pd.DataFrame(cursor.fetchall(), columns=['location_id', 'location_name', 'long', 'lat'])

	# Meteo features of closest meteo station are linked to every image
	main_df = pd.merge(merged_nearest, df_meteo, on=['MeteoStationLocationID', 'datetime'])
	main_df['visibility'] = main_df['mor_vis'].apply(ordinal_visibility, 1)

	# Make sure that meteo variables above 7500m distance are nan. We want to fill these using IDW
	main_df.loc[main_df.distanceInMeters > 7500, ['wind_speed', 'rel_humidity', 'air_temp', 'dew_point']] = np.nan

	main_df = pd.merge(locations_df, main_df, on='location_id')

	# Drop unnecessary columns
	retain_columns = ['location_id', 'long', 'lat', 'location_name', 'camera_id', 'datetime', 'filepath', 'MeteoStationLocationID',
					 'camera_name', 'meteo_station_id', 'meteo_station_name', 'distanceInMeters', 'wind_speed',
					 'rel_humidity', 'air_temp', 'dew_point', 'mor_vis', 'visibility']
	main_df = main_df[retain_columns]

	print('Succesfully created the main dataframe\n')

	return main_df


def retrieve_IDW_df(IDW_directory):

	# Load dataframes that have the meteorological variables/stations in them
	IDW_dew_df = pd.read_csv(IDW_directory + 'dewpoint.csv').dropna()
	IDW_temp_df = pd.read_csv(IDW_directory + 'temperature.csv').dropna()
	IDW_humidity_df = pd.read_csv(IDW_directory + 'humidity.csv').dropna()
	IDW_windspeed_df = pd.read_csv(IDW_directory + 'windspeed.csv').dropna()

	# Merge IDW dfs
	IDW_dfs = [IDW_dew_df, IDW_temp_df, IDW_humidity_df, IDW_windspeed_df]
	IDW_final_df = reduce(lambda left,right: pd.merge(left,right,on=['DS_CODE', 'IT_DATETIME'], how='outer'), IDW_dfs)

	# Change DS_CODE, IT_DATETIME and meteo coliumn names
	IDW_final_df['DS_CODE'] = IDW_final_df['DS_CODE'].apply(replace_DS, 1)
	IDW_final_df['datetime'] = IDW_final_df['IT_DATETIME'].apply(change_date, 1)
	IDW_final_df = IDW_final_df.rename(columns={'TOT.T_DEWP_10': 'dew_point', 'TOT.T_DRYB_10': 'air_temp',
				   'TOT.U_10' : 'rel_humidity', 'TOW.FF_10M_10' : 'wind_speed'})

	return IDW_final_df

def link_IDW_locations(link_filepath, IDW_df):

	# Load df that contains latitude/longitude information for stations in IDW_final_df
	link_df = pd.read_csv(link_filepath, error_bad_lines=False, delimiter=';')
	link_df.rename(columns = {'KISID':'DS_CODE'}, inplace = True)
	link_df['DS_CODE'] = link_df['DS_CODE'].apply(replace_DS, 1)

	# Only need one row with lat/long information per station
	link_df = link_df.drop_duplicates(subset = 'DS_CODE', keep='last')

	# Link latitude and longitude to stations
	IDW_df = pd.merge(IDW_df, link_df, on='DS_CODE')
	IDW_df = IDW_df[IDW_df['lat'] != 0]

	return IDW_df

def change_cabauw_bilt_paths(filepath):
	'''
	Changes the filepaths of old de Bilt images and of Cabauw images.

	:param filepath: Either a de Bilt or Cabauw image filepath.
	'''

	if 'nobackup/users/' in filepath:
		return filepath[-29:]
	elif 'CABAUW' in filepath:
		return filepath[7:]

def IDW(coords_camera, coords_stations, y, p=2, k=3):
    '''
    Interpolate meteorological variable value for a camera using inverse distance weighting.
    
    :param coords_camera: Coordinates for camera to get interpolated meteo value for.
    :param coords_stations: Array containing [k, [latitude, longitude]] of stations with known meteo values.
    :param p: Hyperparameter that determines the decay of weight given to distant points (should be between 1 and 3).
    :param k: Number of k nearest stations to coords_camera to use for estimating y.
    '''

   	# Determine distance from meteorological stations and select the k closest stations
    distance_from_meteos = [vincenty(coords_camera, coords_station).km for coords_station in coords_stations]
    indices = np.argpartition(np.array(distance_from_meteos),k)[:k].tolist()
    closest_k_dist = [distance_from_meteos[idx] for idx in indices]

    # Get values for these stations
    values = [list(y)[idx] for idx in indices]

    # Intepolate value for camera
    numerator = [value / (distance ** p) for distance, value in zip(closest_k_dist, values)]
    denominator = [1 / (distance ** p) for distance in closest_k_dist]
    y_pred = np.sum(numerator) / np.sum(denominator)
    
    return y_pred

def replace_DS(ds_code):
    '''
    Makes DS_CODE numerical to match the 'KISID' in the dataframe that links meteorological dataframe to 
    long/lat values.
    
    :param ds_code: DS_CODE out of 
    '''
    
    try:
        number = re.search(r'\d*', ds_code)
        number = number.group(0)
        return number
    
    except:
        return 'NaN'

def change_date(date_string):
    '''
    Changes string date in the IDW df to a datetime object.
    
    :param date_string: String date.
    '''
    date_string = date_string[:15]
    if date_string[9:11] == '24':
        date_string = date_string[:9] + '00' + date_string[11:]
    
    date_time = datetime.strptime(date_string, '%Y%m%d_%H%M%S')
    return date_time

def main():

	# Paths
	config_path = 'helpers/DBconfig.csv'
	distance_path = 'helpers/distanceKNMIStationsToLocations.csv'
	IDW_df_path = 'helpers/IDW/'
	linking_path = 'helpers/station_linking.csv'
	windspeed_path = 'helpers/windspeed.csv'
	temp_humidity_path = 'helpers/temp_humidity.csv'

	cursor = db_connect(config_path)

	# Get the primary dataframes
	merged_nearest_df, df_meteo_features = fetch_primary_dataframes(cursor, distance_path)

	# Add missing meterological variables to the meteo dataframe
	df_meteo_features = add_missing_windspeed(windspeed_path, df_meteo_features)
	df_meteo_features = add_missing_temp_hum(temp_humidity_path, df_meteo_features)

	# Create main dataframe
	main_df = create_main_df(df_meteo_features, merged_nearest_df, cursor)

	# Get the dataframe used for Inverse Distance Weighting and map locations to it
	IDW_df = retrieve_IDW_df(IDW_df_path)
	IDW_df = link_IDW_locations(linking_path, IDW_df)

	# Interpolate meteorological NaN values using inverse distance weighting
	main_df = perform_IDW(IDW_df, main_df)

	main_df['filepath'] = main_df['filepath'].apply(change_cabauw_bilt_paths, 1)

	# Exclude NaN's
	main_df = main_df[main_df['visibility'].notnull()]
	main_df = main_df[main_df['wind_speed'].notnull()]
	main_df = main_df[main_df['rel_humidity'].notnull()]
	main_df = main_df[main_df['dew_point'].notnull()]
	main_df = main_df[main_df['air_temp'].notnull()]

	# Pickle dataframe
	main_df.to_pickle('semi-processed/test_df')

if __name__ == '__main__':
    main()
