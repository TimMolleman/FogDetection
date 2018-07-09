import numpy as np
import pandas as pd
import os
import psycopg2
import re
import csv
import argparse
import tempfile
import pygrib

from tempfile import TemporaryDirectory
from functools import reduce 
from datetime import timedelta
from datetime import datetime as dt
from PIL import Image
from geopy.distance import vincenty 
from scipy.ndimage import imread
from zipfile import ZipFile


parser = argparse.ArgumentParser(description='Raw data retrieval')

# Filepaths
parser.add_argument('--config_path', type=str, default='helpers/DBconfig.csv', help='path to the config file for fog accessing database')
parser.add_argument('--distance_path', type=str, default='helpers/distanceKNMIStationsToLocations.csv', help='path to file containing distances of cameras to nearest meteo stations')
parser.add_argument('--windspeed_path', type=str, default='helpers/windspeed.csv', help='path to file containing missing windspeed values')
parser.add_argument('--temperature_humidity_path', type=str, default='helpers/temp_humidity.csv', help='path to file containing missing temperature/relative humidity values')
parser.add_argument('--linking_path', type=str, default='helpers/station_linking.csv', help='path to file containing longitude/latitude locations of meteorological stations used for IDW interpolations')
parser.add_argument('--IDW_dfs_path', type=str, default='helpers/IDW/', help='path to directory containing dataframes needed for IDW interpolations')
parser.add_argument('--test_images_path', type=str, default='/Volumes/TIMKNMI/KNMIPictures/RWS/TestImagesRefined.txt', help='path to file with manually labeled test images')
parser.add_argument('--semi_processed_dir', type=str, default='semi-processed/', help='directory to which to save dataframes')

# Meteo-filling method
parser.add_argument('--missing_meteo_method', type=str, default='IDW', help='method for filling missing meteo variables. Either IDW or harmonie')

# New dataframe
parser.add_argument('--new_main_df', type=bool, default=True, help='decide if new main dataframe should be created')
parser.add_argument('--new_test_df', type=bool, default=True, help='decide if new test dataframe should be created')

args = parser.parse_args()

KNMI_LOCATIONS = ['De Bilt (260_A_a)', 'Cabauw (348)', 'BEEK airport', 'EELDE airport', 'ROTTERDAM airport', 'SCHIPHOL airport']

def db_connect(config_path):
	'''
	Establishes connection to KNMI fog database.

	:param config_path: filepath to configuration file for accessing the database
	:return: cursor object that can be used to fetch data from database
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

def fetch_primary_dataframes(cursor, distance_filepath):
	'''
	Load the two primary dataframes.

	:param cursor: cursor object obtained by running db_connect function
	:param distance_filepath: filepath to csv containing distances between cameras and nearest meteorological stations
	:return: 1. a dataframe containing cameras and their closest meteorological stations, 2. a dataframe containing
	the meteorological variables and MOR visibility for every image
	'''

	# Distance to nearest meteo stations for all cameras
	distance_df = pd.read_csv(distance_filepath)
	print('Loaded the distance dataframe')
	
	# Get the images in dataframe
	cursor.execute("SELECT * FROM images WHERE day_phase = '1'")
	img_df = pd.DataFrame(cursor.fetchall(), columns=['img_id', 'camera_id', 'datetime', 'filepath', 'day_phase'])
	img_df['filepath'] = img_df['filepath'].apply(adjust_filepath, 1)

	print('Loaded images and adjusted filepaths')

	# Fetch all the camera/location id pairs and put into df
	cursor.execute("SELECT * FROM cameras")
	df_cameras = pd.DataFrame(cursor.fetchall(), columns=['camera_id', 'location_id', 'cam_description', 'camera_name'])

	print('Got cameras dataframe')

	# Get the meteorological features
	cursor.execute("SELECT * FROM meteo_features_copy")

	df_meteo_features = pd.DataFrame(cursor.fetchall(), columns=['key','MeteoStationLocationID', 'datetime',
																 'wind_speed', 'rel_humidity', 'air_temp', 'dew_point',
																'mor_vis'])

	print('Now merging image dataframe with the distance dataframe')

	# Merge image df with the cameras df and then with distance df
	merged_image_cameras = pd.merge(img_df, df_cameras, on='camera_id')
	merged_nearest = pd.merge(merged_image_cameras, distance_df, on='location_id')

	print('Succesfully fetched the primary dataframes')

	return merged_nearest, df_meteo_features

def create_main_df(df_meteo, merged_nearest_df, cursor):
	'''
	Creates the main dataframe from the two primary dataframes.

	:param df_meteo: primary dataframe containing meteo variables and visibility labels for every image
	:param merged_nearest_df: dataframe containing information on cameras and their closest meteorological stations
	:param cursor: cursor object obtained by running db_connect function
	:return: main dataframe
	'''
	# Load locations df
	cursor.execute("SELECT * FROM locations")
	locations_df = pd.DataFrame(cursor.fetchall(), columns=['location_id', 'location_name', 'long', 'lat'])

	# Meteo features of closest meteo station are linked to every image
	main_df = pd.merge(merged_nearest_df, df_meteo, on=['MeteoStationLocationID', 'datetime'])
	main_df['visibility'] = main_df['mor_vis'].apply(ordinal_visibility, 1)

	# Get locations for main_df rows
	main_df = pd.merge(locations_df, main_df, on='location_id')

	# Only keep meteorological variable values for cameras next to visibility sensors. Others will be filled with HARMONIE or IDW
	main_df.loc[~main_df.location_name.isin(KNMI_LOCATIONS), ['wind_speed', 'rel_humidity', 'air_temp', 'dew_point']] = np.nan

	# Drop unnecessary columns
	retain_columns = ['location_id', 'long', 'lat', 'location_name', 'camera_id', 'datetime', 'filepath', 'MeteoStationLocationID',
					 'camera_name', 'meteo_station_id', 'meteo_station_name', 'distanceInMeters', 'wind_speed',
					 'rel_humidity', 'air_temp', 'dew_point', 'mor_vis', 'visibility']
	main_df = main_df[retain_columns]

	print('Succesfully created the main dataframe')

	return main_df, locations_df

def adjust_filepath(filepath):
	'''
	Used for getting the right filepaths to match the filepath structure in image directory.

	:param filepath: any filepath in fog database pandas dataframe.
	:return: right format for filepath to match directory filenames
	'''

	# Changes filepaths of non-highway images
	regex = re.compile(r'\d[/].*$')
	search = re.search(regex, filepath)
	jpg_name = search.group(0)[2:]

	# Second search for filepaths of highways
	regex2 = re.compile(r'[A]\d*-.*')
	search2 = re.search(regex2, jpg_name)

	# Only do this if filepath is highway. So not None
	if search2 != None:
		jpg_name = search2.group(0)
		return str(jpg_name)

	return jpg_name

def change_cabauw_bilt_filepaths(filepath):
	'''
	Changes the filepaths of old de Bilt images and of Cabauw images.

	:param filepath: either a de Bilt or Cabauw image filepath
	:return: changed filepath
	'''

	if 'nobackup/users/' in filepath:
		return filepath[-29:]
	elif 'CABAUW' in filepath:
		return filepath[7:]
	else:
		return filepath

def add_missing_windspeed(filepath, df_meteo):
	'''
	Adds missing windspeed values for Schiphol, Rotterdam and Eelde airport.

	:param filepath: filepath to the helper file that contains missing windspeed values
	:param df_meteo: dataframe with missing windspeed values for Schiphol, Rotterdam and Eelde airport
	:returns: dataframe containing the filled missing windspeed values
	'''

	with open(filepath, 'r') as file:

		# Skip over header
		csv_read = csv.reader(file)
		next(csv_read)

		for c, row in enumerate(csv_read):
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

			d_time = dt.strptime(date, '%Y%m%d%H%M%S')

			# Get dataframe rows that match the specific datetime and location
			indices = df_meteo[(df_meteo['datetime'] == d_time) &
							  (df_meteo['MeteoStationLocationID'] == meteo_id)].index

			for idx in indices:
				try:
					df_meteo.at[idx, 'wind_speed'] = row[2]
				except:
					print(location_name, d_time)

				if c % 10000 == 0:
					print('Added {} wind speed values'.format(c))

	return df_meteo

def add_missing_temp_hum(filepath, df_meteo):
	'''
	Adds the missing temperature and humidity values for Schiphol.

	:param filepath: filepath to helper file that contains missing temperature and relative humidity values
	:param df_meteo: dataframe with missing temperature and humidity for Schiphol, Rotterdam and Eelde airport
	:return: dataframe with filled missing temperature and relative humidity values
	'''

	with open(filepath, 'r') as file:
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

			d_time = dt.strptime(date, '%Y%m%d%H%M%S')

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

	:param mor_vis: ratio MOR-visibility value
	:return: ordinal visibility label
	'''
	if mor_vis > 1000:
		return 0
	elif mor_vis < 250:
		return 2
	elif mor_vis >= 250 and mor_vis <= 1000:
		return 1

def drop_null_rows(main_df):
	'''
	Drops rows in main dataframe that contain NaN values for visibility or meteorological variables.

	:param main_df: main dataframe just before pickling
	:return: main_df with rows containing NaN's excluded
	'''
	# Drop NaN's
	main_df = main_df[main_df['visibility'].notnull()]
	main_df = main_df[main_df['wind_speed'].notnull()]
	main_df = main_df[main_df['rel_humidity'].notnull()]
	main_df = main_df[main_df['dew_point'].notnull()]
	main_df = main_df[main_df['air_temp'].notnull()]

	return main_df


def perform_IDW(IDW_df, main_df):
	'''
	Performs Inverse Distance Weighting interpolation.

	:param IDW_df: dataframe containing meteorological stations not in main dataframe that is used for interpolating main dataframe values
	:param main_df: main dataframe
	:return: main dataframe with filled meterological variable values
	'''

	print('Starting Inverse Distance Weighting')

	unique_dates = IDW_df['datetime'].unique()
	num_dates = len(unique_dates)
	variable_list = ['air_temp', 'dew_point', 'rel_humidity', 'wind_speed']

	# List of optimized hyperparameters for the IDW. See notebook 'data manipulation' for this
	hyper_IDW = {'air_temp' : {'k' : 9, 'p' : 1.0}, 'dew_point' : {'k' : 5, 'p' : 1.5},
				'rel_humidity' : {'k' : 8, 'p' : 1.0}, 'wind_speed' : {'k' : 14, 'p' : 1.0}}

	# Loop over unique dates
	for c, date in enumerate(unique_dates):

		# IDW should be performed for each variable 
		for variable in variable_list:

			# Get the right dates and variable values from IDW_df to use for interpolation 
			meteo_stations = IDW_df[IDW_df['datetime'] == date]
			meteo_stations = meteo_stations[meteo_stations[variable].notnull()]

			# Get lat/lons and variable vallues
			x = meteo_stations[['lat', 'lon']].as_matrix()
			y = meteo_stations[variable]

			# Get the main dataframe rows for which IDW has to be applied
			date_df = main_df[main_df['datetime'] == date]
			date_df = date_df[date_df[variable].isnull()]

			# Perform IDW fore very row in date_df
			for idx, row in date_df.iterrows():

				coords_camera = np.asarray(row[['lat', 'long']])
				interpolated = IDW(coords_camera, x, y, k=hyper_IDW[variable]['k'], p=hyper_IDW[variable]['p'])
				main_df.at[idx, variable] = interpolated

		if c % 1000 == 0:
			print('Change for {} of the {} unique dates'.format(c, num_dates))

	return main_df


def retrieve_IDW_df(IDW_directory):
	'''
	Retrieves the IDW df used for IDW interpolation.

	:param IDW_directory: directory with csv's that contain values for the variables that are interpolated using IDW
	:return: dataframe containing values of meteorological variables for 30-something meteorological stations
	'''

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

def link_IDW_locations(link_csv_filepath, IDW_df):
	'''
	Links the IDW_df's DS_CODE column with KISID column of dataframe used for linking lat/lons and location names to IDW df.

	:param link_csv_filepath: filepath to CSV-file containing location information on KNMI meteorological stations
	:param IDW_df: dataframe obtained by running 'retrieve_IDW_df'
	:return: IDW_df with linked location information 
	'''

	# Load df that contains latitude/longitude information for stations in IDW_final_df
	link_df = pd.read_csv(link_csv_filepath, error_bad_lines=False, delimiter=';')
	link_df.rename(columns = {'KISID':'DS_CODE'}, inplace = True)
	link_df['DS_CODE'] = link_df['DS_CODE'].apply(replace_DS, 1)

	# Only need one row with lat/long information per station
	link_df = link_df.drop_duplicates(subset = 'DS_CODE', keep='last')

	# Link latitude and longitude to stations
	IDW_df = pd.merge(IDW_df, link_df, on='DS_CODE')
	IDW_df = IDW_df[IDW_df['lat'] != 0]

	return IDW_df

def IDW(coords_camera, coords_stations, y, p=1, k=5):
	'''
	Interpolate meteorological variable value for a camera using inverse distance weighting.
	
	:param coords_camera: coordinates for camera to get interpolated meteo value for
	:param coords_stations: array containing [k, [latitude, longitude]] of stations with known meteo values
	:param p: hyperparameter that determines the decay of weight given to distant points (should be between 1 and 3)
	:param k: number of k nearest stations to coords_camera to use for estimating y
	:return: interpolated value for a meteorological variable
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
	
	:param ds_code: DS_CODE of IDW_df row
	:return: numerical DS_CODE
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
	
	:param date_string: string date
	:return: datetime object
	'''
	date_string = date_string[:15]
	if date_string[9:11] == '24':
		date_string = date_string[:9] + '00' + date_string[11:]
	
	date_time = dt.strptime(date_string, '%Y%m%d_%H%M%S')

	return date_time

def fill_harmonie(df):
	'''
	Fills missing meteorological values in a dataframe with HARMONIE model values.

	:param df: either main dataframe or dataframe containing test data
	:return: dataframe with filled meteorological values
	'''

	unique_dates = df['datetime'].unique()
	num_dates = len(unique_dates)

	for c, one_date in enumerate(unique_dates):

		# Get df for unique date and extract date
		date_df = df[df['datetime'] == one_date]

		# Convert date to string for getting zipfile
		t = pd.to_datetime(str(one_date)) 
		timestring = t.strftime('%Y%m%d%H')

		# Get the zipfile path and the right date for getting gribfile in ZIP
		zip_path, zip_date, grib_hour = get_zipfile_path(timestring)
		
		try:
			grib = open_grib(zip_path, zip_date, grib_hour)
			
			# Obtain grid point positions and meteo variables
			lats, lons, temps, rel_hums, windspeeds = get_meteo_grids(grib)

			for idx, row in date_df.iterrows():

				# Get the closest lat/lon index of grid
				obs_lat, obs_lon = row['lat'], row['long']
				closest_idx = get_latlon_idx(lats, lons, obs_lat, obs_lon)

				# Get the variable values
				temp, rel_hum, windspeed = temps.flat[closest_idx], rel_hums.flat[closest_idx], windspeeds.flat[closest_idx]
				temp = float(temp) - 272.15
				dew_point = calculate_dewpoint(rel_hum, temp)

				if np.isnan(row['air_temp']):
					df.at[idx, 'air_temp'] = temp
				if np.isnan(row['rel_humidity']):
					df.at[idx, 'rel_humidity'] = rel_hum
				if np.isnan(row['wind_speed']):
					df.at[idx, 'wind_speed'] = windspeed
				if np.isnan(row['dew_point']):
					df.at[idx, 'dew_point'] = dew_point

		except:
			print('Grib-file unavailable at index: {}, for date: {}'.format(c, timestring))
			continue
	
	
		if c % 500 == 0:
			print('Iterated over {} of {} unique dates'.format(c, num_dates))

	print('Filled dataframe with using the HARMONIE forecasting model')
	
	return df

def get_zipfile_path(date):
	'''
	Gets a filepath for opening a zipfile containing a number of HARMONIE model GRIB-files. 

	:param date: date in string format used for opening right zipfile 
	:return: path to zipfile, the date for zipfile and the right hour for opening GRIB-file
	'''

	original_hour = date[-2:]
	
	# Change hour to closest HARMONIE model forecast 
	if int(original_hour) < 7:
		hour = '00'
	elif 6 < int(original_hour) < 13:
		hour = '06'
	elif 12 < int(original_hour) < 19:
		hour = '12'
	elif 18 < int(original_hour) < 24:
		hour = '18'
	
	# Get difference between forecast and inputted hour
	difference = int(original_hour) - int(hour)

	if difference >= 10:
		grib_hour = str(difference)
	else:
		grib_hour = '0{}'.format(str(difference))

	# Part of path to be used for opening zipfile
	zip_date = date[:-2]+hour
	
	# Get complete path to zipfile
	zipfile_name = 'HARM_{}_P1.zip'.format(zip_date)
	zip_path = '/Volumes/externe schijf/timWeatherModel/{}'.format(zipfile_name)
	
	return zip_path, zip_date, grib_hour

def open_grib(zip_path, zip_date, grib_hour):
	'''
	Opens a zipfile after which the right HARMONIE model GRIB-file is extracted from the opened file.

	:param zip_path: path to zipfile that contains needed GRIB-file
	:param zip_date: date of the zipfile
	:param grib_hour: hour that is used for opening the right GRIB-file
	:return: GRIB-file
	'''
	
	# Create temporary directory for storing the GRIB-files in the zipfile
	tDir = tempfile.mkdtemp('Harmonies')
	
	with TemporaryDirectory() as tmp_dir:
		
		# Open zipfile and get grib filepath
		archive = ZipFile(zip_path)

		grib_name = 'HA36_P1_{}00_0{}00_GB'.format(zip_date, grib_hour)

		# Extract right grib file from zip and bring to temporary directory
		extract_zip = archive.extract(grib_name, tmp_dir)
		gribfile = pygrib.open('{}/{}'.format(tmp_dir, grib_name))
		
	return gribfile

def get_latlon_idx(lats, lons, obs_lat, obs_lon):
	'''
	Gets index of HARMONIE grid-point that a camera in main dataframe row is closest to.

	:param lats: 300x300 grid containing latitudes
	:param lons: 300x300 gird containing longitudes
	:param obs_lat: observed latitude for a row in main dataframe
	:param obs_lon: observed longitude for a row in main dataframe
	:return: index of closest grid-point
	'''

	# Get the absolute distances between all grid points and observation point
	abslat = np.abs(lats - obs_lat)
	abslon = np.abs(lons - obs_lon)

	# Get the absolute distance between camera and every grid-point
	lat_index = np.argmin(abslat)
	lon_index = np.argmin(abslon)

	# Get closest grid-point
	c = np.maximum(abslon, abslat)
	latlon_idx = np.argmin(c)
	
	return latlon_idx

def calculate_dewpoint(RH, T):
	'''
	Calculates dewpoint given relative humidity and temperature.

	:param RH: relative humidity
	:param T: temperature
	:return: dewpoint value

	'''
	RH = RH * 100
	DP = 243.04*(np.log(RH/100)+((17.625*T)/(243.04+T)))/(17.625-np.log(RH/100)-((17.625*T)/(243.04+T))) + 0.08
	return DP

def get_meteo_grids(grib):
	'''
	Retrieves 300x300 grids for the meteorological variables and latitudes/longitudes.

	:param grib: GRIB-file
	:return: latitude/longitude grids, temperature gird, relative humidity grid and windspeed grid
	'''

	# Get variable values for date
	temps = grib.select(name='2 metre temperature')[0]
	rel_hums = grib.select(name='Relative humidity')[0]
	
	# Calculate the wind speed
	windcomponent_U = grib.select(name='10 metre U wind component')[0].values
	windcomponent_V = grib.select(name='10 metre V wind component')[0].values
	windspeeds = np.sqrt(windcomponent_U ** 2 + windcomponent_V **2)

	# Latitudes and longitudes are the same for every variable, because same grid is used
	lats, lons = temps.latlons()

	temps, rel_hums = temps.values, rel_hums.values
	
	return lats, lons, temps, rel_hums, windspeeds

def round_minutes(tm):
	'''
	Rounds a datetime down to 10 minutes.

	:param tm: datetime object
	:return: datetime rounded down to 10 minutes
	'''
	tm = tm - timedelta(minutes=tm.minute % 10,
								 seconds=tm.second,
								 microseconds=tm.microsecond)
	return tm

def get_test_df(test_filepath, merged_nearest, locations_df):
	'''
	Gets the test dataframe without meteorological variables filled.

	:param test_filepath: filepath to .txt file containing the test images paths and labels
	:param merged_nearest: dataframe containing images with merged nearest locations
	:param locations_df: dataframe containing location information
	:return: test df with empty meteorological variables
	'''
	with open(test_filepath) as filestream:
		test_filenames = []

		for row in filestream:
			row = row.strip().split(',')
			filename = row[0]
			test_filenames.append(filename)

	print(len(test_filenames))
	# Obtain test df without the meteo values filled
	test_loc_df = pd.merge(merged_nearest, locations_df, on='location_id')
	test_loc_df = test_loc_df[test_loc_df['filepath'].isin(test_filenames)]  
	test_loc_df['datetime'] = test_loc_df['datetime'].apply(round_minutes, 1)
	test_loc_df['wind_speed'], test_loc_df['rel_humidity'], test_loc_df['air_temp'], test_loc_df['dew_point'] = np.nan, np.nan, np.nan, np.nan

	return test_loc_df


def main():

	cursor = db_connect(args.config_path)

	# Get the primary dataframes
	merged_nearest_df, df_meteo_features = fetch_primary_dataframes(cursor, args.distance_path)

	# # Add missing meterological variables to the meteo dataframe
	df_meteo_features = add_missing_windspeed(args.windspeed_path, df_meteo_features)
	df_meteo_features = add_missing_temp_hum(args.temperature_humidity_path, df_meteo_features)

	# # Create main dataframe
	main_df, locations_df = create_main_df(df_meteo_features, merged_nearest_df, cursor)

	# main_df = pd.read_pickle('semi-processed/before_filling_meteo')
	main_df = main_df[:1000]
	print('first info')
	print(main_df.info())
	# Fill missing meteo values 
	if args.missing_meteo_method == 'IDW':
		
		# Get the dataframe used for Inverse Distance Weighting and map locations to it
		IDW_df = retrieve_IDW_df(args.IDW_dfs_path)
		IDW_df = link_IDW_locations(args.linking_path, IDW_df)

		# Interpolate meteorological NaN values using inverse distance weighting
		main_df = perform_IDW(IDW_df, main_df)

	elif args.missing_meteo_method == 'harmonie':
		main_df = fill_harmonie(main_df)

	else:
		raise ValueError("Specify either 'IDW' or 'harmonie' for missing_meteo_method") 

	# Change de Bilt/Cabauw filepaths and drop NaN rows 
	main_df['filepath'] = main_df['filepath'].apply(change_cabauw_bilt_filepaths, 1)
	main_df = drop_null_rows(main_df)

	# Pickle main dataframe
	if args.new_main_df:
		main_df.to_pickle('{}/main_dataframe_{}'.format(args.semi_processed_dir, args.missing_meteo_method))
		print('Saved main dataframe')

	# Get test dataframe
	test_df_nometeo = get_test_df(args.test_images_path , merged_nearest_df, locations_df)

	# Fill meteo values
	if args.missing_meteo_method == 'IDW':
		test_df = perform_IDW(IDW_df, test_df_nometeo)
	elif args.missing_meteo_method == 'harmonie':
		test_df = fill_harmonie(test_df_nometeo)

	# Pickle test dataframe
	if args.new_test_df:
		test_df.to_pickle('{}/test_df_{}'.format(args.semi_processed_dir, args.missing_meteo_method))
		print('Saved test dataframe')

	print('Finished running')

if __name__ == '__main__':
	main()
