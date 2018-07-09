# Fog Detection Using Highway Cameras

This thesis project studied whether or not convolutional neural networks (CNNs) can be used for the purpose of classifying highway images on fog conditions. 

## Data
To start training models, data has to be acquired first. In the command line, navigate to the XXX folder and run

'''
python3 raw_to_semiprocessed.py
'''

This acquires the raw data from the KNMI fog database and creates a pandas dataframe for the train/validation dataset images and a pandas dataframe for the test dataset images. After this script has finished run

'''
python3 to_processed.py
'''

This script takes the pandas dataframes, links the images to them and transforms them to several dictionaries that all contain four numpy arrays: One numpy array containing the images, a numpy array containing labels for the images, a numpy array containing filepaths of images and a numpy array containing the meteorological variables associated with the images.

## Model training
To train a model, navigate tot the top-level directory 

'''
python3 main.py
'''