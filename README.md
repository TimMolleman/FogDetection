# Fog Detection Using Highway Cameras

This thesis project studied whether or not convolutional neural networks (CNNs) can be used for the purpose of classifying highway images on fog conditions. Additionally, it was researched if adding four meteorological variables (air temperature, relative humidity, dew point and wind speed)to the model resulted in a better performance.

## Data
To start training models, data has to be processed first. In the command line, navigate to the `data/` folder and run

```
python3 raw_to_semiprocessed.py
```

This acquires the raw data from the KNMI fog database and creates a pandas dataframe for the train/validation dataset images and a pandas dataframe for the test dataset images. After this script has finished run

```
python3 to_processed.py
```

This script takes the pandas dataframes, links the images to them and transforms them to several dictionaries that all contain four numpy arrays: One numpy array containing the images, a numpy array containing labels for the images, a numpy array containing filepaths of images and a numpy array containing the meteorological variables associated with the images.

## Model training
To train a model, navigate to the top-level directory `FogDetection/` and run

```
python3 main.py
```

In the `main.py` file, argparser is used for passing arguments to functions. If you want to specify a different model for training or change other arguments (e.g. Use meteoroligical variables for training, run on CUDA, number of epochs to run training), this can be done in this main file. After the model training is completed, the models are saved to '.pth.tar' files and are directly tested on the test dataset.
'