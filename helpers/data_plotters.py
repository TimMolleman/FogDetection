import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# Globals (adjust to your own purposes)
IMG_SIZE = 100
CHANNELS = 3
PLOT_DIR = '../img/plotss'


def distribution_meteo_variables(df, filepath):
    '''
    Plots the distribution of weather variables at the KNMI weather stations. No IDW/harmonie variables are used.

    :param df: either main dataframe or KNMI weather stations dataframe
    :param filepath: location to store distribution image
    '''

    # Selection lists with right criteria for plotting the meteo variables
    KNMI_names = ['De Bilt (260_A_a)', 'Cabauw (348)', 'BEEK airport', 'EELDE airport', 'ROTTERDAM airport', 'SCHIPHOL airport']
    meteo_variables = ['wind_speed', 'air_temp', 'rel_humidity', 'dew_point']
    plot_df = df[df['location_name'].isin(KNMI_names)]

    # Create limits for plotting variables
    clip = {'wind_speed': (0, 12), 'air_temp': (-1000,1000), 'rel_humidity': (90, 120), 'dew_point':(-1000, 1000)}
    
    # Plot layout
    fig = plt.figure(figsize=(15, 10))
    fig.tight_layout()
    
    # Distribution plot for every variable
    for i, variable in enumerate(meteo_variables):
    
        target_0 = plot_df.loc[plot_df['visibility'] == 0]
        target_1 = plot_df.loc[plot_df['visibility'] == 1]
        target_2 = plot_df.loc[plot_df['visibility'] == 2]
        
        # Make-up of plot
        fig.add_subplot(2, 2, i + 1)
        plt.title('Density plot of {}'.format(variable))
        plt.ylabel('Density')
        
        # Plot
        sns.distplot(target_0[variable], hist=False, kde=True, kde_kws = {'linewidth':3, 'clip': clip[variable], 'shade':True}, label='no fog')
        sns.distplot(target_1[variable], hist=False, kde=True, kde_kws = {'linewidth':3,'clip': clip[variable], 'shade':True}, label='light fog')
        sns.distplot(target_2[variable], hist=False, kde=True, kde_kws = {'linewidth':3,'clip': clip[variable], 'shade':True}, label='dense fog')
    
    fig.savefig(filepath)

def run_PCA(X_data, y_data):
        '''
        Gets 2-dimensional principal components for pixel data
        
        :param X_data: pixel values of images
        :param y_data: corresponding target values of images
        :return: dataframe for performing PCA analysis
        '''
        
        # Define and transform data
        pca = PCA(n_components=2)
        principal_components= pca.fit_transform(X_data.flatten().reshape(len(X_data), IMG_SIZE * IMG_SIZE * CHANNELS))
        
        # Create DF of PCA's and targets 
        PCA_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
        target_df = pd.DataFrame(data=y_data, columns=['target'])
        PCA_df = pd.concat([PCA_df, target_df['target']], axis=1)
        
        return PCA_df

def save_PCAfig(X_data, y_data, filepath):
    '''
    This saves a plot of a dataframe created with the run_PCA function
    
    :param X_data: pixel values of images
    :param y_data: corresponding target values of images
    :param filepath: location to store PCA image  
    '''

    # Get the PCA data
    PCA_df = run_PCA(X_data, y_data)

    # Create figure
    fig = plt.figure(figsize= (8,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    targets = [0, 1, 2]
    class_names= ['Clear', 'Light Fog', 'Dense Fog']
    colors = ['r', 'g', 'b']
    
    # Create plot on figure                  
    for target, color in zip(targets, colors):
        indicesToKeep = PCA_df['target'] == target
        ax.scatter(PCA_df.loc[indicesToKeep, 'PCA1'],
                   PCA_df.loc[indicesToKeep, 'PCA2'],
                   c = color,
                   s = 50)

        ax.legend(class_names)
        ax.grid()

    fig.savefig(filepath)

'''
Below are two examples. First one for plotting 'distribution_meteo_variables' and second one for
plotting PCA function. Uncomment and run and figure will show up in the figures folder. Note that for
performing the PCA, it is recommended to use a subset of the no-fog images. Otherwise it will most likely overflow memory capacity.
'''

# # Example meteo distribution plot (adjust pickle filename to your own)
# meteo_dist_df = pd.read_pickle('../data/semi-processed/filled_harmonie')
# dist_filename = 'dist_example.png'
# dist_filepath = PLOT_DIR+dist_filename
# distribution_meteo_variables(meteo_dist_df, dist_filepath)


# # Example PCA
# highway_train = np.load('../data/processed/highway_train.npy')[()]
# X = highway_train['images']
# y = highway_train['targets']
# PCA_filename = 'PCA_example.png'
# PCAfilepath = PLOT_DIR+PCA_filename

# save_PCAfig(X, y, PCAfilepath)



