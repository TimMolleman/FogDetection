import matplotlib as plt

def distribution_meteo_variables(df):
    '''
    Plots the distribution of weather variables at the KNMI weather stations. No interpolated/harmonie variables are used.

    :param df: Either main dataframe or KNMI weather stations dataframe.
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
        
        if variable == 'rel_humidity':
            plot_df = plot_df[plot_df[variable] > 95]
    
        target_0 = plot_df.loc[plot_df['visibility'] == 0]
        target_1 = plot_df.loc[plot_df['visibility'] == 1]
        target_2 = plot_df.loc[plot_df['visibility'] == 2]
        
        # Make-up
        fig.add_subplot(2, 2, i + 1)
        plt.title('Density plot of {}'.format(variable))
        plt.ylabel('Density')
        
        sns.distplot(target_0[variable], hist=False, kde=True, kde_kws = {'linewidth':3, 'clip': clip[variable], 'shade':True}, label='no fog')
        sns.distplot(target_1[variable], hist=False, kde=True, kde_kws = {'linewidth':3,'clip': clip[variable], 'shade':True}, label='light fog')
        sns.distplot(target_2[variable], hist=False, kde=True, kde_kws = {'linewidth':3,'clip': clip[variable], 'shade':True},label='dense fog')
    
    fig.savefig('../figures/meteo_distribution.png')



def plot_PCA(X_data, y_data):
    '''
    This plots a PCA dataframe created with the run_PCA function
    
    :param X_data: Pixel values of images
    :param y_data: Corresponding target values of images
    
    '''

    def run_PCA(X_data, y_data):
        '''
        Gets 2-dimensional principal components for pixel data
        
        :param X_data: Pixel values of images
        :param y_data: Corresponding target values of images
        '''
        
        # Define and transform data
        pca = PCA(n_components=2)
        principal_components= pca.fit_transform(X_data.flatten().reshape(len(X_data), IMG_SIZE * IMG_SIZE * CHANNELS))
        
        # Create DF of PCA's and targets 
        PCA_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
        target_df = pd.DataFrame(data=y_data, columns=['target'])
        PCA_df = pd.concat([PCA_df, target_df['target']], axis=1)
        
        return PCA_df

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

        


