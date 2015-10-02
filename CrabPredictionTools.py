#! /usr/bin/env python2.7

#####################################################
# Tools to predict the number of crabs in the
#     Chesapeake Bay
#####################################################

#
# Import Some Packages
#
import pandas as pd
import numpy as np
from ggplot import *
from scipy import stats
import MySQLdb
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy as sp
from sklearn import linear_model
from sklearn import svm
from sklearn import cross_validation
from sklearn import feature_selection 
from sklearn import metrics
from sklearn import grid_search
from sklearn import tree
from sklearn import ensemble
from sklearn.decomposition import PCA 
from geopy.distance import vincenty
import pickle
import os


#####################################################
# fit_model
#
# Fit the model to the data and get coefficients
#
def fit_model(prediction_years, flag_save):
    """clf, mean_norm, stdev_norm, sind, tind = fit_model(prediction_years)

       Fit model to crab data.

       Inputs:
           prediction_years = Array of years to use for labels for data to fit model

       Outputs:    
           clf = Linear model fit to data
           mean_norm = Mean normalization values
           stdev_norm = Feature normalization values
    """  

    # Get data
    print('Getting data...')
    fn = 'pickle-get_data.pickle'
    if os.path.isfile(fn):
        with open(fn) as f:
            sdata, hdata, wdata, seddata, savdata, ldata = pickle.load(f)
    else:
        sdata, hdata, wdata, seddata, savdata, ldata = get_data()
        if flag_save:
            with open(fn, 'w') as f:
                pickle.dump([sdata, hdata, wdata, seddata, savdata, ldata], f)
    
    # Get labels
    print('Getting labels...')
    fn = 'pickle-get_labels.pickle'
    if os.path.isfile(fn):
        with open(fn) as f:
            labels, label_data, prediction_years = pickle.load(f)
    else:
        labels, label_data = get_labels(sdata, prediction_years)
        if flag_save:
            with open(fn, 'w') as f:
                pickle.dump([labels, label_data, prediction_years], f)
        
    # Get features
    print('Getting features...')
    fn = 'pickle-get_features.pickle'
    if os.path.isfile(fn):
        with open(fn) as f:
            features, featureNames = pickle.load(f)
    else:
        features, featureNames = get_features(label_data, sdata, hdata, wdata, seddata, savdata, ldata, prediction_years)
        if flag_save:
            with open(fn, 'w') as f:
                pickle.dump([features, featureNames], f)
        
    # Select features
    print('Selecting features...')
    fn = 'pickle-select_features.pickle'
    if os.path.isfile(fn):
        with open(fn) as f:
            sind, nlflag = pickle.load(f)
    else:
       sind, nlflag = select_features(labels, features, 0.000001, 1.0)
       if flag_save:
           with open(fn, 'w') as f:
               pickle.dump([sind, nlflag], f)
    features = features[:, sind]
    featureNames = featureNames[sind]
 
    # Normalize features
    print('Normalizing features...')
    fn = 'pickle-norm_features.pickle'
    if os.path.isfile(fn):
        with open(fn) as f:
            features, mean_norm, stdev_norm = pickle.load(f)
    else:
        features, mean_norm, stdev_norm = norm_features(features)
        if flag_save:
            with open(fn, 'w') as f:
                pickle.dump([features, mean_norm, stdev_norm], f)
        
    # Train the model
    print('Training model...')
    fn = 'pickle-train_model.pickle'
    if os.path.isfile(fn):
        with open(fn) as f:
            clf, tind = pickle.load(f)
    else:
        clf, tind = train_model(labels, features)
        if flag_save:
            with open(fn, 'w') as f:
                pickle.dump([clf, tind], f)
        
    # Return results
    return clf, mean_norm, stdev_norm, sind, tind, labels, features
    
#
# End fit_model
#####################################################


#####################################################
# calc_bias_function
#
# Calculate bias function
#
def calc_bias_function(measured, predicted):
    """fit = calc_bias_function(measured, predicted)

       Calculate bias function for individual sites. 

       Inputs:
           measured = measured crab counts
           predicted = predicted crab counts

       Outputs:    
           fit = fit results
    """  

    # Do a linear fit
    fit = sp.stats.linregress(predicted, measured)
        
    # Return results
    return fit
    
#
# End calc_bias_function
#####################################################


#####################################################
# get_data
#
# Get crab data to perform fit to.
#
def get_data():
    """stock_data, harvest_data, water_data, sediment_data, landings_data = get_data()

       Get data to fit. 

       Inputs:

       Outputs:    
           stock_data = Data about stock abundance of crabs. 
           harvest_data = Data about crab harvest from the bay. 
           water_data = Data about water quality in the bay. 
           sediment_data = Data about sediment in bay.
           landings_data = Data about commercial landings in Maryland
    """  

    # Connect to database
    db = MySQLdb.connect(host = "localhost", user = "crabby", passwd = "crabby", db = "crabs")
    cur = db.cursor()
    
    # Get crab stock counts
    # ['date', 'sal', 'wtemp', 'depth', 'year', 'lat', 'lon', 'm0_per_area_corr',
    #  'm1_per_area_corr', 'm2_per_area_corr', 'f0_per_area_corr',
    #  'f1_per_area_corr', 'f2_per_area_corr', 't_per_area_corr']
    sdata = pd.read_sql('SELECT * FROM DREDGEDETAIL;', con = db)

    # Get crab harvest counts
    # ['Year', 'Total_Crabs']
    hdata = pd.read_sql('SELECT * FROM HARVEST;', con = db)
    hdata['Total_Crabs'] = hdata['Total_Crabs']/1e6 # Convert to millions

    # Get water quality data
    # ['Parameter', 'Unit', 'Latitude', 'Longitude', 'MeasureValue', 'Year']
    wdata = pd.read_sql('SELECT * FROM WATERQUALITY;', con = db)

    # Get sediment data
    # ['DecimalLatitude', 'DecimalLongitude', 'BulkDensity', 'TotalCarbon',
    #  'OrganicCarbon', 'Sulfur', 'Gravel', 'Phiminus1', 'Phiminus075',
    #  'Phiminus050', 'Phiminus025', 'Phi0', 'Phi025', 'Phi050', 'Phi075',
    #  'Phi100', 'Phi125', 'Phi150', 'Phi175', 'Phi200', 'Phi225', 'Phi250',
    #  'Phi275', 'Phi300', 'Phi325', 'Phi350', 'Phi375', 'Phi400', 'Phi433',
    #  'Phi467', 'Phi500', 'Phi533', 'Phi567', 'Phi600', 'Phi633', 'Phi667',
    #  'Phi700', 'Phi733', 'Phi767', 'Phi800', 'Phi833', 'Phi867', 'Phi900',
    #  'Phi933', 'Phi967', 'Phi10', 'Phi11', 'Phi12', 'Phi13', 'Phi14', 'Sand',
    #  'Silt', 'Clay', 'ShepardsClass', 'Median', 'GraphicMean', 'GraphicSorting',
    #  'GraphicSkewness', 'GraphicKurtosis', 'MomentMean', 'MomentSorting',
    #  'MomentSkewness', 'MomentKurtosis']
    seddata = pd.read_sql('SELECT * FROM SEDIMENT;', con = db)
    # Drop these because too many NaN values
    seddata.drop(['TotalCarbon', 'OrganicCarbon', 'Sulfur'], inplace = True, axis = 1)

    # Get SAV data
    # ['Latitude', 'Longitude', 'SAV_ha', 'Year']
    savdata = pd.read_sql('SELECT * FROM SAV;', con = db)

    # Get Landings data
    # ['Species', 'Landings_lbs', 'Year']
    ldata = pd.read_sql('SELECT * FROM LANDINGS;', con = db)
    
    # Close database
    cur.close()
    db.close()

    # Return data
    return sdata, hdata, wdata, seddata, savdata, ldata
 
#
# End get_data
#####################################################


#####################################################
# get_labels
#
# Get labels from data.
#
def get_labels(sdata, prediction_years):
    """labels, label_data = get_labels(stock_data, prediction_years)

       Get labels for data. 

       Inputs:
           stock_data = Stock abundance data from get_data.
           prediction_years = Get labels for these years

       Outputs:    
           labels = labels for data
           label_data = additional information about labels
    """  

    # Calculate labels for prediction_years
    # = total crabs/area for each dredge site

    # Get data for all years in order of list
    for ii in range(len(prediction_years)):
        if ii == 0:
            label_data = sdata[sdata.year == prediction_years[ii]]
            label_data = label_data.loc[:,['year','lat','lon','t_per_area_corr']]
            labels = label_data['t_per_area_corr'].values
        else:
            tlabel_data = sdata[sdata.year == prediction_years[ii]]
            tlabel_data = tlabel_data.loc[:,['year','lat','lon','t_per_area_corr']]
            tlabels = tlabel_data['t_per_area_corr'].values
            label_data = label_data.append(tlabel_data)
            labels = np.append(labels, tlabels)
    
    # Return labels
    return labels, label_data
    
#
# End get_labels
#####################################################


#####################################################
# get_features
#
# Get features from data.
#
def get_features(label_data, sdata, hdata, wdata, seddata, savdata, ldata, prediction_years):
    """features, featureNames = get_features(label_data, stock_data, harvest_data, water_data, sediment_data, sav_data, landings_data, prediction_years)

       Get features from data. 

       Inputs:
           label_data = Information about labels from get_labels()
                        Set this to -1 if no labels are available (e.g., if you're
                        predicting for a future year.
           stock_data = Stock abundance data from get_data()
           harvest_data = Harvest data from get_data()
           water_data = Water quality data from get_data()
           sediment_data = Sediment data from get_data()
           sav_data = Submerged aquatic vegetation data from get_data()
           landings_data = Landings data from get_data()
           prediction_years = Get feature data to predict these years

       Outputs:    
           features = features from data
           featureNames = Name describing the feature
    """  

    #
    # Get everything in order of prediction_years list
    #
    for ii in range(len(prediction_years)):

        # Get features for prediction_year[ii]
        tfeatures, tfeatureNames = get_features_PredictionYear(label_data, sdata, hdata, wdata, seddata, savdata, ldata, prediction_years[ii])
        
        # Add this prediction year data to previous prediction years
        if ii == 0:
            features = tfeatures
            featureNames = tfeatureNames
        else:
            features = np.append(features, tfeatures, axis = 0)

    #
    # Return features
    #
    return features, featureNames
    
#
# End get_features
#####################################################

 

#####################################################
# get_features_PredictionYear
#
# Get features from data for a single year that
#    you would like to predict
#
def get_features_PredictionYear(label_data, sdata, hdata, wdata, seddata, savdata, ldata, prediction_year):
    """features, featureNames = get_features_PredictionYear(label_data, stock_data, harvest_data, water_data, sediment_data, sav_data, ldata, prediction_year)

       Get features from data for a single year that you
           would like to predict

       Inputs:
           label_data = Information about labels from get_labels()
                        Set this to -1 if no labels are available (e.g., if you're
                        predicting for a future year.
           stock_data = Stock abundance data from get_data()
           harvest_data = Harvest data from get_data()
           water_data = Water quality data from get_data()
           sediment_data = Sediment data from get_data()
           sav_data = Submerged aquatic vegetation data from get_data()
           landings_data = Landings data from get_data()
           prediction_year = A single year that you would like to predict

       Outputs:    
           features = features from data
           featureNames = Name describing the feature
    """  

    #
    # Features from Dredge data set for prediction_year
    #
    features, featureNames = get_features_PredictionYear_dredge(label_data, sdata, prediction_year)
    

    # 
    # Add number of crabs harvested for prediction_year
    #
    Nrows = features.shape[0]
    hold, holdNames = get_features_PredictionYear_harvest(hdata, prediction_year, Nrows)
    features = np.column_stack((features, hold))
    featureNames = np.append(featureNames, holdNames)

    
    #
    # Water quality features for prediction_year
    #
    hold, holdNames = get_features_PredictionYear_waterquality(label_data, sdata, wdata, prediction_year)
    features = np.column_stack((features, hold))
    featureNames = np.append(featureNames, holdNames)


    #
    # Sediment data for prediction_year
    #
    hold, holdNames = get_features_PredictionYear_sediment(label_data, sdata, seddata, prediction_year)
    features = np.column_stack((features, hold))
    featureNames = np.append(featureNames, holdNames)


    #
    # SAV features for prediction_year
    #
    hold, holdNames = get_features_PredictionYear_sav(label_data, sdata, savdata, prediction_year)
    features = np.column_stack((features, hold))
    featureNames = np.append(featureNames, holdNames)


    #
    # Distance from bay mouth for prediction_year
    #
    hold, holdNames = get_features_PredictionYear_dbe(label_data, sdata, prediction_year)
    features = np.column_stack((features, hold))
    featureNames = np.append(featureNames, holdNames)


    #
    # Landings features for prediction_year
    #
    Nrows = features.shape[0]
    hold, holdNames = get_features_PredictionYear_landings(ldata, prediction_year, Nrows)
    features = np.column_stack((features, hold))
    featureNames = np.append(featureNames, holdNames)


    #
    # Return
    #
    return features, featureNames
    
#
# End get_features_PredictionYear
#####################################################


#####################################################
# get_features_PredictionYear_dredge
#
# 
#
def get_features_PredictionYear_dredge(label_data, sdata, prediction_year):
    """features, featureNames = get_features_PredictionYear_dredge(label_data, sdata, prediction_year)

       Get features from *Dredge* data set for a single prediction year.

       Inputs:
           label_data = Information about labels from get_labels()
                        Set this to -1 if no labels are available (e.g., if you're
                        predicting for a future year.
           stock_data = Stock abundance data from get_data()
           prediction_year = A single year that you would like to predict

       Outputs:  
           features = features from data
           featureNames = Name describing the feature
    """  

    #
    # (Prediction Year - 1) through (Prediction Year - 5)
    #
    if prediction_year <= 2015:
        rlist = range(1, 6)
    else: # For 2016, repeat last year twice because you don't know values for 2015
        rlist = [2, 2, 3, 4, 5]
    ylist = range(1, 6)
    for ii in range(len(rlist)):
        # Data locations (Prediction Year - rlist[ii])
        feature_data = sdata[sdata.year == (prediction_year - rlist[ii])]
        feature_lon = feature_data.lon.values
        feature_lat = feature_data.lat.values
        feature_points = np.column_stack((feature_lon, feature_lat))

        # Fill in NaN values for all features (if they're not strings)
        for jj in range(feature_data.shape[1]):
            fdata = feature_data.iloc[:,jj].values
            if type(fdata[0]) != str:
                nind = np.where(np.isnan(fdata))[0]
                if (len(nind) > 0):
                    # If everything is NaN, then fill with the median
                    # over all years and locations
                    if (len(nind) == len(fdata)): 
                        feature_data.iloc[nind, jj] = np.median(sdata.iloc[:,jj].values)
                    else:
                        nlon = feature_lon[nind]
                        nlat = feature_lat[nind]
                        oind = np.where(np.isfinite(fdata))[0]
                        olon = feature_lon[oind]
                        olat = feature_lat[oind]
                        odata = feature_data.iloc[oind, jj]
                        npoints = interp_neighbor_combo(odata, olon, olat, nlon, nlat) 
                        feature_data.iloc[nind, jj] = npoints

        # Get features values you're interested in
        feature_data = feature_data.loc[:,['sal', 'wtemp', 'depth', 'm0_per_area_corr',
                                           'm1_per_area_corr', 'm2_per_area_corr',
                                           'f0_per_area_corr', 'f1_per_area_corr',
                                           'f2_per_area_corr', 't_per_area_corr']]

        holdNames = feature_data.columns.values + '_PredictionYr-'+str(ylist[ii])
    
        # Interpolate feature data to prediction_year locations if necessary
        if type(label_data) == int:
            # Interpolate to prediction_year - 1 because apparently no labels are available for this year
            pdata = sdata[sdata.year == (prediction_year - rlist[0])]
            predict_lon = pdata.lon.values
            predict_lat = pdata.lat.values
            hold = interp_neighbor_combo(feature_data, feature_lon, feature_lat, predict_lon, predict_lat)
        else:    
            # Interpolate feature data
            predict_lon = label_data[label_data.year == prediction_year].lon.values # prediction locations for prediction_year
            predict_lat = label_data[label_data.year == prediction_year].lat.values
            hold = interp_neighbor_combo(feature_data, feature_lon, feature_lat, predict_lon, predict_lat)

        if ii == 0:
            features = hold
            featureNames = holdNames
        else:
            features = np.column_stack((features, hold))
            featureNames = np.append(featureNames, holdNames)

    #
    # Return
    #
    return features, featureNames
        
#
# End get_features_PredictionYear_dredge
#####################################################




#####################################################
# get_features_PredictionYear_harvest
#
# 
#
def get_features_PredictionYear_harvest(hdata, prediction_year, Nrows):
    """features, featureNames = get_features_PredictionYear_harvest(hdata, prediction_year, Nrows)

       Get features from *Harvest* data set for a single prediction year.

       Inputs:
           harvest_data = Harvest data from get_data()
           prediction_year = A single year that you would like to predict
           Nrows = number of rows in dredge data set (so you know how big to make
                   features that you return from this program)

       Outputs:  
           features = features from data
           featureNames = Name describing the feature
    """  

    #
    # (Prediction Year - 1) through (Prediction Year - 5)
    #
    if prediction_year <= 2015:
        rlist = range(1, 6)
    else: # For 2016, repeat last year twice because you don't know values for 2015
        rlist = [2, 2, 3, 4, 5]
    ylist = range(1, 6)
    for ii in range(len(rlist)):
        hold = np.zeros((Nrows,1),dtype = np.float64)
        hold[:] = hdata[hdata.Year == (prediction_year - rlist[ii])].Total_Crabs.values
        holdNames = 'Harvest_PredictionYr-'+str(ylist[ii])
        if ii == 0:
            features = hold
            featureNames = holdNames
        else:
            features = np.column_stack((features, hold))
            featureNames = np.append(featureNames, holdNames)

    
    #
    # Return
    #
    return features, featureNames
        
#
# End get_features_PredictionYear_harvest
#####################################################


#####################################################
# get_features_PredictionYear_waterquality
#
# 
#
def get_features_PredictionYear_waterquality(label_data, sdata, wdata, prediction_year):
    """features, featureNames = get_features_PredictionYear_waterquality(label_data, sdata, wdata, prediction_year)

       Get features from *Water Quality* data set for a single prediction year.

       Inputs:
           label_data = Information about labels from get_labels()
                        Set this to -1 if no labels are available (e.g., if you're
                        predicting for a future year.
           stock_data = Stock abundance data from get_data()
           water_data = Water quality data from get_data()
           prediction_year = A single year that you would like to predict

       Outputs:  
           features = features from data
           featureNames = Name describing the feature
    """  

    #
    # (Prediction Year - 1) through (Prediction Year - 5)
    #

    # Get parameter names that are present for all years
    # LATER YOU MAY WANT TO CHANGE THIS
    hold = wdata.groupby(['Parameter', 'Unit']).apply(lambda x: len(np.unique(x.Year))).reset_index()
    ind = np.where(hold[0] == 25)[0]
    wparams = hold['Parameter'].iloc[ind].values
    wparamName = hold['Parameter'].iloc[ind].values + '_' + hold['Unit'].iloc[ind].values

    if prediction_year <= 2015:
        rlist = range(1, 6)
    else: # For 2016, repeat last year twice because you don't know values for 2015
        rlist = [2, 2, 3, 4, 5]
    ylist = range(1, 6)
    for ii in range(len(rlist)):
        
        # Interpolate each parameter separately to prediction_year
        # because each parameter could be sampled at different lat/lon

        # Interpolate data
        whold = wdata[wdata.Year == (prediction_year - rlist[ii])]
        for jj in range(len(wparams)):
            # Water quality data and locations for this parameter
            wlon = (whold[whold['Parameter'] == wparams[jj]].Longitude.values)
            wlat = (whold[whold['Parameter'] == wparams[jj]].Latitude.values) 
            wvalues = whold[whold['Parameter'] == wparams[jj]].MeasureValue.values
            # Locations we need to predict to
            if type(label_data) == int:
                # If you don't have labels, just regrid water quality data to features
                feature_data = sdata[sdata.year == (prediction_year - rlist[0])]
                predict_lon = feature_data.lon.values
                predict_lat = feature_data.lat.values
            else:
                predict_lon = label_data[label_data.year == prediction_year].lon.values 
                predict_lat = label_data[label_data.year == prediction_year].lat.values
            # Interpolate water quality data to new locations
            hold2 = interp_neighbor_combo(wvalues, wlon, wlat, predict_lon, predict_lat)
            if jj == 0:
                hold = hold2
            else:
                hold = np.column_stack((hold, hold2))
    
        holdNames = wparams+'_PredictionYear-'+str(ylist[ii])

        if ii == 0:
            features = hold
            featureNames = holdNames
        else:
            features = np.column_stack((features, hold))
            featureNames = np.append(featureNames, holdNames)


    #
    # Return
    #
    return features, featureNames
        
#
# End get_features_PredictionYear_waterquality
#####################################################



#####################################################
# get_features_PredictionYear_sediment
#
# 
#
def get_features_PredictionYear_sediment(label_data, sdata, seddata, prediction_year):
    """features, featureNames = get_features_PredictionYear_sediment(label_data, sdata, seddata, prediction_year)

       Get features from *Sediment* data set for a single prediction year.

       Inputs:
           label_data = Information about labels from get_labels()
                        Set this to -1 if no labels are available (e.g., if you're
                        predicting for a future year.
           stock_data = Stock abundance data from get_data()
           sediment_data = Sediment data from get_data()
           prediction_year = A single year that you would like to predict

       Outputs:  
           features = features from data
           featureNames = Name describing the feature
    """  


    # Interpolate each parameter separately to prediction_year lat/lon
    for ii in range(seddata.shape[1]):
        slon = seddata.DecimalLongitude.values
        slat = seddata.DecimalLatitude.values
        svalues = seddata.iloc[:, ii]
        # Drop NaN values
        ind = np.where(np.isfinite(svalues))[0]
        slon = slon[ind]
        slat = slat[ind]
        svalues = svalues[ind]
        if type(label_data) == int:
            # If you don't have labels, just regrid water quality data to features
            if prediction_year == 2015:
                feature_data = sdata[sdata.year == (prediction_year - 1)]
            elif prediction_year == 2016:
                feature_data = sdata[sdata.year == (prediction_year - 2)]
            predict_lon = feature_data.lon.values
            predict_lat = feature_data.lat.values
        else:
            predict_lon = label_data[label_data.year == prediction_year].lon.values 
            predict_lat = label_data[label_data.year == prediction_year].lat.values
        hold = interp_neighbor_combo(svalues, slon, slat, predict_lon, predict_lat)
        if ii == 0:
            features = hold
        else:
            features = np.column_stack((features, hold))
            
    featureNames = seddata.columns.values


    #
    # Return
    #
    return features, featureNames
        
#
# End get_features_PredictionYear_sediment
#####################################################


#####################################################
# get_features_PredictionYear_sav
#
# 
#
def get_features_PredictionYear_sav(label_data, sdata, savdata, prediction_year):
    """features, featureNames = get_features_PredictionYear_sav(label_data, sdata, savdata, prediction_year)

       Get features from *SAV* data set for a single prediction year.

       Inputs:
           label_data = Information about labels from get_labels()
                        Set this to -1 if no labels are available (e.g., if you're
                        predicting for a future year.
           stock_data = Stock abundance data from get_data()
           sav_data = SAV data from get_data()
           prediction_year = A single year that you would like to predict

       Outputs:  
           features = features from data
           featureNames = Name describing the feature
    """  

    #
    # (Prediction Year - 1) through (Prediction Year - 5)
    #
    if prediction_year <= 2015:
        rlist = range(1, 6)
    else: # For 2016, repeat last year twice because you don't know values for 2015
        rlist = [2, 2, 3, 4, 5]
    ylist = range(1, 6)
    for ii in range(len(rlist)):

        # Interpolate data
        hold = savdata[savdata.Year == (prediction_year - rlist[ii])]

        # SAV data and locations for this parameter
        slon = hold['Longitude'].values
        slat = hold['Latitude'].values 
        svalues = hold['SAV_ha'].values
        # Locations we need to predict to
        if type(label_data) == int:
            # If you don't have labels, just regrid SAV data to features
            feature_data = sdata[sdata.year == (prediction_year - rlist[0])]
            predict_lon = feature_data.lon.values
            predict_lat = feature_data.lat.values
        else:
            predict_lon = label_data[label_data.year == prediction_year].lon.values 
            predict_lat = label_data[label_data.year == prediction_year].lat.values
        # Interpolate water quality data to new locations
        hold = interp_neighbor_combo(svalues, slon, slat, predict_lon, predict_lat)
    
        holdNames = 'SAV_ha_PredictionYear-'+str(ylist[ii])

        if ii == 0:
            features = hold
            featureNames = holdNames
        else:
            features = np.column_stack((features, hold))
            featureNames = np.append(featureNames, holdNames)


    #
    # Return
    #
    return features, featureNames
        
#
# End get_features_PredictionYear_sav
#####################################################



#####################################################
# get_features_PredictionYear_dbe
#
# 
#
def get_features_PredictionYear_dbe(label_data, sdata, prediction_year):
    """features, featureNames = get_features_PredictionYear_dbe(label_data, sdata, prediction_year)

       Get distance from bay mouth feature for a single prediction year.

       Inputs:
           label_data = Information about labels from get_labels()
                        Set this to -1 if no labels are available (e.g., if you're
                        predicting for a future year.
           stock_data = Stock abundance data from get_data()
           prediction_year = A single year that you would like to predict

       Outputs:  
           features = features from data
           featureNames = Name describing the feature
    """  

    #
    # Get at Prediction Year locations 
    #
    if type(label_data) == int:
        # If you don't have labels, just regrid water quality data to features
        if prediction_year == 2015:
            feature_data = sdata[sdata.year == (prediction_year - 1)]
        elif prediction_year == 2016:
            feature_data = sdata[sdata.year == (prediction_year - 2)]
        predict_lon = feature_data.lon.values
        predict_lat = feature_data.lat.values
    else:
        predict_lon = label_data[label_data.year == prediction_year].lon.values 
        predict_lat = label_data[label_data.year == prediction_year].lat.values

    # Calculate distance to bay entrance at each of those locations
    # Chesapeake Bay entrance:
    # http://www.nauticalcharts.noaa.gov/nsd/distances-ports/distances.pdf
    features = np.zeros((len(predict_lon), 3), dtype = np.float64)
    cbelat = 36+(56.3/60.0)
    cbelon = (-1)*(75+(58.6/60))
    cbe = (cbelat, cbelon)
    for ii in range(len(predict_lon)):
        features[ii, 0] = vincenty(cbe, (predict_lat[ii], predict_lon[ii])).meters/1000.0
        features[ii, 1] = predict_lat[ii]
        features[ii, 2] = predict_lon[ii]

    featureNames = ['DistCBMouth_km', 'Latitude', 'Longitude']

            
    #
    # Return
    #
    return features, featureNames
        
#
# End get_features_PredictionYear_dbe
#####################################################



#####################################################
# get_features_PredictionYear_landings
#
# 
#
def get_features_PredictionYear_landings(ldata, prediction_year, Nrows):
    """features, featureNames = get_features_PredictionYear_landings(ldata, prediction_year, Nrows)

       Get features from *Landings* data set for a single prediction year.

       Inputs:
           ldata = Landings data from get_data()
           prediction_year = A single year that you would like to predict
           Nrows = number of rows in dredge data set (so you know how big to make
                   features that you return from this program)


       Outputs:  
           features = features from data
           featureNames = Name describing the feature
    """  

    #
    # (Prediction Year - 1) through (Prediction Year - 5)
    #

    # Get landings species
    allspecies = np.unique(ldata['Species'].values)
    Nspecies = len(allspecies)
    if prediction_year <= 2015:
        rlist = range(1, 6)
    else: # For 2016, repeat last year twice because you don't know values for 2015
        rlist = [2, 2, 3, 4, 5]
    ylist = range(1, 6)
    for ii in range(len(rlist)):
        hold = np.zeros((Nrows, Nspecies), dtype = np.float64)
        for jj in range(Nspecies): 
            hold[:, jj] = ldata[(ldata.Year == (prediction_year - rlist[ii])) & (ldata.Species == allspecies[jj])].Landings_lbs.values[0]
            if jj == 0:
                holdNames = 'Landings_'+ allspecies[jj] + '_PredictionYr-'+str(ylist[ii])
            else:
                holdNames = np.append(holdNames, 'Landings_'+ allspecies[jj] + '_PredictionYr-'+str(ylist[ii]))
        if ii == 0:
            features = hold
            featureNames = holdNames
        else:
            features = np.column_stack((features, hold))
            featureNames = np.append(featureNames, holdNames)

    #
    # Return
    #
    return features, featureNames
        
#
# End get_features_PredictionYear_landings
#####################################################


#####################################################
# select_features
#
# Select features that you would like to use
#
def select_features(labels, features, nvar_threshold, r2_threshold):
    """ind = select_features(labels, features, nvar_threshold, r2_threshold)

       Select and order features that you would like to use.

       Inputs:
           labels = the measured values that you would like to predict
           features = the values that you would like to use to predict the labels

       Outputs:    
           ind = indices that you would like to use to select features
           nlflag = flag telling you if feature is non-linear
    """  


    #
    # Remove low (no) variance features
    #
    #diff = np.abs(np.max(features, axis = 0) - np.min(features, axis = 0))
    #ind = np.where(diff > 0)[0]


    #
    # Remove low variance features
    #
    ind = np.array(range(features.shape[1]))
    Nfeatures = len(ind)
    fstd = np.zeros((Nfeatures, ), dtype = np.float64)
    for ii in range(Nfeatures):
        fstd[ii] = np.nanstd(features[:, ii])/np.mean(features[:, ii])
    tind = np.where(fstd >= nvar_threshold)
    ind = ind[tind]

    
    #
    # Remove any features that are highly
    #     correlated with other features
    #
    ii = 0
    while ii < (len(ind) - 1):
        chold = np.zeros((len(ind) - ii, ), dtype = np.float64)
        for jj in range(0, len(ind) - ii - 1):
            chold[jj] = (stats.spearmanr(features[:, ind[ii]], features[:, ind[jj + ii + 1]])[0])**2
        cind = np.where(chold >= r2_threshold)[0]
        ind = np.delete(ind, cind + ii + 1)
        ii = ii + 1

    
    #
    # Sort by Spearman correlation
    #     and calculate non-linear flag
    #
    Nfeatures = features.shape[1]
    scor2 = np.zeros((Nfeatures, 2), dtype = np.float64)
    nlflag = np.zeros((Nfeatures, ), dtype = bool)
    for ii in range(Nfeatures):
        hold = stats.spearmanr(features[:, ii], labels)
        scor2[ii, 0] = hold[0]**2
        scor2[ii, 1] = hold[1]
        hold = stats.pearsonr(features[:, ii], labels)
        if (scor2[ii, 0] > (hold[0]**2)):
            nlflag[ii] = True

    tind = np.argsort(scor2[ind, 0])[::-1]
    ind = ind[tind]

    
    #
    # Return
    #
    return ind, nlflag
    
            
#
# End select_features
#####################################################


#####################################################
# norm_features
#
# Normalize features and do mean subtraction
#
def norm_features(*args):
    """features, mean_norm, stdev_norm = norm_features(features [, mean_norm, stdev_norm])

       Normalize and mean subtract features to get them ready 
           for regression. 

       Inputs:
           features = features from get_features()
           mean_norm = [Optional] use this to do normalization if provided
           stdev_norm = [Optional] use this to do normalization if provided

       Outputs:    
           features = normalized and mean subtracted features
           mean_norm = mean normalization values
           stdev_norm = stdev normalization values
    """  

    features = args[0]

    # Calculate mean and standard deviations if they
    # weren't provided
    if len(args) == 1:
        mn_norm = np.nanmean(features, axis = 0)
        sd_norm = np.nanstd(features, axis = 0)
    else:
        mn_norm = args[1]
        sd_norm = args[2]
        
    features = (features - mn_norm) / sd_norm

    # Return values
    return features, mn_norm, sd_norm

#
# End norm_features
#####################################################

    
#####################################################
# optimize_model
#
# Examine different models. 
#
def optimize_model(labels, features):
    """optimize_model(labels, features)

       Try several different methods to train the model. 

       Inputs:    
           labels = Labels for the data set
           features = Features for the data set

       Outputs:

    """

    # Open output file to save results
    fn = 'CPT_optimize_model.output'
    f = open(fn, 'w')

    spca=PCA(n_components=0.95) 
    spca.fit(features)
    pfeatures=spca.transform(features)

    # Linear Regression
    clf = linear_model.LinearRegression(fit_intercept = True, normalize = False, copy_X = True)
    r2_linear = get_cv_r2(labels, pfeatures, clf)
    hold = np.mean(r2_linear, axis = 0)
    thold = 'Linear: '+ str(hold[0]) + ' ' + str(hold[1])
    f.write(thold)
    print(thold)

    # SVR with linear kernel - VERY SLOW AND NOT BETTER THAN LINEAR REGRESSION
    #Nfolds = 5
    #kf = cross_validation.KFold(pfeatures.shape[0], n_folds = Nfolds, shuffle = True, random_state = 35)    
    #svr = svm.SVR(cache_size = 3000, kernel = 'linear')
    #parameters = {'C': [0.3, 1.0, 3.0], 'epsilon': [0.03, 0.1, 0.3]}
    #clf  = grid_search.GridSearchCV(svr, parameters, scoring = 'r2', cv = kf, verbose = 1, n_jobs = 3)
    #clf.fit(pfeatures, labels)
    #r2_linearSVR = get_cv_r2(labels, pfeatures, clf.best_estimator_)
    
    #hold = np.mean(r2_linearSVR, axis = 0)
    #thold = 'SVR with linear kernel: '+ str(hold[0]) + ' ' + str(hold[1])
    #f.write(thold)
    #f.write(str(clf.best_params_))
    #print(thold)
    #print(clf.best_params_)
    
    # SVR with gaussian kernel
    Nfolds = 5
    kf = cross_validation.KFold(pfeatures.shape[0], n_folds = Nfolds, shuffle = True, random_state = 35)    
    svr = svm.SVR(cache_size = 3000, kernel = 'rbf')
    parameters = {'C': [0.003, 0.01, 0.03, 0.1], 'epsilon': [0.0001, 0.003, 0.01, 0.03], 'degree': [2, 3, 4, 5]}
    clf  = grid_search.GridSearchCV(svr, parameters, scoring = 'r2', cv = kf, verbose = 1, n_jobs = 3)
    clf.fit(pfeatures, labels)
    r2_rbfSVR = get_cv_r2(labels, pfeatures, clf.best_estimator_)

    hold = np.mean(r2_rbfSVR, axis = 0)
    thold = 'SVR with Gaussian kernel: ' + str(hold[0]) + ' ' + str(hold[1])
    f.write(thold)
    f.write(str(clf.best_params_))
    f.write(str(clf.grid_scores_))
    print(thold)
    print(clf.best_params_)
    print(clf.grid_scores_)

    # Random Forest
    Nfolds = 5
    kf = cross_validation.KFold(pfeatures.shape[0], n_folds = Nfolds, shuffle = True, random_state = 35)    
    rf = ensemble.RandomForestRegressor(verbose = 2)
    parameters = {'n_estimators': [80, 90, 100], 'max_depth': [20, 30, 40, 50, 60, 70]}
    clf  = grid_search.GridSearchCV(rf, parameters, scoring = 'r2', cv = kf, verbose = 1, n_jobs = 3)
    clf.fit(pfeatures, labels)
    r2_rf = get_cv_r2(labels, pfeatures, clf.best_estimator_)

    hold = np.mean(r2_rf, axis = 0)
    thold = 'Random Forest: ' + str(hold[0]) + ' ' + str(hold[1]) 
    f.write(thold)
    f.write(str(clf.best_params_))
    f.write(str(clf.grid_scores_))
    print(thold)
    print(clf.best_params_)
    print(clf.grid_scores_)
    
    # Close output file
    f.close()
    
    # Return 
    return 
    
#
# End optimize_model
#####################################################


#####################################################
# train_model
#
# Train the model with your data
#
def train_model(labels, features):
    """clf = train_model(labels, features)

       Train the model on your data.

       Inputs:    
           labels = Labels for the data set
           features = Features for the data set

       Outputs:
           clf = Regression model

    """

    #Nfolds = 5
    #kf = cross_validation.KFold(features.shape[0], n_folds = Nfolds, shuffle = True, random_state = 35)    
    #lr = linear_model.LinearRegression(fit_intercept = True, copy_X = True)
    #clf = feature_selection.RFECV(lr, step = 1, cv = kf, scoring = 'r2')
    #clf.fit(features, labels)
    #feature_mask = rfecv.support_
    #ind = np.where(feature_mask)[0]
    #features = features[:, ind]
    #featureNames = featureNames[ind]


    # Random Forest
    clf = ensemble.RandomForestRegressor(n_estimators = 90, max_depth = 50, n_jobs = 3)
    clf.fit(features, labels)

    ind = np.array(range(features.shape[1]))
    
    # Return linear regression model
    return clf, ind
    
#
# End train_model
#####################################################


#####################################################
# interp_neighbor_combo
#
# Interpolate data to new locations using a combination
#     of nearest neighbors
#
def interp_neighbor_combo(features, xx_original, yy_original, xx_new, yy_new):
    """features_interpolated = interp_neighbor_combo(features, xx_original, yy_original, xx_new, yy_new)

       Interpolate data to new location using a combination
           of nearest neighbors.

       Inputs:
           features = features from get_features()
           xx_original = x location of original data
           yy_original = y location of original data
           xx_new = x location of new data 
           yy_new = y location of new data

       Outputs:    
           features_interpolated = new estimated data values
    """  

    # Convert to numpy array if it's a dataframe
    if type(features) == pd.DataFrame:
        features = np.array(features)

    # Process each feature separately
    Nsamples = len(xx_new)
    if (len(features.shape) != 2):
        features = features.reshape((len(features), 1))
    Nfeatures = features.shape[1]
    features_interpolated = np.zeros((Nsamples, Nfeatures), dtype = np.float64)
    Nuse = 5
    for ii in range(Nfeatures):
        data = features[:,ii]
        for jj in range(Nsamples):
            dist = np.sqrt(((xx_original - xx_new[jj])**2) + ((yy_original - yy_new[jj])**2))
            ind = np.where(dist == 0)[0]
            dist[ind] = 0.00001 # prevent dividing by zero when calculating weights
            ind = np.argsort(dist)
            ind = ind[0:Nuse] # Keep only a few values for interpolation
            dist = dist[ind] # Weight average by 1/distance
            features_interpolated[jj,ii] = np.average(features[ind,ii], weights = 1/(dist**2))
            
    # Return interpolated data
    return features_interpolated
    
#
# End interp_neighbor_combo
#####################################################


   
#####################################################
# get_cv_r2
#
# Get cross-validated r2 for a model.
#
def get_cv_r2(labels, features, model):
    """r2scores = get_cv_r2(labels, features, model)

       Calculate cross-validated R2 score for a model. 

       Inputs:    
           labels = Labels for the data set
           features = Features for the data set
           model = the model

       Outputs:
           r2scores = R2 scores for training and cross-validation data set.

    """

    # Get training and cross-validation metrics for each k-fold
    Nfolds = 5
    kf = cross_validation.KFold(features.shape[0], n_folds = Nfolds, shuffle = True, random_state = 47)    
    
    r2scores = np.zeros((Nfolds, 2), dtype = np.float64)
    ik = 0
    for itrain, icross in kf:
        ftrain = features[itrain, :]
        ltrain = labels[itrain]
        fcross = features[icross, :]
        lcross = labels[icross]
        model.fit(ftrain, ltrain)
        r2scores[ik, 0] = metrics.r2_score(ltrain, model.predict(ftrain))
        r2scores[ik, 1] = metrics.r2_score(lcross, model.predict(fcross))
        ik = ik + 1

    # Return linear regression model
    return r2scores
    
#
# End get_cv_r2
#####################################################


#####################################################
# apply_bias_correction
#
# Apply bias correction to predictions
# [This was calculated on non-bias corrected predictions
#  using calcBiasFunction.py]
#
def apply_bias_correction(predictions, fit):
    """predictions_corrected = apply_bias_correction(predictions)

       Apply bias correction to predictions. 

       Inputs:    
           predictions = predictions you would like to correct
           fit = fit from calc_bias_correction

       Outputs:
           predictions_corrected = predictions corrected for model bias

    """

    # Apply correction
    predictions_corrected = predictions * fit[0] + fit[1]
    ind = np.where(predictions_corrected < 0)[0]
    predictions_corrected[ind] = 0.0

    # Return linear regression model
    return predictions_corrected
    
#
# End apply_bias_correction
#####################################################



#if __name__ == '__main__':
#    main()


