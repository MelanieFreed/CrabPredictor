#! /usr/bin/env python2.7

#####################################################
# Predict the number of crabs in the Chesapeake Bay
#####################################################

#
# Import Some Packages
#
import pandas as pd
import numpy as np
import MySQLdb
import sys
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import shapefile as shp
import matplotlib.pyplot as plt
import pickle
import os

import CrabPredictionTools as cpt


######################################################
# Main 
#
def main():

    yr = 2015
    train_years = range(1995, 2015)
    clf, mean_norm, stdev_norm, sind, tind = cpt.fit_model(train_years, False)

    # Predict crab population for left out year
    sdata, hdata, wdata, seddata, savdata, ldata = cpt.get_data()
    features, featureNames = cpt.get_features(-1, sdata, hdata, wdata, seddata, savdata, ldata, [yr])
    features = features[:, sind]
    featureNames = featureNames[sind]
    features = (features - mean_norm) / stdev_norm
    features = features[:, tind]
    featureNames = featureNames[tind]

    # Predict values for next year from that data
    predicted_t_per_area_corr = clf.predict(features)
    predicted_t_per_area_corr = cpt.apply_bias_correction(predicted_t_per_area_corr)

    # Save individual results
    predicted = predicted_t_per_area_corr

    # Calculate total number of crabs in the bay
    area_bay = 9812.0 # per square kilometer
    area_bay = area_bay*((1e3**2)) # per square meter
    total_crabs = np.mean(predicted_t_per_area_corr)*area_bay

    fn = 'pickle-predictCrabs2015.pickle'
    with open(fn, 'wb') as f:
        pickle.dump([predicted, total_crabs, clf, features, featureNames, sind, tind, mean_norm, stdev_norm], f)



# End Main
#####################################################

if __name__ == '__main__':
    main()


