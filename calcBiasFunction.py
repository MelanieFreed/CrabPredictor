#! /usr/bin/env python2.7

#####################################################
# Calculate the bias function of the predictor
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
import matplotlib.cm as cm
import scipy as sp

import pickle
import os
import seaborn as sns

import CrabPredictionTools as cpt


######################################################
# Main 
#
def main():

    
    for yr in range(1995, 2015):

        # Load predictions
        fn = 'pickle-predictCrabs_'+str(yr)+'.pickle'
        with open(fn) as f:
            label_data, measured, predicted, clf, features, featureNames = pickle.load(f)

        if yr == 1995:
            allmeas = measured
            allpred = predicted
        else:
            allmeas = np.append(allmeas, measured)
            allpred = np.append(allpred, predicted)

    fit = sp.stats.linregress(allpred, allmeas)
    print(fit)
    
    plt.figure()
    sns.set(font_scale = 2)    
    sns.regplot(allmeas, allpred)
    plt.tight_layout()
    plt.savefig('calcBiasFunction.png')
    plt.close()


# End Main
#####################################################

if __name__ == '__main__':
    main()


