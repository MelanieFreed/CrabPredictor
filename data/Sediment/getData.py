#! /usr/bin/env python2.7

#####################################################
# Get blue crab harvest data (for Maryland only)
#
# Read historical data from a URL.  Manually add
#     data from recent years from individual reports.
#
# MFreed 2015-Sept-10 Created
#####################################################

#
# Import Some Packages
#
import numpy as np
import pandas as pd
import urllib
import requests
import zipfile


######################################################
# Main 
#
def main():

    #
    # Get sediment data
    #


    # Download file - Metadata
    url = "http://www.mgs.md.gov/coastal_geology/documents/CBMetatxt.zip"
    testfile = urllib.URLopener()
    testfile.retrieve(url, "Metadata.zip")

    # Download file - Data
    url = "http://www.mgs.md.gov/coastal_geology/documents/CBESSdat.zip"
    testfile = urllib.URLopener()
    testfile.retrieve(url, "Data.zip")

    # Unzip files
    with zipfile.ZipFile("Metadata.zip", "r") as z:
        z.extractall("./")
    with zipfile.ZipFile("Data.zip", "r") as z:
        z.extractall("./")


    # Read in data
    # Nearly all locations were only sampled once
    # Only 6/4249 locations were sampled twice
    fn = 'Chesdata.asc'
    data = pd.read_csv(fn,
                       header = 0,
                       usecols = ['Decimal Latitude', 'Decimal Longitude', 'Bulk Density',
                                  'Total Carbon', 'Organic Carbon', 'Sulfur', 'Gravel', 'Phi  -1',
                                  'Phi -075', 'Phi -050', 'Phi -025', 'Phi 0', 'Phi 025', 'Phi 050',
                                  'Phi 075', 'Phi 100', 'Phi 125', 'Phi 150', 'Phi 175', 'Phi 200',
                                  'Phi 225', 'Phi 250', 'Phi 275', 'Phi 300', 'Phi 325', 'Phi 350',
                                  'Phi 375', 'Phi 400', 'Phi 433', 'Phi 467', 'Phi 500', 'Phi 533',
                                  'Phi 567', 'Phi 600', 'Phi 633', 'Phi 667', 'Phi 700', 'Phi 733',
                                  'Phi 767', 'Phi 800', 'Phi 833', 'Phi 867', 'Phi 900', 'Phi 933',
                                  'Phi 967', 'Phi 10', 'Phi 11', 'Phi 12', 'Phi 13', 'Phi 14', 'Sand',
                                  'Silt', 'Clay', "Shepard's Class", 'Median', 'Graphic Mean',
                                  'Graphic Sorting', 'Graphic Skewness', 'Graphic Kurtosis',
                                  'Moment Mean', 'Moment Sorting', 'Moment Skewness',
                                  'Moment Kurtosis'],
                       dtype = np.float64)
    cnames_orig = list(data.columns.values)
    cnames_new = [val.replace(' ','').replace("'","").replace('-','minus') for val in cnames_orig]
    cnames = {cnames_orig[ii]: cnames_new[ii]  for ii in range(len(cnames_orig))}
    data.rename(columns = cnames, inplace = True)

    # Get rid of values out of range
    ind = np.where((data['BulkDensity'] < 0) | (data['BulkDensity'] > 3))[0]
    data.loc[ind, 'BulkDensity'] = float('NaN')

    cnames = data.iloc[:, 3:53].columns.values
    for ii in range(len(cnames)):
        ind = np.where((data[cnames[ii]] < 0) | (data[cnames[ii]] > 100))[0]
        data.loc[ind, cnames[ii]] = float('NaN')

    # Make longitude have correct sign
    data['DecimalLongitude'] = (-1)*data['DecimalLongitude']

    # Shepard's class is a categorical variable, so replace it with
    # dummy variables and leave one out (because it's redundant)
    Nsamples = data.shape[0]
    for ii in range(np.int32(np.max(data['ShepardsClass']))):
        hold = np.zeros((Nsamples, ), dtype = np.int32)
        hold[np.where(data['ShepardsClass'] == ii)[0]] = 1
        data['ShepardsClass'+str(ii)] = hold
    data.drop(['ShepardsClass'], inplace = True, axis = 1)
    
    # Save as CSV file
    # Harvest is in pounds
    #
    data.to_csv('Sediment.csv', index=False)


# End Main
#####################################################


if __name__ == '__main__':
    main()


