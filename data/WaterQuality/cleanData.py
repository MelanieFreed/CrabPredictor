#! /usr/bin/env python2.7

#####################################################
# Clean water quality data
#
# MFreed 2015-Sept-10 Created
#####################################################

#
# Import Some Packages
#
import pandas as pd
import numpy as np
import glob
from scipy import integrate


######################################################
# Main 
#
def main():

    #
    # Get average of all *Water Quality* parameters
    # over all stations for each year
    # 
    fglob=sorted(glob.glob("WaterQualityValues_*.csv"))

    data = pd.DataFrame()
    for fn in fglob:

        print(fn)
        
        # Get year
        yr=int(fn[fn.index('_')+1:fn.index('.csv')])
        
        # Read in "raw" water quality data
        # MeasureValue,EventId,Station,Source,Project,
        # SampleDate,SampleTime,Depth,TotalDepth,Layer,
        # SampleType,SampleReplicateType,Parameter,Qualifier,
        # MeasureValue,Unit,Method,Lab,Problem,Details,Latitude,
        # Longitude,UpperPycnocline,LowerPycnocline
        with open(fn) as f:
           Nlines = sum(1 for line in f)
        dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M:%S')
        if yr == 1994: # Because has a weird double quote on that line
            tdata = pd.read_csv(fn,
                                header = 0,
                                usecols = ['MeasureValue', 'SampleDate', 'SampleTime', 'Parameter', 'Unit', 'Latitude', 'Longitude'],
                                dtype = object, 
                                skiprows=[194424], nrows = Nlines - 4, 
                                parse_dates = {'datetime': ['SampleDate', 'SampleTime']}, date_parser = dateparse)
        else:
            tdata = pd.read_csv(fn,
                                header = 0,
                                usecols = ['MeasureValue', 'SampleDate', 'SampleTime', 'Parameter', 'Unit', 'Latitude', 'Longitude'],
                                dtype = object, 
                                nrows = Nlines - 3, 
                                parse_dates = {'datetime': ['SampleDate', 'SampleTime']}, date_parser = dateparse)

            
        # Do some stuff
        tdata.MeasureValue = tdata.MeasureValue.astype(float)
        tdata.Longitude = tdata.Longitude.astype(float)
        tdata.Latitude = tdata.Latitude.astype(float)
        tdata = tdata.dropna()

        # Calculate average for each parameter
        hold = tdata.groupby(['Parameter', 'Unit', 'Latitude', 'Longitude']).mean().reset_index()

        # Calculate amount of time that each location spends with oxygen less than 5 mg/l
        hold2 = tdata[tdata.Parameter == 'DO']
        hold2 = hold2.groupby(['Parameter', 'Unit', 'Latitude', 'Longitude']).apply(lambda x: float(len(x[x.MeasureValue < 5]))/len(x)).reset_index()
        hold2.rename(columns={0: 'MeasureValue'}, inplace = True)
        hold2['Parameter'] = 'fracLowDO'

        # Add new feature to dataframe
        hold = pd.concat([hold, hold2])
        
        # Add year information
        hold['Year'] = yr

        # Add to all data
        data = data.append(hold)
        
        # Write CSV File
        #print("Writing: "+outfn+" "+str(datetime.datetime.now()))
        data.to_csv('WaterQualityAverage.csv', index=False)


# End Main
#####################################################


if __name__ == '__main__':
    main()


