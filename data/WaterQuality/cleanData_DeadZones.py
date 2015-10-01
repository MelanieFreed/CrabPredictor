#! /usr/bin/env python2.7

#####################################################
# Calculate number of stations with "dead zones"
#
# MFreed 2015-Sept-11 Created
#####################################################

#
# Import Some Packages
#
import pandas as pd
import numpy as np
import glob


######################################################
# Main 
#
def main():

    #
    # Find all your data files and process each one
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
        if yr == 1994: # Because has a weird double quote on that line
            tdata = pd.read_csv(fn,
                                header = 0,
                                usecols = ['MeasureValue', 'Parameter', 'Unit', 'Station', 'SampleDate', 'SampleTime'],
                                dtype = object, skiprows=[194424])
        else:
            tdata = pd.read_csv(fn,
                                header = 0,
                                usecols = ['MeasureValue', 'Parameter', 'Unit', 'Station', 'SampleDate', 'SampleTime'],
                                dtype = object)
        # delete last row, it's just the total record number
        tdata = tdata.iloc[0:len(tdata)-1]
        tdata.MeasureValue = tdata.MeasureValue.astype(float)
        tdata = tdata.dropna()
        
        # Keep only DO values (dissolved oxygen)
        tdata = tdata[tdata.Parameter == 'DO']

        # Keep only data in July
        tdata = tdata[tdata.SampleDate >= ('7/1/'+str(yr))]
        tdata = tdata[tdata.SampleDate <= ('7/31/'+str(yr))]
        
        # Group by station
        # Crabs need 3 mg/l of dissolved oxygen to survive
        #   calculate fraction of stations that were always
        #   above this threshold
        do = tdata.groupby(['Station']).min().reset_index().MeasureValue
        fdead = float(len(np.where(do < 5)[0]))/float(len(do))
        
        # Add year information
        tdata['Year'] = yr

        # Add to all data
        data = data.append(pd.Series({'Year': yr, 'NlowO': fdead}), ignore_index=True)
        
        #
        # Write CSV File
        #
        #print("Writing: "+outfn+" "+str(datetime.datetime.now()))
        data.to_csv('WaterQualityDeadZones.csv', index=False)



# End Main
#####################################################



if __name__ == '__main__':
    main()


