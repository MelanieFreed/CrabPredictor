#! /usr/bin/env python2.7

#####################################################
# Get Landings data
#####################################################

#
# Import Some Packages
#
import numpy as np
import pandas as pd

######################################################
# Main 
#
def main():

    #
    # Get Landings data
    #
    data = pd.DataFrame()
    for yr in range(1990, 2015):
        fn = 'Landings_MD_'+str(yr)+'.csv'
        with open(fn) as f:
            Nlines = sum(1 for line in f)
        hold = pd.read_csv(fn,
                           header = None, skiprows = 10, nrows = Nlines - 18,
                           names = ['Species', 'remove0', 'Landings_lbs', 'remove1'],
                           usecols = ['Species', 'Landings_lbs'], 
                           dtype = {'Species': str, 'Landings_lbs': np.float64})
        shold = [sp.replace(',', '').replace('-', '').replace(' ', '_') for sp in hold.Species]
        hold['Species'] = shold
        hold['Year'] = yr
        data = data.append(hold)


    # Add zeros when there is no landings data
    allyears = np.unique(data['Year'].values)
    for sp in np.unique(data['Species'].values):
        yrs = data[data['Species'] == sp].Year.values
        missingyrs = [yy for yy in allyears if yy not in yrs]
        for my in missingyrs:
            data.loc[data.shape[0]] = [sp, 0.0, my]
        
    # Save as CSV file
    data.to_csv('Landings.csv', index=False)


# End Main
#####################################################


if __name__ == '__main__':
    main()


