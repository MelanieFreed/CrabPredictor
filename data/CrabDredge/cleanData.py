#! /usr/bin/env python2.7

#####################################################
# Process crab dredge data (for Maryland only)
#
# This data was provided to me by Glenn Davis at
#     the Maryland Department of Natural Resources
#     as an Excel file, which I manuually saved
#     as a CSV file.
#
#####################################################

#
# Import Some Packages
#
import pandas as pd
import numpy as np
from geopy.distance import vincenty

######################################################
# Main 
#
def main():

    #
    # Calculate number of detected crabs per area,
    # uncorrected by gear efficiency
    #
    
    # Read in "raw" dredge data
    # DATE,LOC,NOAA,SITE,TOW,AREA,ALAT,ALONG,BLAT,BLONG,SAL,WTEMP,DEPTH,VESSEL,STRATUM,M0,M1,M2,F0,F1,F2,T,YEAR
    data = pd.read_csv('WD9014.md.csv',
                       names = ['date', 'loc', 'noaa', 'site', 'tow', 'area', 'alat', 'along', 'blat', 'blong', 'sal',
                                'wtemp', 'depth', 'vessel', 'stratum', 'm0', 'm1', 'm2', 'f0', 'f1', 'f2', 't', 'year'],
                       header = 0,
                       usecols = ['date', 'site', 'tow', 'area', 'alat', 'along', 'blat', 'blong',
                                  'sal', 'wtemp', 'depth', 'm0', 'm1', 'm2', 'f0', 'f1', 'f2', 't', 'year'],
                       dtype = {'locid': object, 'noaa': np.int32, 'site': np.int32, 
                                'tow': np.int32, 'area': np.float64, 'alat': object,
                                'blat': object, 'along': object, 'blong': object, 'wtemp': np.float64, 
                                'depth': np.float64, 'vessel': object, 'm0': np.int32, 'm1': np.int32, 'm2': np.int32, 
                                'f0': np.int32, 'f1': np.int32, 'f2': np.int32, 't': np.int32, 'year': np.int32},
                       converters = {'locid': convert_strip, 'alat': convert_latlon, 'along': convert_latlon,
                                     'blat': convert_latlon, 'blong': convert_latlon}) 

    # Site is a unique identifier
    # Tows can be repeated rarely if they thought there was a problem with
    #    the first two, so just keep the last tow for any site on a given day
    data = data.groupby(['date', 'site']).apply(lambda g: g.iloc[len(g)-1:len(g), :]).reset_index(drop=True)

    # Get rid of site and tow columns (you only need lat/long from now on)
    data.drop(['site', 'tow'], inplace=True, axis=1)

    # Convert date string to date format
    # First part of data is non-zero padded mm/dd/yyyy
    # Second part of data (starting 12/2013) is zero padded mm/dd/yy
    # The date parser seems to understand both these types fine
    data.date = pd.to_datetime(data.date)

    # Calculate average latitude and longitude
    data['lat'] = data[['alat', 'blat']].mean(axis=1)
    data['long'] = data[['along', 'blong']].mean(axis=1) * (-1)
    data.drop(['alat', 'blat', 'along', 'blong'], inplace=True, axis=1)

    # Calculate number of crabs per dredge area instead of absolute numbers
    data['m0_per_area'] = data['m0'].div(data['area'])
    data['m1_per_area'] = data['m1'].div(data['area'])
    data['m2_per_area'] = data['m2'].div(data['area'])
    data['f0_per_area'] = data['f0'].div(data['area'])
    data['f1_per_area'] = data['f1'].div(data['area'])
    data['f2_per_area'] = data['f2'].div(data['area'])
    data['t_per_area'] = data['t'].div(data['area'])
    data.drop(['m0', 'm1', 'm2', 'f0', 'f1', 'f2', 't', 'area'], inplace=True, axis=1)

    # Correct for gear efficiency
    # gear efficiency = 0.15
    # Source: http://brage.bibsys.no/xmlui/handle/11250/194624
    # Note: Sharov et al. (2003) got a better correlation with
    #       the crab harvest by applying a year-dependent
    #       gear efficiency, but they only have values for the
    #       1990s.  So far, I haven't included this more
    #       detailed effect, but it would be interesting
    #       if the data were available.
    #       They also found that the gear efficiency was vessel
    #       dependent.  We do have vessel information, but it's
    #       not clear what the abbreviations mean.
    #       Zhang et al. (1993) found an exponential fall off
    #       in gear efficiency with increase crab density.  May
    #       consider implementing this in the future.
    #       From Kaufmann thesis (2014), Glenn should be able
    #       to provide gear and vessel specific gear efficiencies

    gear_efficiency = 0.15
    data['m0_per_area_corr'] = data['m0_per_area']/gear_efficiency
    data['m1_per_area_corr'] = data['m1_per_area']/gear_efficiency
    data['m2_per_area_corr'] = data['m2_per_area']/gear_efficiency
    data['f0_per_area_corr'] = data['f0_per_area']/gear_efficiency
    data['f1_per_area_corr'] = data['f1_per_area']/gear_efficiency
    data['f2_per_area_corr'] = data['f2_per_area']/gear_efficiency
    data['t_per_area_corr'] = data['t_per_area']/gear_efficiency

    # Correct for undetected juveniles
    # juvenile efficiency = 0.31 for m0 and f0 crabs
    # Source: Kaufmann (2015) thesis p.91 and previous methods section
    juvenile_efficiency=0.31
    data['m0_per_area_corr'] = data['m0_per_area_corr']/juvenile_efficiency
    data['f0_per_area_corr'] = data['f0_per_area_corr']/juvenile_efficiency

    # Get rid of unreasonable numbers for salinity
    # Units: parts per thousand
    ind = np.where((data.sal == -9) | (data.sal == 1e16))[0]
    data.loc[ind, 'sal'] = float('NaN')

    # Get rid of unreasonable numbers for water temperature
    # Units: Celsius
    ind = np.where((data.wtemp == -9) | (data.wtemp > 40))[0]
    data.loc[ind, 'wtemp'] = float('NaN')

    # Get rid of unreasonable numbers for depth
    # Units: Not sure, but probably feet
    ind = np.where(data.depth < 0)[0]
    data.loc[ind, 'depth'] = float('NaN')

    # Write CSV File
    # Raw crab counts per meter squared
    # And crab counts per meter squared corrected by gear efficiency
    #     and juvenile efficiency
    #print("Writing: "+outfn+" "+str(datetime.datetime.now()))
    data.to_csv('CrabCounts.csv', index=False)


    #
    # Calculate total number of crabs per year, corrected
    #     for gear efficiency and juvenile efficiency
    #

    # Area of Chesapeake Bay
    #     = 9182 km^2
    # Source: Sharov et al. (2003; Abundance and exploitation rate
    #     of the blue crab (callinectes sapidus) in Chesapeake Bay
    area_bay = 9812.0 # per square kilometer
    area_bay = area_bay*((1e3**2)) # per square meter
    
    # Calculate average total number of crabs per square meter
    data = data.groupby(['year']).mean().reset_index()
    data.drop(['lat', 'long'], inplace=True, axis=1)

    # Estimate total number of crabs in entire bay
    data['M0 Crabs']=data['m0_per_area_corr']*area_bay
    data['M1 Crabs']=data['m1_per_area_corr']*area_bay
    data['M2 Crabs']=data['m2_per_area_corr']*area_bay
    data['F0 Crabs']=data['f0_per_area_corr']*area_bay
    data['F1 Crabs']=data['f1_per_area_corr']*area_bay
    data['F2 Crabs']=data['f2_per_area_corr']*area_bay
    data['Total Crabs']=data['t_per_area_corr']*area_bay
    data.drop(['m0_per_area_corr'], inplace=True, axis=1)
    data.drop(['m1_per_area_corr'], inplace=True, axis=1)
    data.drop(['m2_per_area_corr'], inplace=True, axis=1)
    data.drop(['f0_per_area_corr'], inplace=True, axis=1)
    data.drop(['f1_per_area_corr'], inplace=True, axis=1)
    data.drop(['f2_per_area_corr'], inplace=True, axis=1)
    data.drop(['t_per_area_corr'], inplace=True, axis=1)
    data.drop(['m0_per_area'], inplace=True, axis=1)
    data.drop(['m1_per_area'], inplace=True, axis=1)
    data.drop(['m2_per_area'], inplace=True, axis=1)
    data.drop(['f0_per_area'], inplace=True, axis=1)
    data.drop(['f1_per_area'], inplace=True, axis=1)
    data.drop(['f2_per_area'], inplace=True, axis=1)
    data.drop(['t_per_area'], inplace=True, axis=1)

    # Make nicer column names
    data.rename(columns={'year': 'Year'}, inplace=True)

    # Write CSV File
    # Total crab counts in entire bay, corrected by gear efficiency
    #print("Writing: "+outfn+" "+str(datetime.datetime.now()))
    data.to_csv('CrabCountsAveraged.csv', index=False)

    
# End Main
#####################################################


######################################################
# Some converters for pandas 
#

# Strip whitespace from locid
def convert_strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text

# Convert latitude and longitude from
# degree + decimal minutes
# to
# decimal degrees
def convert_latlon(latlon):
    decimal_min = float(latlon[2:6])/100.0
    decimal_degrees = float(latlon[0:2])
    return decimal_degrees + decimal_min/60.0


# End Some converters for pandas 
#####################################################


if __name__ == '__main__':
    main()


