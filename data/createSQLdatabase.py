#! /usr/bin/env python2.7

#####################################################
# Create and fill database with all your data
#
# MFreed 2015-Sept-14 Created
#####################################################

#
# Import Some Packages
#
import pandas as pd
import numpy as np
import MySQLdb


######################################################
# Main 
#
def main():

    #
    # Connect to MySQL
    #
    db = MySQLdb.connect(host = "localhost", user="crabby", passwd="crabby", db="crabs")
    #cur = db.cursor()

    
    #
    # Create a database for all crabs data
    # Note: This was done directly in MySQL by root user
    #       And "crabby" user was granted permission by root user
    #cur.execute('CREATE DATABASE crabs;')

    
    #
    # Create and fill in dredge table
    #
    data = pd.read_csv('CrabDredge/CrabCountsAveraged.csv',
                       header = 0,
                       usecols = ['Year', 'sal', 'wtemp', 'depth', 'M0 Crabs', 'M1 Crabs',
                                  'M2 Crabs', 'F0 Crabs', 'F1 Crabs', 'F2 Crabs', 'Total Crabs'],
                       dtype = {'Year': np.int16, 'sal': np.float64, 'wtemp': np.float64,
                                'depth': np.float64, 'M0 Crabs': np.float64, 'M1 Crabs': np.float64,
                                'M2 Crabs': np.float64, 'F0 Crabs': np.float64, 'F1 Crabs': np.float64,
                                'F2 Crabs': np.float64, 'Total Crabs': np.float64})
    data.rename(columns = {'M0 Crabs': 'M0_Crabs', 'M1 Crabs': 'M1_Crabs', 'M2 Crabs': 'M2_Crabs',
                           'F0 Crabs': 'F0_Crabs', 'F1 Crabs': 'F1_Crabs', 'F2 Crabs': 'F2_Crabs',
                           'Total Crabs': 'Total_Crabs'}, inplace = True)
    data.to_sql(con = db, name = 'DREDGE', if_exists = 'replace', flavor = 'mysql', index = False)


    #
    # Create and fill in detailed dredge table
    #
    data = pd.read_csv('CrabDredge/CrabCounts.csv',
                       header = 0,
                       usecols = ['date', 'sal', 'wtemp', 'depth',
                                  'year', 'lat', 'long', 'm0_per_area_corr',
                                  'm1_per_area_corr', 'm2_per_area_corr', 'f0_per_area_corr', 'f1_per_area_corr',
                                  'f2_per_area_corr', 't_per_area_corr'],
                       dtype = {'date': object, 'sal': np.float64, 'wtemp': np.float64,
                                'depth': np.float64, 'year': np.int32,
                                'lat': np.float64, 'long': np.float64, 'm0_per_area_corr': np.float64,
                                'm1_per_area_corr': np.float64, 'm2_per_area_corr': np.float64,
                                'f0_per_area_corr': np.float64, 'f1_per_area_corr': np.float64,
                                'f2_per_area_corr': np.float64, 't_per_area_corr': np.float64})
    data.rename(columns = {'long': 'lon'}, inplace = True)
    data.to_sql(con = db, name = 'DREDGEDETAIL', if_exists = 'replace', flavor = 'mysql', index = False)



    #
    # Create and fill in water quality station data
    #
    data = pd.read_csv('WaterQuality/WaterQualityStations.csv',
                       header = 0,
                       usecols = ['Station', 'Latitude', 'Longitude'],
                       dtype = {'Station': object, 'Latitude': np.float64, 'Longitude': np.float64})
    data = data.iloc[0:len(data)-1]
    data.to_sql(con = db, name = 'STATIONS', if_exists = 'replace', flavor = 'mysql', index = False)



    
    #
    # Create and fill in water quality data
    #
    #Parameter,Unit,Latitude,Longitude,MeasureValue,Year
    data = pd.read_csv('WaterQuality/WaterQualityAverage.csv',
                       header = 0,
                       dtype = {'MeasureValue': np.float64, 'Parameter': object,
                                'Unit': object, 'Latitude': np.float64,
                                'Longitude': np.float64, 'Year': np.int32})
    data.to_sql(con = db, name = 'WATERQUALITY', if_exists = 'replace', flavor = 'mysql', index = False)

    
    #
    # Create and fill in harvest table
    #
    data = pd.read_csv('CrabHarvest/CrabHarvest.csv',
                       names = ['Year', 'Harvest', 'Total Crabs'],
                       header = 0,
                       usecols = ['Year', 'Harvest', 'Total Crabs'],
                       dtype = {'Year': np.int16, 'Harvest': np.float64, 'Total Crabs': np.float64})
    data.rename(columns = {'Total Crabs': 'Total_Crabs'}, inplace = True)
    data.to_sql(con = db, name = 'HARVEST', if_exists = 'replace', flavor = 'mysql', index = False)

        
    #
    # Create and fill in sediment table
    #
    data = pd.read_csv('Sediment/Sediment.csv',
                       header = 0,
                       dtype = np.float64)
    data.to_sql(con = db, name = 'SEDIMENT', if_exists = 'replace', flavor = 'mysql', index = False)


    #
    # Create and fill in SAV (submerged aquatic vegetation) table
    #
    data = pd.read_csv('SAV/SAV.csv',
                       header = 0,
                       dtype = np.float64)
    data.to_sql(con = db, name = 'SAV', if_exists = 'replace', flavor = 'mysql', index = False)

    
    #
    # Create and fill in Landings table
    #
    data = pd.read_csv('Landings/Landings.csv',
                       header = 0,
                       dtype = {'Species': str, 'Landing_lbs': np.float64, 'Year': np.int32})
    data.to_sql(con = db, name = 'LANDINGS', if_exists = 'replace', flavor = 'mysql', index = False)


    #
    # Close database
    #
    cur = db.cursor()
    cur.execute("SHOW TABLES;")
    response = cur.fetchall()
    for row in response:
        print(row[0])
        cur.execute("SHOW COLUMNS FROM "+row[0]+";")
        print [col[0] for col in cur.fetchall()]
    cur.close()
    db.close()
    

# End Main
#####################################################


if __name__ == '__main__':
    main()


