#! /usr/bin/env python2.7

#####################################################
# Get SAV data
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
    # Get SAV data
    #
    fn = "SAV Area by Quadrangle (1971-2014).csv"
    sdata = pd.read_csv(fn,
                        header = 0, skiprows = 1,
                        dtype = {'Quadid': np.int32, 'Quad_Name': str, '1971_ha': np.float64 ,
                                 '1971_nd': str , '1974_ha': np.float64 , '1974_nd': str ,
                                 '1978_ha': np.float64 , '1978_nd': str , '1979_ha': np.float64 ,
                                 '1979_nd': str , '1980_ha': np.float64 , '1980_nd': str ,
                                 '1981_ha': np.float64 , '1981_nd': str , '1984_ha': np.float64 ,
                                 '1984_nd': str , '1985_ha': np.float64 , '1985_nd': str ,
                                 '1986_ha': np.float64 , '1986_nd': str , '1987_ha': np.float64 ,
                                 '1987_nd': str , '1989_ha': np.float64 , '1989_nd': str ,
                                 '1990_ha': np.float64 , '1990_nd': str , '1991_ha': np.float64 ,
                                 '1991_nd': str , '1992_ha': np.float64 , '1992_nd': str ,
                                 '1993_ha': np.float64 , '1993_nd': str , '1994_ha': np.float64 ,
                                 '1994_nd': str , '1995_ha': np.float64 , '1995_nd': str ,
                                 '1996_ha': np.float64 , '1996_nd': str , '1997_ha': np.float64 ,
                                 '1997_nd': str , '1998_ha': np.float64 , '1998_nd': str ,
                                 '1999_ha': np.float64 , '1999_nd': str , '2000_ha': np.float64 ,
                                 '2000_nd': str , '2001_ha': np.float64 , '2001_nd': str ,
                                 '2002_ha': np.float64 , '2002_nd': str , '2003_ha': np.float64 ,
                                 '2003_nd': str , '2004_ha': np.float64 , '2004_nd': str ,
                                 '2005_ha': np.float64 , '2005_nd': str , '2006_ha': np.float64 ,
                                 '2006_nd': str , '2007_ha': np.float64 , '2007_nd': str ,
                                 '2008_ha': np.float64 , '2008_nd': str , '2009_ha': np.float64 ,
                                 '2009_nd': str , '2010_ha': np.float64 , '2010_nd': str ,
                                 '2011_ha': np.float64 , '2011_nd': str , '2012_ha': np.float64 ,
                                 '2012_nd': str , '2013_ha': np.float64 , '2013_nd': str ,
                                 '2014_ha': np.float64 , '2014_nd': str})

    # Drop specific places you can't match to quadrangles
    sdata = sdata.drop(179)
    
    # Get quadrangle data
    fn = 'map_indexes_24KMD/map_indexes.csv'
    qdata_md = pd.read_csv(fn,
                           header = 0,
                           usecols = ['QUADNAME', 'BOTTOM', 'TOP_', 'LEFT_', 'RIGHT_'],
                           dtype = {'QUADNAME': str, 'BOTTOM': np.float64, 'TOP_': np.float64,
                                    'LEFT_': np.float64, 'RIGHT_': np.float64})
    fn = 'map_indexes_24KVA/map_indexes.csv'
    qdata_va = pd.read_csv(fn,
                           header = 0,
                           usecols = ['QUADNAME', 'BOTTOM', 'TOP_', 'LEFT_', 'RIGHT_'],
                           dtype = {'QUADNAME': str, 'BOTTOM': np.float64, 'TOP_': np.float64,
                                    'LEFT_': np.float64, 'RIGHT_': np.float64})
    fn = 'map_indexes_24KDC/map_indexes.csv'
    qdata_dc = pd.read_csv(fn,
                           header = 0,
                           usecols = ['QUADNAME', 'BOTTOM', 'TOP_', 'LEFT_', 'RIGHT_'],
                           dtype = {'QUADNAME': str, 'BOTTOM': np.float64, 'TOP_': np.float64,
                                    'LEFT_': np.float64, 'RIGHT_': np.float64})
    qdata = pd.concat([qdata_md, qdata_va, qdata_dc])
    


    # Find SAV quadrangles in Maryland
    snames = np.unique(sdata.Quad_Name.values)
    mask = ['Md' in sn[sn.find(';')+2:len(sn)] for sn in sdata.Quad_Name.values]
    ind = np.where(mask)[0]
    sdata = sdata.iloc[ind]
    snames = np.unique(sdata.Quad_Name.values)

    # Strip state from SAV quadrangle names
    snames = [sn[0:sn.find(';')] for sn in sdata.Quad_Name.values]
    sdata['Quad_Name'] = snames
    
    # Convert "Mt." to "Mount" and "St." to "Saint"
    snames = [sn.replace("Mt.", "Mount") if ("Mt." in sn) else sn for sn in sdata.Quad_Name.values]
    sdata['Quad_Name'] = snames
    snames = [sn.replace("St.", "Saint") if ("St." in sn) else sn for sn in sdata.Quad_Name.values]
    sdata['Quad_Name'] = snames

    # Have to hand adjust a few names because they don't match exactly
    mask = [sn == 'Girdle Tree' for sn in sdata.Quad_Name.values]
    ind = np.where(mask)[0]
    sdata.Quad_Name.iloc[ind[0]] = 'Girdletree'
    
    # Match two data sets
    qnames = [qn.replace("St.", "Saint") if ("St." in qn) else qn for qn in qdata.QUADNAME.values]
    qdata['QUADNAME'] = qnames
    Nsamples = sdata.shape[0]
    slat = np.zeros((Nsamples, ), dtype = np.float64)
    slon = np.zeros((Nsamples, ), dtype = np.float64)
    for sn in sdata.Quad_Name.values:
        qmask = [(sn.lower() == qn.lower()) for qn in qdata['QUADNAME'].values]
        qind = np.where(qmask)[0][0]
        smask = [sdn == sn for sdn in sdata.Quad_Name.values]
        sind = np.where(smask)[0][0]
        slat[sind] = np.mean([qdata['BOTTOM'].iloc[qind], qdata['TOP_'].iloc[qind]])
        slon[sind] = np.mean([qdata['LEFT_'].iloc[qind], qdata['RIGHT_'].iloc[qind]])
    sdata['Latitude'] = slat
    sdata['Longitude'] = slon


    # Convert to format that you want
    data = pd.DataFrame()
    for yr in range(1990, 2015):
        ind = np.where((sdata[str(yr)+'_nd'].values != 'nd') & (sdata[str(yr)+'_nd'].values != 'pd'))[0]
        hold = sdata[['Latitude', 'Longitude', str(yr)+'_ha']].iloc[ind]
        hold.rename(columns={str(yr)+'_ha': 'SAV_ha'}, inplace = True)
        hold['Year'] = yr
        data = data.append(hold)

    # Save as CSV file
    data.to_csv('SAV.csv', index=False)


# End Main
#####################################################


if __name__ == '__main__':
    main()


