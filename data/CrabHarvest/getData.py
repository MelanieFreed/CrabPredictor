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

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO


######################################################
# Main 
#
def main():

    #
    # Use corrected harvest values from Miller et al.
    # (2011; Stock assessment of the blue crab in the
    #  Chesapeake Bay)
    #

    # Get all text from pdf
    fn = 'references/Miller_etal_2001_ChesapeakeBayStockAssessment.pdf'
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = file(fn, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = [93,94,95] #set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)
    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()

    # Clean it up to get numbers
    # count is in metric tonnes x 10^3
    lines = text.replace('\xc2\xa0',' ').split('\n\n')
    year = lines[6].split(' \n')
    year = [int(yr) for yr in year]
    count = lines[19].split(' \n')
    count = [float(ct) for ct in count]
    hold = lines[38].split(' \n')
    hold = [int(yr) for yr in hold]
    year = year + hold
    hold = lines[45].split(' \n')
    hold = [float(ct) for ct in hold]
    count = count + hold
    hold = lines[55].split(' \n')
    hold = [int(yr) for yr in hold]
    year = year + hold
    hold = lines[60].split(' \n')
    hold = [float(ct) for ct in hold]
    count = count + hold

    # Convert count from 10^3 metric tonnes to pounds
    count = [ct*2204622.8 for ct in count]
    
    # Convert to dataframe
    data = pd.DataFrame()
    data['Year'] = year
    data['Harvest_lbs'] = count # pounds


    #
    # Manually added data from more recent years.
    # Taken from annual Blue Crab Advisory Reports
    #

    # 2009
    # http://chesapeakebay.noaa.gov/images/stories/fisheries/keyFishSpecies/cbsacreport2010.pdf
    #data = data.append(pd.Series({'Year': 2009, 'Harvest_lbs': 28500000.0}), ignore_index=True)

    # 2010
    # http://www.chesapeakebay.noaa.gov/images/stories/fisheries/keyFishSpecies/cbsacreport2011.pdf
    data = data.append(pd.Series({'Year': 2010, 'Harvest_lbs': 53400000.0}), ignore_index=True)

    # 2011
    # http://www.chesapeakebay.net/documents/CBSAC_Final_Advisory_Report_2012_July_20th_2012.pdf
    data = data.append(pd.Series({'Year': 2011, 'Harvest_lbs': 35300000.0}), ignore_index=True)

    # 2012
    # http://www.chesapeakebay.net/documents/Final_CBSAC_Advisory_Report_2013_.pdf
    data = data.append(pd.Series({'Year': 2012, 'Harvest_lbs': 31000000.0}), ignore_index=True)
  
    # 2013
    # http://www.chesapeakebay.net/documents/CBSAC_2014_Blue_Crab_Advisory_Report_Final_June30th_2014.pdf
    data = data.append(pd.Series({'Year': 2013, 'Harvest_lbs': 18700000.0}), ignore_index=True)
      
    # 2014
    # http://www.chesapeakebay.net/documents/CBSAC_2015_Advisory_Report_6-30_FINAL.pdf
    data = data.append(pd.Series({'Year': 2014, 'Harvest_lbs': 16500000.0}), ignore_index=True)


    #
    # Apply correction: recreation harvest is estimated as 8% of commercial harvest
    # Source: Miller et al. (2011; Stock assessment of blue crab in chesapeake bay)
    #
    data['Harvest_lbs'] = data['Harvest_lbs']*1.08

    #
    # Convert pounds to number of crabs
    # For Maryland dredge survey:
    #    Average male weight = 0.421 lbs
    #    Average female weight = 0.310 lbs
    # Source: Miller et al. (2011; Stock assessment of blue crab in Chesapeake Bay, p.34)
    #

    # Get male/female ratio from dredge survey data
    hold = pd.read_csv('~/insight/Project/data/CrabDredge/CrabCountsAveraged.csv',
                       header = 0,
                       usecols = ['Year', 'M0 Crabs', 'M1 Crabs', 'M2 Crabs', 'F0 Crabs', 'F1 Crabs', 'F2 Crabs'],
                       dtype = {'Year': np.int32, 'M0 Crabs': np.float64, 'M1 Crabs': np.float64,  
                                'M2 Crabs': np.float64, 'F0 Crabs': np.float64, 'F1 Crabs': np.float64,
                                'F2 Crabs': np.float64})
    hold['Male'] = hold['M0 Crabs'] + hold['M1 Crabs'] + hold['M2 Crabs']
    hold['Female'] = hold['F0 Crabs'] + hold['F1 Crabs'] + hold['F2 Crabs']
    hold['fracM'] = hold['Male']/(hold['Male'] + hold['Female'])
    hold['fracF'] = hold['Female']/(hold['Male'] + hold['Female'])
    hold.drop(['M0 Crabs', 'M1 Crabs', 'M2 Crabs', 'F0 Crabs', 'F1 Crabs', 'F2 Crabs', 'Male', 'Female'],
              inplace = True, axis = 1)

    # Calculate average weight
    hold['Weight'] = hold['fracM']*0.421 + hold['fracF']*0.310
    hold.drop(['fracM', 'fracF'], inplace = True, axis=1)
    
    # Some resources if you want to go into more detail later:
    # 1. Miller et al. (2011; Stock assessment of blue crab in Chesapeake Bay, p.34)
    # Also see sex-specific landings and estimates of weight/carapace of male/female crabs
    # here.
    # 2. Calculate average carapace width of crabs in the population
    # See Sharov et al. (2003; ABUNDANCE AND EXPLOITATION RATE OF THE BLUE CRAB
    # (CALLINECTES SAPIDUS) IN CHESAPEAKE BAY) for distribution of CW in the blue crab
    # population.  It's bimodal.  Crabs with <=
    # 3. For crabs > 120 mm CW, assume average CW = 141 mm (rough eyeball estimate from source)
    # Source: Miller et al. (2001; The Precautionary Approach to Managing the Blue Crab in
    # Chesapeake Bay: Establishing Limits and Targets)

    # Combine harvest and weight information
    data = pd.merge(data, hold, on=['Year'])

    # Calculate harvest in Number
    data['Harvest_num'] = data['Harvest_lbs']/data['Weight']
    data.drop(['Weight'], inplace = True, axis=1)
    
    #
    # Save as CSV file
    # Harvest is in pounds
    #
    data.to_csv('CrabHarvest.csv', index=False)


# End Main
#####################################################


if __name__ == '__main__':
    main()


