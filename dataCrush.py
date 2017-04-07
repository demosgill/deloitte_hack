import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import matplotlib as mp
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 15}

label_size = 12
mp.rcParams['xtick.labelsize'] = label_size
mp.rcParams['ytick.labelsize'] = label_size


############################################
#        Delloite Hackathon. 2017.         #
############################################

path = '/Users/demos/Desktop/deloitte /Data/'

def getFeaturesFromCountry(data, country='Zimbabwe'):
    """
    Get features from country
    """
    return data[data.Cname==country].copy()


############################################
def getFeaturesFromVacineTrype(data, vaccine='DTP3'):
    """
    Get all results for a given vaccine
    """
    return data[data.Vaccine==vaccine].copy()


############################################
def plotFeatureByRegion(data, feat):
    data.groupby(feat).mean().T.boxplot()
    plt.title('Feature: %s'%feat)
    plt.legend('')
    plt.ylim([0,100])
    plt.tight_layout()


############################################
def plotMeanDeathRatePerCountry(data, year='2010', getdata=False, plot=True):

    """
    Get all results for a given vaccine
    """

    cleanData = data.groupby('Country Name').mean().T['2000':'2014'].dropna(axis=1).copy()
    cleanData.head()

    y1, y2 = year, year

    meanDeathRateUp = pd.DataFrame(cleanData[y1:y2].mean().sort_values(ascending=False),
                                 columns=['(Mean) Death Rate (Up); year: %s'%y1])

    meanDeathRate = pd.DataFrame(cleanData[y1:y2].mean().sort_values(ascending=True),
                                 columns=['Mean Death Rate (low); year: %s'%y1])

    if plot == True:
        f,ax = plt.subplots(1,2,figsize=(8,4), sharey=True)
        ax[0].set_ylim([0, 20])
        meanDeathRateUp[0:10].plot(kind='bar', ax=ax[0], sharey=True)
        meanDeathRate[0:10].plot(kind='bar', ax=ax[1], sharey=True)
        ax[0].set_xlabel(''); ax[1].set_xlabel('')
        ax[1].set_ylim([0, 20])
        plt.ylabel('Percentage')
        plt.tight_layout()
    else:
        pass

    if getdata == False:
        pass
    else:
        return meanDeathRateUp, meanDeathRate


############################################
def computeEfectvinessRatioPoliciesPerRegion(Region='WEST AFRICA'):
    """
    :param Region: define region of interest
    :return: dataframe with disbursement
    A disbursement is a form of payment from a
    public or dedicated fund, or a payment sought
    from a client to pass on to a third party.
    """
    val2 = 'NGO_DataDisbursement.csv'
    data2 = pd.read_csv(path+val2)

    df = data2[data2.Region == Region].groupby('Programme').mean().copy()
    df = pd.DataFrame(df[df.columns[0]])
    df.columns = [Region]

    return df


############################################
def CoverageRatePerVaccine(year='2015'):
    # Select a dataset
    val = 'WHO_Coverage_estimates.csv'
    data = pd.read_csv(path + val)

    vcns = data.Vaccine.unique() # Unique vacinnes

    R = pd.DataFrame()
    for i in range(len(vcns)):
        res = getFeaturesFromVacineTrype(data, vaccine=vcns[i])
        r = res.groupby('Region').mean();
        r = r[r.columns[1:]]
        r = pd.DataFrame(r['2015']).T
        R = pd.concat([R, r], axis=0)

    R.index = vcns

    return R