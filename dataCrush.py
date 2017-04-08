import dataCrush as dc
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from os import listdir
import plotly as pl
from sklearn.ensemble import RandomForestRegressor

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

    """
    -> Grouping by region and showing all vaccines. <-
    :param year: Define a year for computing the mean
    :return: Dataframe with the mean values of the coverage rate
    """

    # Select a dataset
    val = 'WHO_Coverage_estimates.csv'
    data = pd.read_csv(path + val)

    vcns = data.Vaccine.unique() # Unique vacinnes

    R = pd.DataFrame()
    for i in range(len(vcns)):
        res = getFeaturesFromVacineTrype(data, vaccine=vcns[i])
        r = res.groupby('Region').mean();
        r = r[r.columns[1:]]
        r = pd.DataFrame(r[year]).T
        R = pd.concat([R, r], axis=0)

    R.index = vcns

    return R


############################################
def printDiseasePerCountry():

    """
    Print the top diseases per region:
    ----------------------------------
        - Idea is to see the top expenditure per disease
          per region.
    """
    # all regions within the dataset
    regios = ['EMRO', 'EURO', 'CENTRAL AFRICA',
              'SEARO', 'WEST AFRICA', 'PAHO',
              'WPRO', 'SOUTH AFRICA', 'EAST AFRICA']

    # create DF
    DF = pd.DataFrame()

    # Iterate over regions
    for reg in regios:
        df = computeEfectvinessRatioPoliciesPerRegion(Region=reg)
        DF = pd.concat([DF, df], axis=1)

    # Printing func
    print('-> Top expenditure per desease per region <-')
    print('----')
    for i in range(len(DF.columns)):
        dis = pd.DataFrame(DF[DF.columns[i]].sort_values(ascending=False)).index[0]
        val = pd.DataFrame(pd.DataFrame(DF[DF.columns[i]].sort_values(ascending=False)).ix[1]).values[0]
        print('(disease -> %s)  --- (Region -> %s) --- (expenditure = %.1f USD Mi.)' % (dis, DF.columns[i], val))


############################################
def plotMeanDisbusementPerRegion(savePdf=False):

    #import portfolio_functions as pf
    # To be used on my pc only for normalizing data

    data2 = pd.read_csv(path + 'NGO_DataDisbursement.csv')

    gd = data2.groupby('Region').mean().T
    gd = gd.ix[gd.index[1:]]

    #gd = pf.normalize_data_for_comparacy(gd, sklearn=False)

    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    gd.plot(ax=ax, marker='o', linewidth=.5, alpha=0.9)
    plt.title(r'$Mean\, Disbursement\, per\, region$', fontsize=18)
    plt.ylabel('$Disbursement\, in\, USD\, Bi.$', fontsize=18)
    plt.legend(fancybox=True, framealpha=0.4, loc='upper left',
               fontsize=9)
    plt.grid(linewidth=.3)
    plt.tight_layout()

    if savePdf == False:
        pass
    else:
        plt.savefig('/Users/Demos/Desktop/cnt.pdf')

    return gd

############################################
def getRankedCountriesForGivenVaccine(data, vaccine='DTP3'):
    """
    Get all other features for a given vaccine.
    """
    df = getFeaturesFromVacineTrype(data, vaccine=vaccine)
    return df


############################################
def computeAndPlotAggCoveragePerRegion(year='2015', vaccine='DTP3'):
    # Select a dataset
    val = 'WHO_Coverage_estimates.csv'
    data = pd.read_csv(path + val)

    val2 = 'Death rate, crude (per 1,000 people).xls'
    data2 = pd.read_excel(path + val2)

    # PER REGION
    df = getRankedCountriesForGivenVaccine(data, vaccine=vaccine)
    aggData = df.groupby('Region').mean().T
    aggData = aggData.ix[aggData.index[1:-1]][::-1]

    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    df[df.columns[5:-1]].T[::-1].mean(axis=1).plot(kind='area', color='k',
                                                   alpha=0.3, ax=ax)
    df[df.columns[5:-1]].T[::-1].mean(axis=1).plot(color='k', marker='o', ax=ax,
                                                   alpha=0.5)
    aggData.plot(ax=ax)
    ax.grid(True)
    plt.title(r'$Mean(Aggregated$ $Coverage$ $Rate)$ $in$ $P.P.$', fontsize=18)
    plt.ylabel('$Percentage\, of\, coverage (\%)$', fontsize=18)
    plt.ylim([0, 100])
    plt.tight_layout()

    return aggData


############################################
def predictCovRatioForAfricanCountries(gd, aggData):

    """
    :param gd:
    :param aggData:
    :return: Predicting future trends of coverage ratio and
    """

    # DATA MUNGE: Selecting training and testing data
    x_train = gd[['WEST AFRICA', 'EAST AFRICA', 'SOUTH AFRICA', 'CENTRAL AFRICA']]['2003':'2015']
    x_train = x_train.values
    x_test = gd[['WEST AFRICA', 'EAST AFRICA', 'SOUTH AFRICA', 'CENTRAL AFRICA']]['2015':'2017'].values
    labels = aggData['AFR'][gd['WEST AFRICA'].index]['2003':'2015'].values

    # TRAIN AND PREDICT
    model2 = RandomForestRegressor()
    model2.fit(x_train, labels)
    pred = model2.predict(x_test)
    newDf = aggData['AFR'][gd['WEST AFRICA'].index]
    newDf['2016'] = pred[0];
    newDf['2017'] = pred[1]

    # PLOT
    # gd[['WEST AFRICA','EAST AFRICA', 'SOUTH AFRICA', 'CENTRAL AFRICA']].plot()
    gd[['WEST AFRICA', 'EAST AFRICA', 'SOUTH AFRICA', 'CENTRAL AFRICA']].plot()
    plt.legend(fancybox=True, framealpha=0.2, loc='upper left')
    plt.ylabel('$Expenditure$ $in$ $USD$ $Bi$', fontsize=16)
    ax = plt.twinx()
    newDf.plot(linestyle='--', color='k', ax=ax)
    newDf['2001':'2015'].plot(linestyle='-', color='k', ax=ax)
    aggData['AFR'][gd['WEST AFRICA'].index].plot(kind='area', alpha=0.2,
                                                 ax=ax, color='k')
    ax.set_ylabel('$Percentage$ $coverage$', fontsize=16)
    ax.set_ylim([60, 100])
    plt.tight_layout()
    # plt.savefig('/Users/demos/Desktop/deloitte /prediction.pdf')