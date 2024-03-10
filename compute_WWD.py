# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:45:01 2024

@author: lricard
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
from sklearn.metrics.cluster import normalized_mutual_info_score 
import matplotlib as mpl
import cmasher as cmr
import scipy
from scipy.stats import spearmanr, kurtosis, skew, moment
import seaborn as sns

#Compute signals
def cumulative_anomaly(data,domain):
    return np.nansum(data*domain,axis=(1,2))

def average_anomaly(data,domain):
    return np.nanmean(data*domain,axis=(1,2))

def maps_to_plot (d_maps):
    d_maps_to_plot = d_maps.copy() * np.nan
    d_maps_to_plot[d_maps != 0] = d_maps[d_maps != 0]
    return d_maps_to_plot

def build_domain_map (d_maps):
    #Build domain map[x,y] from d_maps [d,x,y]
    #Put 0 if grid cells belongs to any domains and i if belongs to i-th domain
    d,x,y = d_maps.shape
    domain_map_output = np.zeros((x,y))
    #Read in reverse direction because index 0 is stronger than index 31
    for d in range(len(d_maps)) [::-1]:     
        domain_map_output[d_maps[d] == 1] = d+1
    return domain_map_output


def normalize (signals):
    stds = np.std(signals, axis = 1)
    signals_norm = []
    for i in range (len(signals)) :
        signals_norm.append(signals[i,:]/stds[i])
    signals_norm = np.array(signals_norm)
    return signals_norm

def compute_signals (inname, d_maps):
    file= Dataset(inname, 'r')
    try :
        sst = file['sst'][:]
    except :
        sst = file ['tos'][:]
    sst_filled = np.ma.filled(sst.astype(float), np.nan)
    print(np.nanmax(sst_filled).round(3))
    print(np.nanmin(sst_filled).round(3))
    print('\n')
    weighted_domains = d_maps*lat_weights 
    signals = []
    for i in range(len(weighted_domains)):
        # signals.append(average_anomaly(sst_filled,weighted_domains[i]))
        signals.append(cumulative_anomaly(sst_filled, weighted_domains[i]))
    signals = np.array(signals)
    norm_signals = normalize (signals)
    return signals, norm_signals


if __name__ == '__main__':   
    %matplotlib qt
    
    rep = 'WWD_30_regions/'
    
    # Longitude-Latitude
    lat = np.arange(-59.,60.,2)
    lon = np.arange (0.,360.,2)
    LON, LAT = np.meshgrid(lon, lat)
    lat_rad = np.radians(lat)
    lat_weights = np.cos(lat_rad).reshape(len(lat_rad),1)
    
    # 17 SST regions and 480 months
    d_maps1 = np.load ('Data/regions_reference_HadISST_sst_1975-2014.npy')
    N_regions = len(d_maps1)
    print ('Number SST regions = %s'%N_regions)
    d_ids1 = np.arange(1, N_regions +1)
    domain_map1 = build_domain_map (d_maps1)
    domain_map1_plot = maps_to_plot (domain_map1)
    
    strengths = np.load ('Data/strength_list_HadISST_sst_1975-2014.npy')
    weights_strengths = strengths[:,1]/np.sum(strengths[:,1])  #normalized strength
    weights_equal = np.ones((N_regions))*1/N_regions
    
    # Signals 
    filename_HadISST = 'Data/processed_HadISST_sst_2x2_197501-201412.nc'
    signals_HadISST, signals_norm_HadISST = compute_signals (filename_HadISST, d_maps1)
    print (signals_HadISST.shape)
    
    filename_COBEv2 = 'Data/processed_COBEv2_sst_2x2_197501-201412.nc'
    signals_COBEv2, signals_norm_COBEv2 = compute_signals (filename_COBEv2, d_maps1)
    print (signals_COBEv2.shape)
    
    vect_time = pd.date_range(start='1/1/1975', end='1/1/2015', freq = 'M')
    print (len(vect_time))
    
    
    # COBEv2 vs HadISST
    vect_WWD_obs = []
    for i in range (len(signals_HadISST)):
        WD_value = scipy.stats.wasserstein_distance (signals_HadISST[i,], signals_COBEv2[i,])
        WWD_value = WD_value*weights_strengths[i]     
        vect_WWD.append (WWD_value)
    vect_WWD_obs = np.array(vect_WWD_obs)
    # np.save (rep + 'WD_COBEv2_mean_regions_HadISST_1975-2014.npy', vect_WD)
 
    
    # Cumulative sum of normalized strength "weight strength" regions
    plt.figure(figsize = (4,3), dpi = 150)
    plt.plot (np.arange(1,N_regions+1), np.cumsum(weights_strengths), marker = '.', color= 'b')
    plt.ylabel ('Cumulative sum of strengths', size = 12)
    plt.xlabel ('SST regions')
    plt.grid()
    plt.tight_layout()


    
    ## Retrieve mean, std, skewness and kurtosis 
    # std_domains = np.std(signals_HadISST, axis = 1)
    # skewness_domains = scipy.stats.skew(signals_HadISST, axis = 1)
    # kurtosis_domains = scipy.stats.kurtosis(signals_HadISST, axis = 1)
    
    list_model_names =   ['NorESM_CMIP6', 'NorESM-LM_CMIP6', 'MPI_CMIP6', 'MPI-LR_CMIP6', 'EC-Earth_CMIP6',
                          'EC-Earth-Veg-LR_CMIP6', 'MIROC_CMIP6', 'GFDL-ESM4_CMIP6', 
                          'GISS_CMIP6', 'HadGEM3_CMIP6', 'CNRM-CM6_CMIP6', 'CanESM5_CMIP6',  'INM-CM5_CMIP6', 
                          'UKESM1-0-LL_CMIP6', 'IPSL_CMIP6',  'NESM3_CMIP6',  'MRI-ESM2-0_CMIP6',
                          'BCC-ESM1_CMIP6', 'BCC-CSM2-MR_CMIP6',  'EC-Earth_AerChem_CMIP6',
                          'SAMO_CMIP6', 'ACCESS-ESM1-5_CMIP6', 'FGOALS-f3-L_CMIP6', 'ESM-1-0_CMIP6', 
                          'CAMS-CSM1-0_CMIP6', 'CESM2_CMIP6', 'CESM2_WACCM_CMIP6']
   
    N_models = len (list_model_names)

    
    #Stock
   
    WWD_in_models = []    
    mat_contrib_in_models  = []

    for i, model_name in enumerate(list_model_names):
        print (model_name)
        # vect_WD_allruns = np.ones((N_regions))*np.nan

        vect_contrib_runs = np.ones((N_regions))*np.nan
        vect_WWD = []


        table_runs = pd.read_csv('Data/List_historical_runs.txt', sep=" ", index_col =["Model"]).T
        N, runs, filename_simu2, filename_simu = table_runs [model_name]
        les_runs = list(map(int, runs.split(',')))
        
        print (model_name + ' len(runs)= %s' %len(les_runs))
        
        for run in les_runs: 

            print('\nrun =', run)
            run_name = str(run)
            d_ids3 = np.copy(d_ids1)
            d_maps3 = np.copy(d_maps1)
            signals_simu, signals_norm_simu = compute_signals ('../Github/Data_1975-2014/' + model_name + '/' + filename_simu %run, d_maps3)
            
            vect_contrib = []
            
            sum_WWD = 0
            
  
            for k in range (len(signals_HadISST)):
                WD_value = scipy.stats.wasserstein_distance (signals_HadISST[k,], signals_simu[k,])
                # WD_value =  scipy.stats.wasserstein_distance (signals_COBEv2[k,], signals_simu[k,])
                WWD_value = WD_value*weights_strengths[k]     
                vect_contrib.append (WWD_value)
                sum_WWD += WWD_value
             
            WWD_run = sum_WWD
            vect_WWD.append (WWD_run)
                     
            vect_contrib_runs =  np.vstack((vect_contrib_runs, vect_contrib))
            vect_WWD_allruns = np.vstack((vect_WWD_allruns, vect_WWD))
      
            
        # Remove the first artefact and get n_runs*n_regions
        vect_contrib_runs = vect_contrib_runs[1:,:]
        vect_WWD_allruns = vect_WWD_allruns[1:,:]

        # Stock the matrics n_runs*n_regions in the list 
        WD_model_mean_regions.append (vect_WD_allruns)

        # WD_in_models.append (vect_WD_allruns)
        WWD_in_models.append (np.array(vect_WWD))
        mat_contrib_in_models.append (np.array(vect_contrib_runs))
    
  
    # np.save (rep + 'WWD_model_regions_HadISST_1975-2014.npy', WWD_in_models)
    # np.save (rep + 'Mat_contrib_WWD_model_regions_HadISST_1975-2014.npy', mat_contrib_in_models)

 


       
            

        

        
