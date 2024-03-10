# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:12:14 2024

@author: lricard

Compute two distance metrics NWWD and D_ACE 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import tigramite
from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp
from tigramite.independence_tests import ParCorr
from tigramite.models import LinearMediation
from tigramite.causal_effects import CausalEffects
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from tigramite import plotting as tp

def cumulative_anomaly(data,domain):
    return np.nansum(data*domain,axis=(1,2))

def average_anomaly(data,domain):
    return np.nanmean(data*domain,axis=(1,2))

def maps_to_plot (d_maps):
    d_maps_to_plot = d_maps.copy() * np.nan
    d_maps_to_plot[d_maps != 0] = d_maps[d_maps != 0]
    return d_maps_to_plot

def build_domain_map (d_maps):
    d,x,y = d_maps.shape
    domain_map_output = np.zeros((x,y))
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
        signals.append(cumulative_anomaly(sst_filled,weighted_domains[i]))
    signals = np.array(signals)
    norm_signals = normalize (signals)
    return norm_signals

def causal_strength (signals_PCMCI, link_dict):
    ## Causal Effects 
    df= pp.DataFrame(signals_PCMCI.T, datatime = np.arange(len(signals_PCMCI.T)), var_names= d_ids1)
    med = LinearMediation(dataframe=df)
    med.fit_model(all_parents = link_dict, tau_max=3) 
    ACE = med.get_all_ace()
    return ACE
    
def count_past_links (N_nodes, tau_max, dict_past_links):
    count = 0
    for i in range (N_nodes):
        count += len(dict_past_links[i])
    total_links = N_nodes*(N_nodes-1)*tau_max 
    proportion = 100*count/total_links
    return count, proportion


def dag_to_links (dag):

    N = dag.shape[0]
    parents = dict([(j, []) for j in range(N)])

    stock1 =[]
    stock2 =[]

    for (i, j, tau) in zip(*np.where(dag=='-->')):
        parents[j].append((i, -tau))
        
    print ('number undirected links = %s' %len(np.where(dag=='o-o')[0]))
    number = 0
    
    for (i, j, tau) in zip(*np.where(dag=='o-o')):
        if (i,j) not in stock1 :
            parents[j].append((i, tau))
            number +=1
            stock1.append((i,j))
            stock1.append((j,i))
    print ('number check = %s, %s' %(number, 2*number))
    print ('number conflicting links = %s' %len(np.where(dag=='x-x')[0]))
    number2 = 0
    
    for (i, j, tau) in zip(*np.where(dag=='x-x')):
        if (i,j) not in stock2 :
            parents[j].append((i, tau))
            number2 +=1
            stock2.append((i,j))
            stock2.append((j,i))
    print ('number2 check = %s, %s' %(number2, number2*2))
    return parents


def apply_PCMCI (signals_PCMCI, names):
    N, T = signals_PCMCI.shape
    df= pp.DataFrame(signals_PCMCI.T, datatime = np.arange(len(signals_PCMCI.T)), var_names= names)
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr())
    
    ##PCMCI
    alpha_lev = 0.05
    res = pcmci.run_pcmci(tau_min=0, tau_max=3, alpha_level = alpha_lev, pc_alpha= 0.05)
    # res = pcmci.run_pcmciplus(tau_min=0, tau_max=3, pc_alpha= [0.01, 0.02, 0.05, 0.10, 0.20, 0.30])

    causal_graph =  res ['graph']
    link_dict = dag_to_links(causal_graph)
    # Link dict without the contemporanous (unoriented) links
    link_dict_past = pcmci.return_parents_dict(graph = res['graph'], val_matrix = res['val_matrix'])
    return df, res, link_dict, link_dict_past

def build_strength_map (d_maps, strengths):
    #causal strengths are the ACE
    dom_strength = []
    for i in range(len(strengths)):
        dom_strength.append(d_maps[i]*strengths[i])
    dom_strength = np.array(dom_strength)
    strength_map = np.nanmax(dom_strength,axis = 0)
    return strength_map

def distance_maps (strength_map1, strength_map2) :
    numerator = np.nansum(np.abs(strength_map1 - strength_map2))
    np.random.seed(14)
    strength_map_bis1 = np.random.permutation (strength_map1)
    np.random.seed(18)
    strength_map_bis2 = np.random.permutation (strength_map2)
    denominator = np.nansum(np.abs(strength_map_bis1 - strength_map_bis2))
    D = numerator/denominator
    return D

def plot (field, cmp, title, cbar_label, bounds, bounds2, domains, ext) :
    plt.figure(figsize = (10,6))
    m = Basemap(projection='cyl', llcrnrlat=-60,urcrnrlat=60, llcrnrlon=0,urcrnrlon=360)
    xlon, ylat = m(LON, LAT) 
    m.pcolormesh (xlon,ylat, field, cmap = cmp, norm  = mpl.colors.BoundaryNorm(bounds, cmp.N), shading='auto')
    cbar = plt.colorbar(orientation = 'horizontal',  ticks = bounds2, 
                        extend = ext, aspect = 30, pad = 0.08) 
    cbar.set_label(cbar_label, size = 15, style = 'italic')
    cbar.ax.tick_params(labelsize = 13)
    m.drawcoastlines(linewidth = 2)
    m.drawparallels(np.arange(-90.,100.,30.),labels=[1,0,0,0],fontsize = 13,linewidth = 1)
    m.drawmeridians(np.arange(0.,360.,60.), labels=[0,0,0,1],fontsize = 13,linewidth = 1)
    m.fillcontinents(color = 'black')
    plt.title(title, style = 'italic', size = 20)
    plt.tight_layout(rect = (0,0,1,0.95))
    


if __name__ == '__main__':   
    %matplotlib qt
    
    rep = 'D_ACE_9_regions/'
    
    # Longitude-Latitude
    lat = np.arange(-59.,60.,2)
    lon = np.arange (0.,360.,2)
    LON, LAT = np.meshgrid(lon, lat)
    lat_rad = np.radians(lat)
    lat_weights = np.cos(lat_rad).reshape(len(lat_rad),1)

    # Reference SST regions, inferred in reanalysis HadISST with delta-MAPS
    d_maps1  = np.load ('Data/regions_reference_HadISST_sst_1975-2014.npy')[nodes,]
    nodes = np.array([0,1,3,4,5,6,7,8,9])
    
    domain_map1 = build_domain_map (d_maps1)
    domain_map1_plot = maps_to_plot (domain_map1)
    N_regions = domain_map1.max()
    print ('Number regions = ', N_regions)

    # Timeseries of cumulative SST anomalies in regions 
    filename_HadISST = 'Data/processed_HadISST_sst_2x2_197501-201412.nc'
    signals_HadISST = compute_signals (filename_HadISST, d_maps1)
    print (signals_HadISST.shape)
    
    filename_COBEv2 = 'Data/processed_COBEv2_sst_2x2_197501-201412.nc'
    signals_COBEv2 = compute_signals (filename_COBEv2, d_maps1)
    print (signals_COBEv2.shape)
    
    # We apply PCMCI and get the ACE
    df_HadISST, res_HadISST, link_dict_HadISST, link_dict_past_HadISST = apply_PCMCI (signals_HadISST, d_ids1)
    array_ACE_HadISST  = causal_strength (signals_HadISST, link_dict_past_HadISST)
    ACE_strength_map1 = build_strength_map(d_maps1, array_ACE_HadISST)
    # np.save ('Data/ACE_9_regions_HadISST', array_ACE_HadISST)
 
    df_COBEv2, res_COBEv2, link_dict_COBEv2, link_dict_past_COBEv2 = apply_PCMCI (signals_COBEv2, d_ids1)
    array_ACE_COBEv2  = causal_strength (signals_COBEv2, link_dict_past_HadISST)
    ACE_strength_map2 = build_strength_map(d_maps1, array_ACE_COBEv2)
    # np.save ('Data/ACE_9_regions_COBEv2', array_ACE_COBEv2)
    
    # np.save (rep + 'weightsACE_HadISST_1975-2014.npy',  weights_ACE_HadISST)
    # np.save (rep + 'weightsACE_COBEv2_1975-2014.npy',  weights_ACE_COBEv2)
    
    indsort_ACE = np.argsort(array_ACE_HadISST)
    names_reg = np.array(['ENSO', 'WPs', 'IO', 'WPn', 'IOWP','TAn', 'SO', 'TAs', 'TPn'])
    

    # Linear Mediation to infer causal effects 
    df= pp.DataFrame(signals_HadISST.T, datatime = np.arange(len(signals_HadISST.T)), var_names= d_ids1)
    med = LinearMediation(dataframe=df)
    med.fit_model(all_parents = link_dict_past_HadISST, tau_max=3) 
    ACE = med.get_all_ace()
    
    # Causal effects in a matrix 
    n_nodes = len(signals_HadISST)
    mat_MCE = np.zeros((n_nodes, n_nodes)) 
    for node_parent in np.arange(0,n_nodes) :
        for node_child in np.arange(0,n_nodes) :
            if node_parent == node_child :
                mat_MCE[node_parent, node_child] = 0
            else :
                # Max causal effect 
                # mat_MCE[node_parent, node_child] = med.get_ce_max(i= node_parent, j=node_child)
                # Causal effect at a given lag 
                mat_MCE[node_parent, node_child] = med.get_ce(i= node_parent, tau = 1, j=node_child)

    # totCE = np.sum (np.abs(matrix_CE_absmax), axis = 1) 
    
    # Count the number and proportion of past cross-links 
    number_links, proportion_links = count_past_links (9, 3, link_dict_past_HadISST)
    print ('Number links =', number_links)
    print ('Proportion links discovered = %s %%' %np.round(proportion_links,2))
    
    
    # To plot 
    # tp.plot_graph(
    #     figsize=(6, 6),
    #     val_matrix= res_HadISST['val_matrix'],
    #     graph= res_HadISST['graph'],
    #     var_names= names_reg ,
    #     link_colorbar_label='cross-MCI',
    #     node_colorbar_label='auto-MCI',
    #     node_size = 0.4,
    #     node_label_size = 11,
    #     label_fontsize = 14,
    #     ); plt.show()

    # Calculate D_AC COBEv2 vs HadISST
    D_ACE_obs = distance_maps (ACE_strength_map1, ACE_strength_map2)
    print (D_ACE_obs.round(2))

    
    # Calculate D_ACE model output vs HadISST
    list_model_names = ['NorESM_CMIP6', 'NorESM-LM_CMIP6', 'MPI_CMIP6', 'MPI-LR_CMIP6', 'EC-Earth_CMIP6',
                         'EC-Earth-Veg-LR_CMIP6', 'MIROC_CMIP6', 'GFDL-ESM4_CMIP6', 
                         'GISS_CMIP6', 'HadGEM3_CMIP6', 'CNRM-CM6_CMIP6', 'CanESM5_CMIP6',  'INM-CM5_CMIP6', 
                         'UKESM1-0-LL_CMIP6', 'IPSL_CMIP6',  'NESM3_CMIP6',  'MRI-ESM2-0_CMIP6',
                         'BCC-ESM1_CMIP6', 'BCC-CSM2-MR_CMIP6',  'EC-Earth_AerChem_CMIP6',
                         'SAMO_CMIP6', 'ACCESS-ESM1-5_CMIP6', 'FGOALS-f3-L_CMIP6', 'ESM-1-0_CMIP6', 
                         'CAMS-CSM1-0_CMIP6', 'CESM2_CMIP6', 'CESM2_WACCM_CMIP6']
    N_models = len (list_model_names)
    
    #Stock list size n_models  containing matrices different shapes n_runs*n_regions
    D_ACE_model = []
    
    list_model_names =   []
    
    for i, model_name in enumerate(list_model_names):
     
        vect_D_ACE = []
        
        table_runs = pd.read_csv('Data/List_historical_runs.txt', sep=" ", index_col =["Model"]).T
        N, runs, filename_simu2, filename_simu = table_runs [model_name]
        les_runs = list(map(int, runs.split(',')))
        
        print (model_name + ' len(runs)= %s' %len(les_runs))

        for run in les_runs: 
            print('\nrun =', run)

            filename = '%s_r%s_1975-2014_sst' %(model_name, run)
            d_ids3 = np.copy(d_ids1)
            d_maps3 = np.copy(d_maps1)
            signals_simu = compute_signals ('../../Data_1975-2014/' + model_name + '/' + filename_simu %run, d_maps3)[select_nodes, :]

            array_ACE_simu  = causal_strength (signals_simu, link_dict_past_HadISST)
           
            # Distance maps
            ACE_strength_map3 = build_strength_map(d_maps3, array_ACE_simu)
            D_ACE = distance_maps (ACE_strength_map1, ACE_strength_map3)
            # D_ACE = distance_maps (ACE_strength_map2, ACE_strength_map3)
            vect_D_ACE.append (D_ACE)
            
        D_ACE_model.append  (vect_D_ACE)  #length run

    np.save (rep + 'D_ACE_links_and_regions_HadISST_1975-2014.npy', D_ACE_model) # list 27 models containing nruns values
 