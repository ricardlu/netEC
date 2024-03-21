# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:27:35 2024

@author: lricard
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap


def cumulative_anomaly(data,domain):
    return np.nansum(data*domain,axis=(1,2))

def build_domain_map (d_maps):
    d,x,y = d_maps.shape
    domain_map_output = np.zeros((x,y))
    for d in range(len(d_maps)) [::-1]:     
        domain_map_output[d_maps[d] == 1] = d+1
    return domain_map_output

def build_strength_map (d_maps, strengths):
    d,x,y = d_maps.shape
    domain_map_output = np.zeros((x,y))
    for i in range(len(d_maps)):     
        domain_map_output[d_maps[i] == 1] = strengths[i]
    return domain_map_output


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
    #norm_signals = normalize (signals)
    return signals


def maps_to_plot (d_maps):
    d_maps_to_plot = d_maps.copy() * np.nan
    d_maps_to_plot[d_maps != 0] = d_maps[d_maps != 0]
    return d_maps_to_plot

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
    
    # Longitude-Latitude
    lat = np.arange(-59.,60.,2)
    lon = np.arange (0.,360.,2)
    LON, LAT = np.meshgrid(lon, lat)
    lat_rad = np.radians(lat)
    lat_weights = np.cos(lat_rad).reshape(len(lat_rad),1)
    
    # Open SST regions in HadISST : 30 regions over time period of 480 months
    d_maps1 = np.load('Data/regions_reference_HadISST_sst_1975-2014.npy')
    N_regions = len(d_maps1)
    print (N_regions)
    size_regions = np.array([np.nansum(d_maps1[i,]) for i in range (N_regions)])
    d_ids1 = np.arange(1, N_regions +1)
    domain_map1 = build_domain_map (d_maps1) # Shape (nlon, nlat) filled with 0 and 1
    domain_map1_plot = maps_to_plot (domain_map1) # Shape (nlon, nlat) filled with NaN and 1
    N_regions = domain_map1.max()
    
        
    # Signals 
    filename_HadISST = 'Data/processed_HadISST_sst_2x2_197501-201412.nc'
    signals_HadISST, signals_norm_HadISST = compute_signals (filename_HadISST, d_maps1)
    print (signals_HadISST.shape)
    
    filename_COBEv2 = 'Data/processed_COBEv2_sst_2x2_197501-201412.nc'
    signals_COBEv2, signals_norm_COBEv2 = compute_signals (filename_COBEv2, d_maps1)
    print (signals_COBEv2.shape)
    
    vect_time = pd.date_range(start='1/1/1975', end='1/1/2015', freq = 'M')
    print (len(vect_time))
    
    # Plot signals
    plt.figure(figsize = (12,3), dpi = 150)
    ax = plt.gca()
    ax.axhline (color = 'grey', linestyle = '--')
    plt.plot (vect_time, signals_HadISST[0,], marker = '.', color= 'b')
    plt.ylabel ('ENSO signal in HadISST', size = 13)
    plt.xlabel ('Time', size = 13)
    plt.grid(alpha = .5)
    plt.tight_layout()
    # plt.savefig('Figures/Signal_ENSO_in_HadISST_region' , dpi =300)
    
    
    
    # Nine regions
    nodes = np.array([0,1,3,4,5,6,7,8,9])
    N_nodes = len(nodes)
    domain_map2 = build_domain_map (d_maps1[nodes])  # Shape (nlon, nlat) filled with 0 and 1
    domain_map2_plot = maps_to_plot (domain_map2) # Shape (nlon, nlat) filled with NaN and 1
    
    # Plot reference regions 
    cmp = plt.cm.gist_ncar
    bounds = np.arange(1, N_regions+1)
    fig = plot (domain_map1_plot, cmp, '', 'id', bounds, bounds[::5], domain_map1, 'neither')
    # plt.savefig ('Figures/30_Regions_reference_HadISST', dpi = 300)
    
    # Plot subset nine reference regions for causal network 
    cmp = plt.cm.gist_ncar
    bounds = np.arange(1, N_nodes+1)
    fig = plot (domain_map2_plot, cmp, '', 'id', bounds, bounds[::1], domain_map2, 'neither')
    # plt.savefig ('Figures/9_Regions_reference_HadISST', dpi = 300)
    
    # Regions labeled with specific values 
    # Plot strength map 
    cmp = plt.cm.Reds
    bounds = np.array([0,1e2,1e3,1e4, 1e5, 1.5e5, 2e5])
    
    strength_regions_HadISST = np.load ('Data/strength_list_HadISST_sst_1975-2014.npy')[:,1]
    strength_map = build_strength_map (d_maps1, strength_regions_HadISST )
    strength_map_plot = maps_to_plot (strength_map)
    
    fig = plot (strength_map_plot, cmp, '',  'strength', bounds, bounds, domain_map1_plot, 'neither')
    plt.tight_layout()
    # plt.savefig ('Figures/Strength_in_HadISST_30regions', dpi = 300)
    
    # Plot ACE map
    cmp = plt.cm.YlOrRd
    bounds = np.arange(0,0.06, .005)
    
    ACE_regions_HadISST = np.load ('Data/ACE_9_regions_HadISST.npy')
    ACE_map = build_strength_map (d_maps1[nodes], ACE_regions_HadISST )
    ACE_map_plot = maps_to_plot (ACE_map )
    
    fig = plot ( ACE_map_plot, cmp, '',  'ACE', bounds, bounds, domain_map2_plot, 'neither')
    plt.tight_layout()
    # plt.savefig ('Figures/ACE_in_HadISST_9regions', dpi = 300)
    
    

    
    
    
    
    