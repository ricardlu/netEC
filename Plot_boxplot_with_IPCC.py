# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:43:04 2024

@author: lricard
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import string

if __name__ == '__main__':   
     
    # stats = ['whislo', 'q1', 'med', 'q3', 'whisli', 'fliers']
    IPCC_AR5_ECS = [np.nan, 1.5, np.nan, 4.5, np.nan]
    IPCC_AR6_ECS = [2, 2.5, 3, 4, 5]
    unweighted_ECS = [2.25, 2.68, 3.76, 5.3, 5.63]
    weighted_ECS = [2.07, 2.35, 3.04, 4.81, 5.59]
    Cox_ECS = [np.nan, 1.2, 1.60, 1.99, np.nan]
    Nijsse_ECS  = [np.nan, 1.3, 1.68, 2.10, np.nan]
    
    IPCC_AR5_TCR = [np.nan, 1, np.nan, 2.5, np.nan]
    IPCC_AR6_TCR = [1.2, 1.4, 1.8, 2.2, 2.4]
    unweighted_TCR = [1.39, 1.53, 1.92, 2.66, 2.829]
    weighted_TCR = [1.52, 1.55, 1.86, 2.60, 2.76]
    Cox_TCR = [np.nan, 1.3, 1.60, 1.99, np.nan]
    Nijsse_TCR  = [np.nan, 1.30, 1.68, 2.10, np.nan]
    
    
    boxes_TCR =  { 'label' : ['IPCC AR5 \n (2013)',
                           'IPCC AR6 \n (2021)',
                          'Unweighted', 
                          'netEC',
                          'Tokarska et al.,\n 2020', 
                          'Nijsse et al., \n 2020'],
                'whislo': [IPCC_AR5_TCR[0], IPCC_AR6_TCR[0], unweighted_TCR[0], weighted_TCR[0], Cox_TCR[0], Nijsse_TCR[0]],   
                'q1'    : [IPCC_AR5_TCR[1], IPCC_AR6_TCR[1], unweighted_TCR[1], weighted_TCR[1], Cox_TCR[1], Nijsse_TCR[1]],  
                'med'   : [IPCC_AR5_TCR[2], IPCC_AR6_TCR[2], unweighted_TCR[2], weighted_TCR[2], Cox_TCR[2], Nijsse_TCR[2]],  
                'q3'    : [IPCC_AR5_TCR[3], IPCC_AR6_TCR[3], unweighted_TCR[3], weighted_TCR[3], Cox_TCR[3], Nijsse_TCR[3]],  
                'whishi': [IPCC_AR5_TCR[4], IPCC_AR6_TCR[4], unweighted_TCR[4], weighted_TCR[4], Cox_TCR[4], Nijsse_TCR[4]],  
                'fliers': np.ones(6)*np.nan
                }
            
    df_TCR = pd.DataFrame(boxes_TCR, columns = ['label', 'whislo', 'q1',  'med', 'q3',  'whishi', 'fliers'])
  
 
    
    boxes_ECS =  { 'label' : ['IPCC AR5 \n (2013)',
                           'IPCC AR6 \n (2021)',
                          'Unweighted', 
                          'netEC',
                          'Cox et al.,\n 2018', 
                          'Nijsse et al., \n 2020'],
                'whislo': [IPCC_AR5_ECS[0], IPCC_AR6_ECS[0], unweighted_ECS[0], weighted_ECS[0], Cox_ECS[0], Nijsse_ECS[0]],   
                'q1'    : [IPCC_AR5_ECS[1], IPCC_AR6_ECS[1], unweighted_ECS[1], weighted_ECS[1], Cox_ECS[1], Nijsse_ECS[1]],  
                'med'   : [IPCC_AR5_ECS[2], IPCC_AR6_ECS[2], unweighted_ECS[2], weighted_ECS[2], Cox_ECS[2], Nijsse_ECS[2]],  
                'q3'    : [IPCC_AR5_ECS[3], IPCC_AR6_ECS[3], unweighted_ECS[3], weighted_ECS[3], Cox_ECS[3], Nijsse_ECS[3]],  
                'whishi': [IPCC_AR5_ECS[4], IPCC_AR6_ECS[4], unweighted_ECS[4], weighted_ECS[4], Cox_ECS[4], Nijsse_ECS[4]],  
                'fliers': np.ones(6)*np.nan
                }
            
    df_ECS = pd.DataFrame(boxes_ECS, columns = ['label', 'whislo', 'q1',  'med', 'q3',  'whishi', 'fliers'])
    
    

    
    custom_lines = [Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='orange', lw=2)]
    
    np.random.seed(14)
    x = np.random.normal(0, 0.06, 27)
    nb = 6

  
 
    fig, axs= plt.subplot_mosaic([['a)'], ['b']], layout='constrained', figsize = (11,12))
    
    for n, (key, ax) in enumerate(axs.items()):
     
        ax.text(-0.02, 1.02, string.ascii_lowercase[n], transform=ax.transAxes, 
          size=15, weight='bold')
    
    plt.subplot (2,1,1)
    ax2 = plt.gca()
    ax2.bxp(df_ECS.to_dict('records'), 
            positions = np.arange (0.75, 0.75+nb*0.75, 0.75), 
            widths = 0.4, medianprops={"linewidth": 2},  
            meanprops={"linewidth": 2, "color": 'r'}, boxprops={"linewidth": 1.5},
            showmeans=False, meanline=False, showfliers=False)
    plt.ylabel('ECS[°C]', size = 18)
    plt.yticks(np.arange (1,6.1,0.5),fontsize = 13)
    plt.xticks(fontsize = 13)
    ax2.set_xlim([0.27,4.93])
    ax2.set_ylim([0.5, 6])
    ax2.axhline (weighted_ECS[2], linestyle = '--', linewidth = 2, alpha = 0.4, color = 'darkorange')
    ax2.fill_between (np.array([0.25,4.95]), weighted_ECS[1], weighted_ECS[3], alpha=.2, linewidth=0)
    plt.legend(custom_lines, ['66%', 'Median'], loc = 'upper left', ncol = 2)
    plt.title ('\n Best estimates and likely ranges for ECS constrained', size = 16)

    
    plt.subplot (2,1,2)
    ax1 = plt.gca()
    ax1.bxp(df_TCR.to_dict('records'), 
             positions = np.arange (0.75, 0.75+nb*0.75, 0.75), 
             widths = 0.4, medianprops={"linewidth": 2},  
             meanprops={"linewidth": 2, "color": 'r'}, boxprops={"linewidth": 1.5},
             showmeans=False, meanline=False, showfliers=False)
     
    plt.ylabel('TCR[°C]', size = 18)
    plt.yticks(fontsize = 13)
    plt.xticks(fontsize = 13)
    ax1.set_xlim([0.35,4.87])
    ax1.set_ylim([0.9,2.9])
    ax1.axhline (weighted_TCR[2] , linestyle = '--', linewidth = 2, alpha = 0.4, color = 'darkorange')
    ax1.fill_between (np.array([0.25,5]), weighted_TCR[1], weighted_TCR[3], alpha=.2, linewidth=0)
    plt.legend(custom_lines, ['66%', 'Median'], loc = 'upper left', ncol = 2)
    plt.title ('Best estimates and likely ranges for TCR constrained', size = 16)
    plt.tight_layout()
 