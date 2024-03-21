# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:49:25 2024

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
from mpl_toolkits.basemap import Basemap
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
from matplotlib.colors import BoundaryNorm

def build_domain_map (d_maps):
    d,x,y = d_maps.shape
    domain_map_output = np.zeros((x,y))
    for d in range(len(d_maps)) [::-1]:     
        domain_map_output[d_maps[d] == 1] = d+1
    return domain_map_output


def plot_polynome_2d (y_ECS, y_TCR, weights_for_ECS, weights_for_TCR, sigmaD_4_ECS, sigmaD_4_TCR, ylabel) :
    model_4_ECS, RMSE_withECS, r2_score_withECS = fit_polynomial_2nd (y_ECS, weights_for_ECS)
    model_4_TCR, RMSE_withTCR, r2_score_withTCR = fit_polynomial_2nd (y_TCR, weights_for_TCR)
    
    xline_ECS = np.arange(1.7,6,0.01)
    xline_TCR = np.arange(1,3,0.01)
    
    y_pred_ECS = model_4_ECS (xline_ECS)
    ECS_bestestim = xline_ECS[np.argmax(y_pred_ECS)]
    y_pred_TCR = model_4_TCR (xline_TCR)
    TCR_bestestim = xline_TCR[np.argmax(y_pred_TCR)]
    
    plt.figure(figsize = (17,5.5))
    plt.subplot (1,2,1)
    ax = plt.gca()
    ax.axhline (1/N_models, linestyle = '--', color = 'grey', label = 'weights equal')
    ax.axvline (ECS_bestestim, linestyle = '--', color = 'green', label = 'Best estim = %s' %ECS_bestestim.round(2))
    plt.plot (y_ECS,weights_for_ECS, 'o', color = 'b', label = 'sigD = %s' %np.round(sigmaD_4_ECS,2))
    plt.plot (xline_ECS, y_pred_ECS , 'r')
    plt.title ('goodness-of-fit RMSE = %s and r2 score = %s'  %(RMSE_withECS, r2_score_withECS), size = 15)
    plt.legend(bbox_to_anchor = (0.85, -0.17), fontsize = 11, ncol = 3)
    plt.ylabel ('%s' %ylabel, size = 16)
    plt.xlabel ('ECS [°C]', size = 16)
    plt.xticks (fontsize = 13)
    plt.yticks (fontsize = 13)
    plt.grid(alpha = .5)
    plt.subplot (1,2,2)
    ax = plt.gca()
    ax.axhline (1/N_models, linestyle = '--', color = 'grey', label = 'weights equal')
    ax.axvline (TCR_bestestim, linestyle = '--', color = 'green', label = 'Best estim = %s' %TCR_bestestim.round(2))
    plt.plot (y_TCR, weights_for_TCR, 'o', color = 'b', label = 'sigD = %s' %np.round(sigmaD_4_TCR,2))
    plt.plot (xline_TCR, y_pred_TCR, 'r')
    plt.title ('goodness-of-fit RMSE = %s and r2 score = %s' %(RMSE_withTCR, r2_score_withTCR), size = 15)
    plt.legend(bbox_to_anchor = (0.9, -0.17), fontsize = 11, ncol = 3)
    plt.ylabel ('%s' %ylabel, size = 16)
    plt.xlabel ('TCR [°C]', size = 16)
    plt.xticks (fontsize = 13)
    plt.yticks (fontsize = 13)
    plt.grid(alpha = .5)
    plt.tight_layout()
    return None 
 

def compute_generalized_distance(d1, d2):
    # Generalized distance = sum of diagnostic 
    # Diagnostic are area weighted RMSE in Brunner et al., 2019
    diagnostic1 = d1/np.median(d1)
    diagnostic2 = d2/np.median(d2)
    
    std_diagnostic1 = np.std(diagnostic1)
    std_diagnostic2 = np.std(diagnostic2)

    # Inverse variant weighting
    # w1 = 1/(std_diagnostic1**2)
    # w2 = 1/(std_diagnostic2**2)
    
    # w_diagnostic1 = w1/(w1+w2)
    # w_diagnostic2 = w2/(w1+w2)
    
    w_diagnostic1 = .5
    w_diagnostic2 = .5
    
    D = w_diagnostic1*diagnostic1 + w_diagnostic2*diagnostic2
    return D, w_diagnostic1, w_diagnostic2

def fit_polynomial_2nd (target, weights):
    # quadratic fit 
    quadratic_model = np.poly1d(np.polyfit(target, weights, 2)) 
    
    # evaluation metrics
    pred_weights = quadratic_model (target)
    r2_score = np.round(np.corrcoef (weights, pred_weights)[0,1], 2)
    RMSE = np.round(np.sqrt(np.nanmean ((weights - pred_weights)**2)),2)
    return quadratic_model, RMSE, r2_score

def estimate_sigmaD_v2 (D, y):
    # Based on optimzed correlation between weights and the target ECS/TCR
    # Limitation : overconfidence
    median_D = np.median (D)
    low_bound = np.round(median_D*0.20,2) 
    up_bound = np.round(median_D*2,2) 
    # print ('20%% median(D) = %s' %low_bound)
    # print ('200%% median(D) = %s' %up_bound)
    
    sigD_optim = 0
    r2score_optim = 0
    for sigD in np.arange (low_bound, up_bound+0.01, 0.01) :
        weights_x =  compute_weights (D, sigD) 
        model_polynomial, RMSE, r2_score = fit_polynomial_2nd (y, weights_x)
        if np.abs(r2_score) > np.abs(r2score_optim) :
            sigD_optim = sigD
            r2score_optim = r2_score
    return sigD_optim, r2score_optim

def estimate_sigmaD (D, threshold_ECS, threshold_TCR):
    # Based on perfect model test in ClimWIP
    D = D/np.median(D)
    indsort = np.argsort (D)
    D_sort = D[indsort]
    ECS_sort = ECS[indsort]
    TCR_sort = TCR[indsort]
    
    liste_sigma, sigmaD_ECS = perfect_model_test(D_sort, ECS_sort, threshold_ECS)
    liste_sigma, sigmaD_TCR = perfect_model_test(D_sort, TCR_sort, threshold_TCR)   
    return sigmaD_ECS, sigmaD_TCR

    
def perfect_model_test (Distance, y_target, thresh_models) :
    median_D = np.median (Distance)
    low_bound = np.round(median_D*0.20,2) 
    up_bound = np.round(median_D*2,2) 
    # print ('20%% median(D) = %s' %low_bound)
    # print ('200%% median(D) = %s' %up_bound)
    list_sigma_D = []
 
    for sigD in np.arange (low_bound, up_bound+0.01, 0.01) :
        count_models = 0
        for i in range (len(y_target)) :
            y_truth = y_target[i]
            y_work = np.delete (np.copy(y_target), i)
            D_work = np.delete (np.copy(Distance), i)
            w_norm_work = compute_weights(D_work, sigD)
            y_10_estim = quantile(y_work, 0.10, w_norm_work)
            y_90_estim = quantile(y_work, 0.90, w_norm_work)
    
            if (y_truth < y_90_estim) & (y_truth > y_10_estim) :
                count_models +=1
                
        if count_models >= thresh_models :
            list_sigma_D.append (sigD)
            
    list_sigma_D = np.array(list_sigma_D)
    try :
        sigD_optim = np.min(list_sigma_D)
    except :
        sigD_optim = np.nan
    return list_sigma_D, sigD_optim

def compute_weights (D0, sigmaD) :
    w_perf0 = np.exp(-D0**2/sigmaD**2)
    w_perf0_final = w_perf0/np.nansum(w_perf0)
    return w_perf0_final

def compute_list_perf_weights (D, threshold) :
    sigD_ECS_0, sigD_TCR_0 = estimate_sigmaD (D, threshold, threshold)
    sigD_ECS  = check (D, sigD_ECS_0)
    sigD_TCR = check (D, sigD_TCR_0)
    weights_for_ECS = compute_weights (D, sigD_ECS) 
    weights_for_TCR = compute_weights (D, sigD_TCR)  
    print ('sigma D for ECS = ', np.round(sigD_ECS,2))
    print ('sigma D for TCR = ', np.round(sigD_TCR,2))
    return weights_for_ECS, weights_for_TCR

def impose_list_perf_weights (D, sigD_ECS, sigD_TCR) :
    weights_for_ECS = compute_weights (D, sigD_ECS) 
    weights_for_TCR = compute_weights (D, sigD_TCR)  
    # print ('sigma D for ECS = ', np.round(sigD_ECS,2))
    # print ('sigma D for TCR = ', np.round(sigD_TCR,2))
    return weights_for_ECS, weights_for_TCR

def quantile (y, quantiles, weights):
    # Increasing values
    ind_sort = np.argsort(y)
    y = y[ind_sort]
    weights = weights[ind_sort]

    weighted_quantiles = np.cumsum(weights) - 0.5*weights
    weighted_quantiles /= np.sum(weights)

    #Function np.interp (x to interp, x, y)
    results = np.interp(quantiles, weighted_quantiles, y) 
    return results 

def get_stats (data, w):
    mean_es = np.average(data, weights= w).round(2)
    median_es = quantile(data, 1/2,  w).round(2)
    range66_low = quantile(data, 1/6, w).round(2)
    range66_high = quantile(data, 5/6, w).round(2)
    range90_low = quantile(data, 1/20, w).round(2)
    range90_high = quantile(data, 19/20, w).round(2)
    
    return mean_es, median_es, range66_low, range66_high, range90_low, range90_high 

def get_distributions (data, w, space):
    datapoints_distrib = []
    for perc in np.arange(0.05,1., space):
        estim = quantile(data, perc,  w).round(2)
        datapoints_distrib.append (estim)
    datapoints_distrib = np.array(datapoints_distrib)
    return datapoints_distrib


def get_weight_density_distributions (data_target, w):
    # get the bins from ECS or TCR distribution (and associated density)
    bins_target = np.histogram (data_target, 7)

    # get the density of weights
    density_weights = []
    for i in range (len(bins_target)-1):
        low = bins_target [i]
        up = bins_target [i+1]
        indexes = np.where ((data_target >= low) & (data_target < up))
        density_weights.append (w[indexes].sum())
    density_weights = np.array(density_weights)
    return density_weights, bins_target


def check (dist_metric, sigD) :
    weights = compute_weights (dist_metric, sigD) 
    w0 = 1/27
    while len(np.where (weights > w0)[0]) < 10 :
        sigD += 0.01
        weights = compute_weights (dist_metric, sigD) 
    return sigD

def get_boxes_v2 (boxes, list_weights, list_names, target_y):

    for i in range (len(list_weights)):
        mean, med, perc17, perc83, perc5, perc95 = get_stats(target_y, list_weights[i])
        boxes['mean'].append (mean)
        boxes['med'].append (med)
        boxes['q1'].append (perc17)
        boxes['q3'].append (perc83)
        boxes['whislo'].append (perc5)
        boxes['whishi'].append (perc95)
        boxes['label'].append (list_names[i])

    return boxes


def get_boxes (list_weights, list_names, target_y):
    list_perc5, list_perc95 = [], []
    list_perc17, list_perc83 = [], []
    list_mean, list_med = [], []
    for i in range (len(list_weights)):
        mean, med, perc17, perc83, perc5, perc95 = get_stats(target_y, list_weights[i])
        list_mean.append (mean)
        list_med.append (med)
        list_perc17.append (perc17)
        list_perc83.append (perc83)
        list_perc5.append (perc5)
        list_perc95.append (perc95)
        
    boxes =  { 'label' : list_names,
                 'whislo': list_perc5,  
                 'q1'    : list_perc17,     
                 'med'   : list_med,   
                 'mean' : list_mean,   
                 'q3'    : list_perc83,    
                 'whishi': list_perc95,    
                 'fliers': np.ones(len(list_names))*np.nan
                 }
        
    return boxes

def compute_intra_model_mean_and_std (mat_measures):
    intra_model_mean = []
    intra_model_std = []
    runs_measures = []
    for i in range (N_models):
        measures = mat_measures [i]
        n_runs, n_regions = measures.shape
        weighted_mean_measures = np.nansum(measures*weight_size_regions, axis = 1) # Weighted average 
        # print (weighted_mean_measures.shape)
        
        # Stock the information
        runs_measures.append (weighted_mean_measures)
        intra_model_mean.append (np.nanmean (weighted_mean_measures))
        intra_model_std.append (np.nanstd(weighted_mean_measures))
        
    intra_model_mean = np.array(intra_model_mean)
    intra_model_std = np.array(intra_model_std )
    runs_measures = np.array(runs_measures)
    return intra_model_mean, intra_model_std, runs_measures 


def compute_intra_model_mean_and_std_v2 (mat_measures, code):
    intra_model_mean = []
    intra_model_std = []
    runs_measures = []
    for i in range (N_models):
        measures = mat_measures [i]
        n_runs, n_regions = measures.shape
        if code == 'sum':
            weighted_mean_measures = np.nansum(measures, axis = 1) # Weighted average 
        if code == 'mean':
            weighted_mean_measures = np.nanmean(measures, axis = 1) # Weighted average 
        # print (weighted_mean_measures.shape)
        
        # Stock the information
        runs_measures.append (weighted_mean_measures)
        intra_model_mean.append (np.nanmean (weighted_mean_measures))
        intra_model_std.append (np.nanstd(weighted_mean_measures))
        
    intra_model_mean = np.array(intra_model_mean)
    intra_model_std = np.array(intra_model_std )
    runs_measures = np.array(runs_measures)
    return intra_model_mean, intra_model_std, runs_measures 

def plot_num_boxplot_with_boxes (list_perf_weights, list_names_weights, boxes, y, name_y):
    # Settings
    f = 2e3
    np.random.seed(14)
  
    # On the x_axis  
    x1 = np.random.normal(1, 0.09, len(y))  #gaussian centered in 1
    the_x = []
    for i in range (len(list_perf_weights)):
        the_x.append (x1 + i*0.7)

    pos = []
    for i in range (len(list_perf_weights)):
        pos.append  (1 + i*0.7)

    # Legend
    custom_lines = [Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='orange', lw=2),
                    Line2D([0], [0], color='red', linestyle = '--',  lw=2)
                    ]

    # Get boxes
    df = pd.DataFrame(boxes, columns = ['label', 'whislo', 'q1', 'med', 'mean', 'q3',  'whishi', 'fliers'])
    
  
    fig, ax = plt.subplots(figsize = (7,6), dpi = 150)   
    ax.bxp(df.to_dict('records'), 
           positions = pos,
           widths = 0.6,
           medianprops={"linewidth": 2},  
           meanprops={"linewidth": 2, "color": 'r'}, 
           boxprops={"linewidth": 1.5},
           showmeans=True, meanline=True, showfliers=False)
    plt.legend(custom_lines, ['66%', 'Median', 'Mean'], loc = 'upper left', ncol = 3)
    for i in range (len(list_perf_weights)):
        plt.scatter(the_x[i], y, alpha=0.5, s = list_perf_weights[i]*f, color= 'green')
       
    plt.ylabel('%s[°C]' %name_y, size = 18)
    plt.yticks(fontsize = 13)
    plt.xticks(fontsize = 15)
    if name_y == 'TCR' :
        plt.ylim([1.25,3.15])
    if name_y == 'ECS':
        plt.ylim([1.7,6.2])
    plt.title ('Best estimates and likely ranges for %s' %name_y, size = 19)
    plt.tight_layout()

    return df


def plot_num_boxplot (list_perf_weights, list_names_weights, y, name_y):
    # Settings
    f = 2e3
    np.random.seed(14)
  
    # On the x_axis  
    x1 = np.random.normal(1, 0.09, len(y))  #gaussian centered in 1
    the_x = []
    for i in range (len(list_perf_weights)):
        the_x.append (x1 + i*0.7)

    pos = []
    for i in range (len(list_perf_weights)):
        pos.append  (1 + i*0.7)

    # Legend
    custom_lines = [Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='orange', lw=2),
                    Line2D([0], [0], color='red', linestyle = '--',  lw=2)
                    ]
    
    # Get boxes
    boxes = get_boxes (list_perf_weights, list_names_weights,  y)
    df = pd.DataFrame(boxes, columns = ['label', 'whislo', 'q1', 'med', 'mean', 'q3',  'whishi', 'fliers'])
    
  
    fig, ax = plt.subplots(figsize = (7,6), dpi = 150)   
    ax.bxp(df.to_dict('records'), 
           positions = pos,
           widths = 0.6,
           medianprops={"linewidth": 2},  
           meanprops={"linewidth": 2, "color": 'r'}, 
           boxprops={"linewidth": 1.5},
           showmeans=True, meanline=True, showfliers=False)
    plt.legend(custom_lines, ['66%', 'Median', 'Mean'], loc = 'upper left', ncol = 3)
    for i in range (len(list_perf_weights)):
        plt.scatter(the_x[i], y, alpha=0.5, s = list_perf_weights[i]*f, color= 'green')
       
    plt.ylabel('%s[°C]' %name_y, size = 18)
    plt.yticks(fontsize = 13)
    plt.xticks(fontsize = 15)
    if name_y == 'TCR' :
        plt.ylim([1.25,3.15])
    if name_y == 'ECS':
        plt.ylim([1.7,6.2])
    plt.title ('Best estimates and likely ranges for %s' %name_y, size = 19)
    plt.tight_layout()

    return df


def plot_one_boxplot (list_perf_weights, list_names_weights, y, name_y):
    # We plot side to side the unweighted boxplot and weighted boxplot
    f = 2e3
    np.random.seed(14)
    
    x1 = np.random.normal(1, 0.07, len(y))  #gaussian centered in 1
    x2 = x1 + 0.7  # gaussian centered in 1.7

    weights_equal = list_perf_weights[0]
    perf_weights = list_perf_weights[1]
    indsort_weights = np.flip(np.argsort (perf_weights)) # decreasing order
    x2_sort = x2[indsort_weights] 
    y2_sort = y[indsort_weights] 
    
    # Get boxes
    boxes = get_boxes (list_perf_weights, list_names_weights,  y)
    df = pd.DataFrame(boxes, columns = ['label', 'whislo', 'q1', 'med', 'mean', 'q3',  'whishi', 'fliers'])
    
    # Legend
    names2_sort = names_CMIP6[indsort_weights] 
    mapping = (np.arange(1,len(names2_sort)+1), names2_sort)
    
    legend_to_create = ['\n']
    for i in range (len(names2_sort)):
        legend_to_create.append ('%s : %s \n' %(i+1, names2_sort[i]))
    
    # Legend
    custom_lines = [Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='orange', lw=2),
                    Line2D([0], [0], color='red', linestyle = '--',  lw=2)
                    ]
  
    fig, ax = plt.subplots(figsize = (7,6.5), dpi = 150)   
    ax.bxp(df.to_dict('records'), 
           positions = np.array([1,1.7]),
           widths = 0.6,
           medianprops={"linewidth": 2},  
           meanprops={"linewidth": 2, "color": 'r'}, 
           boxprops={"linewidth": 1.5},
           showmeans=True, meanline=True, showfliers=False)
    plt.legend(custom_lines, ['66%', 'Median', 'Mean'], loc = 'upper left', ncol = 3)
    plt.scatter(x1, y, alpha=0.5, s = weights_equal*f, color= 'green')
    plt.scatter(x2, y, alpha=0.5, s = perf_weights*f, color= 'green')
    plt.ylabel('%s[°C]' %name_y, size = 18)
    plt.yticks(fontsize = 13)
    plt.xticks(fontsize = 15)
    if name_y == 'TCR' :
        plt.ylim([1.25,3.15])
        plt.text(2.1, 1.3, ''.join(legend_to_create), backgroundcolor = 'antiquewhite', fontsize =9)
    if name_y == 'ECS':
        plt.ylim([1.7,6.15])
        plt.text(2.1, 2.1, ''.join(legend_to_create), backgroundcolor = 'antiquewhite', fontsize =9)
    plt.title ('Best estimates and likely ranges for %s' %name_y, size = 19)
    plt.tight_layout()

    return df
    
def plot_perf_weights (weights_climwip_for_ECS, weights_climwip_for_TCR, sigD_ECS, sigD_TCR):
    plt.figure(figsize = (5,4), dpi = 150)
    ax = plt.gca()
    ax.axhline (1/N_models, linestyle = '--', color = 'green', label = 'equal weights')
    plt.plot (np.flip(np.sort(weights_climwip_for_ECS)), color = 'b', 
              marker = '.', label = 'ECS (sigD = %s)' %np.round(sigD_ECS,2))
    plt.plot (np.flip(np.sort(weights_climwip_for_TCR)), color = 'r',
              marker = '.', label = 'TCR (sigD = %s)' %np.round(sigD_TCR,2))
    plt.grid()
    plt.legend()
    plt.xticks (fontsize = 12)
    plt.yticks (fontsize = 12)
    plt.ylabel ('Performance weights', size=14)
    plt.xlabel ('CMIP6 models', size=14)
    plt.tight_layout()
    return None 

def plot_perf_weights2 (weights_climwip_for_ECS, weights_climwip_for_TCR):
    plt.figure(figsize = (5,4), dpi = 150)
    ax = plt.gca()
    ax.axhline (1/N_models, linestyle = '--', color = 'green', label = 'equal weights')
    plt.plot (np.flip(np.sort(weights_climwip_for_ECS)), color = 'b', marker = '.', label = 'ECS')
    plt.plot (np.flip(np.sort(weights_climwip_for_TCR)), color = 'r', marker = '.', label = 'TCR')
    plt.grid()
    plt.legend()
    plt.xticks (fontsize = 12)
    plt.yticks (fontsize = 12)
    plt.ylabel ('Performance weights', size=14)
    plt.xlabel ('CMIP6 models', size=14)
    plt.tight_layout()
    return None 


if __name__ == '__main__':   
    %matplotlib qt
    
    lat = np.arange(-59.,60.,2)
    lon = np.arange (0.,360.,2)
    LON, LAT = np.meshgrid(lon, lat)
    lat_rad = np.radians(lat)
    lat_weights = np.cos(lat_rad).reshape(len(lat_rad),1)
    index_others = [2,3,4,6,7,8,9,10,11,12,13,14,15,16]
    
    rep_fig = 'Figures/'
    
    # Names CMIP6
    names_CMIP6 = np.array(['NorESM2-MM (3)', 'NorESM2-LM (3)', 'MPI-HR (10)', 'MPI-LR (30)', 'EC-Earth3 (14)', 
                            'EC-Earth-Veg-LR (3)', 'MIROC6 (8)', 'GFDL-ESM4 (2)', 'GISS-E2-1-G (12)',
                            'HadGEM3 (5)', 'CNRM-CM6-1 (26)', 'CanESM5 (24)', 'INM-CM5 (10)', 'UKESM1-0-LL (17)', 
                            'IPSL (32)', 'NESM3 (5)', 'MRI-ESM2-0 (10)',  'BCC-ESM1 (3)', 'BCC-CSM2-MR (3)',
                            'EC-Earth_AerChem (2)', 'SAMO UNICON (1)', 'ACCESS-ESM1-5 (15)', 'FGOALS-f3-L (3)',
                             'ESM-1-0 (5)', 'CAMS-CSM1-0 (2)', 'CESM2 (11)', 'CESM2 WACCM (3)'])
    
    
    # Load ECS/TCR values of CMIP6 models
    sensitivity_table  = pd.read_csv('Data_climate_sensitivity/ECS_and_TCR.txt', sep=" ", index_col =["name"])
    ECS = np.array(sensitivity_table['ECS'])
    TCR = np.array(sensitivity_table['TCR'])
    
    perc25_ECS, median_ECS, perc75_ECS = np.percentile (ECS, 0.25), np.median (ECS), np.percentile (ECS, 0.75)
    perc25_TCR, median_TCR, perc75_TCR = np.percentile (TCR, 0.25), np.median (TCR), np.percentile (TCR, 0.75)
    print ('stats ECS = ', perc25_ECS, median_ECS, perc75_ECS)
    print ('stats TCR = ', perc25_TCR, median_TCR, perc75_TCR)
    
    N_models = len(names_CMIP6)
    
    # 30 SST regions and 480 months
    d_maps1 = np.load ('Data/regions_reference_HadISST_sst_1975-2014.npy')
    N_regions = len(d_maps1)
    print ('Number SST regions = %s'%N_regions)
    size_regions = np.array([np.nansum(d_maps1[i,]) for i in range (N_regions)])
    d_ids1 = np.arange(1, N_regions +1)
    domain_map1 = build_domain_map (d_maps1)
    


    # Load metric values
    # Load Distance ACE
   
    rep1 = 'D_ACE_9_regions/'
    intra_mean_D_ACE =  np.load (rep1 + 'intra_model_mean_9regions_D_ACE_refHadISST.npy')
    intra_mean_D_ACE_2 = np.load (rep1 + 'intra_model_mean_9regions_D_ACE_refCOBEv2.npy')
    intra_mean_D_ACE_mix = np.load (rep1 + 'intra_model_mean_9regions_D_ACE_refmix.npy')

    # Obs uncertainty 
    print ('\n Correlation HadISST, COBEv2 :')
    print (np.corrcoef (intra_mean_D_ACE, intra_mean_D_ACE_2)[0,1].round(2))
    
    #Inter model
    inter_mean_D_ACE = np.nanmean (intra_mean_D_ACE)
    inter_std_D_ACE = np.nanstd(intra_mean_D_ACE)
    
    inter_mean_D_ACE_2 = np.nanmean (intra_mean_D_ACE_2)
    inter_std_D_ACE_2 = np.nanstd(intra_mean_D_ACE_2)
    
    inter_mean_D_ACE_mix = np.nanmean (intra_mean_D_ACE_mix)
    inter_std_D_ACE_mix = np.nanstd(intra_mean_D_ACE_mix)
    
    
    # Load WWD 
    rep2 = 'WWD_30_regions/'
    # Contributions of each SST region matrix shape n_models*n_regions 
    mat_WWD_HadISST = np.load (rep2 + 'mat_array_WWD_refHadISST.npy', allow_pickle =True)
    mat_WWD_COBEv2 = np.load (rep2 + 'mat_array_WWD_refCOBEv2.npy', allow_pickle =True)
    mat_WWD_mix = np.load (rep2 + 'mat_array_WWD_refMix.npy', allow_pickle =True)
    
    #Intra model
    intra_mean_WWD = np.load (rep2 + 'intra_mean_WWD_refHadISST.npy')
    intra_mean_WWD2 = np.load (rep2 + 'intra_mean_WWD_refCOBEv2.npy')
    intra_mean_WWD_mix = np.load (rep2 + 'intra_mean_WWD_refMix.npy')
    
    # Obs uncertainty 
    print ('\n Correlation HadISST, COBEv2 :')
    print (np.corrcoef (intra_mean_WWD, intra_mean_WWD2)[0,1].round(2))
    
    #Inter model
    inter_mean_WWD = np.nanmean (intra_mean_WWD_mix)
    inter_std_WWD = np.nanstd(intra_mean_WWD_mix)
    
    inter_mean_WWD2 = np.nanmean (intra_mean_WWD2)
    inter_std_WWD2 = np.nanstd(intra_mean_WWD2)
    
    inter_mean_WWD_mix = np.nanmean (intra_mean_WWD_mix)
    inter_std_WWD_mix = np.nanstd(intra_mean_WWD_mix)
    

    # Test Normality distribution with Shapiro-Wilk Test
    from scipy.stats import shapiro

    statistic, p_value = shapiro(intra_mean_WWD_mix)
    print(f"p-value: {p_value}")
    if p_value > 0.05:
        print("Fail to reject H0")
    else:
        print("Ho rejected : distrbution is not normal")

    # Performance weights (from ClimWIP approach)
    x1 = np.arange(1,N_models+1)*5 -1
    x2 = x1 +1
    
    # Two diagnostics to normalize 
    diag1 = intra_mean_WWD_mix/np.median(intra_mean_WWD_mix)
    diag2 = intra_mean_D_ACE_mix/np.median (intra_mean_D_ACE_mix)

    print ('\n Mean normalized diags = ')
    print (np.mean (diag1).round(2), ' and std = ', np.std(diag1).round(2))
    print (np.mean (diag2).round(2), ' and std = ', np.std(diag2).round(2))

    # Calibration approach or Perfect model test to determine shape parameter
    # Threshold = 19 models means 19/27 ~ 70% models satisfy the criterion 
    threshold = 19 
    sigD_ECS_1, sigD_TCR_1 =  estimate_sigmaD (diag1,threshold,threshold)
    print ('\n')
    print ('sigma D for ECS, WWD = ', np.round(sigD_ECS_1,2))
    print ('sigma D for TCR, WWD = ', np.round(sigD_TCR_1,2))
    print ('\n')
    
    sigD_ECS_2, sigD_TCR_2 = estimate_sigmaD (diag2, 19,19)
    print ('\n')
    print ('sigma D for ECS, D_ACE = ', np.round(sigD_ECS_2,2))
    print ('sigma D for TCR, D_ACE = ', np.round(sigD_TCR_2,2))
    print ('\n')
    
    ## If we want to impose 
    sigD_ECS_1, sigD_TCR_1 = 0.32, 0.34
    sigD_ECS_2, sigD_TCR_2 = 0.31, 1.07
 

    ## We can calculate the performance weights 
    weights_for_ECS1, weights_for_TCR1 = impose_list_perf_weights (diag1, sigD_ECS_1, sigD_TCR_1)
    # plot_perf_weights(weights_for_ECS1, weights_for_TCR1, sigD_ECS_1, sigD_TCR_1)
    
    weights_for_ECS2, weights_for_TCR2 = impose_list_perf_weights(diag2, sigD_ECS_2, sigD_TCR_2)
    # plot_perf_weights(weights_for_ECS2, weights_for_TCR2, sigD_ECS_2, sigD_TCR_2)

    ##  We can combine weights derived from each of the distance metric 
    weights_combined_for_ECS = (weights_for_ECS1 + weights_for_ECS2)/2
    weights_combined_for_TCR = (weights_for_TCR1 + weights_for_TCR2)/2
    # plot_perf_weights2(weights_combined_for_ECS, weights_combined_for_TCR)
    # plt.savefig ('Figures/Performance_weights_threshold19_models', dpi = 300)

    ## Weights are not correlated with climate sensitivity
    print (np.corrcoef (weights_combined_for_ECS, ECS)[0,1].round(2))
    print(np.corrcoef (weights_combined_for_TCR, TCR)[0,1].round(2))
    
    ## Variability of weights
    print (np.std(weights_for_TCR1).round(3))
    print (np.std(weights_for_TCR2).round(3))
    
    ##  PLOT 2D weights
    # weights_equals = np.ones(len(names_CMIP6))/len(names_CMIP6)
    
    # x = np.arange (1,5)
    # y = np.arange (1,28)
    
    # indsort = np.argsort (ECS)
    
    # X, Y = np.meshgrid (x, y)
    # norm = BoundaryNorm(np.arange(0,0.11, 0.01), plt.cm.Reds.N, extend='max')
    
    # mat_weights = np.vstack([weights_equals,
    #                          weights_for_ECS1[indsort],
    #                          weights_for_ECS2[indsort],
    #                          weights_combined_for_ECS[indsort]]).T
    
    # plt.figure(figsize = (6,9), dpi = 150)
    # plt.pcolormesh(X, Y, mat_weights, cmap = plt.cm.Reds, norm = norm)
    # plt.xticks (x, ['$w_{equals}$', 'w($WWD$)', 'w($D_{ACE}$)', '$w$'], fontsize = 10)
    # plt.yticks (y, names_CMIP6[indsort], fontsize = 8)
    # plt.colorbar (shrink = 0.6)
    # plt.tight_layout()
    # plt.savefig ('Figures/weights_ECS_imshow', dpi  =300)
    
    ##  PLOT 2D metric space
    # markers = ['o', 'p', 's', '>', '*', 'v', '<', '^', 'h', '8', 'D', 
    #             'o', '>', 'p', 's', '*', 'v', '<', '^', 'h', '8', 'D',
    #             'o', '>', 'p', 's', '*', 'v', '<', '^', 'h', '8', 'D']
    
    # colors = ['red', 'blue', 'lawngreen', 'magenta', 'darkgreen', 'dodgerblue', 'gold', 'sienna', 'black',
    #           'cyan', 'chocolate', 'orange', 'crimson', 'grey', 'purple', 'red', 'lawngreen','blue', 'magenta',
    #           'darkgreen', 'dodgerblue', 'gold', 'sienna', 'black','chocolate', 'cyan', 'orange', 
    #           'crimson', 'grey', 'purple']
    
    # plt.rcParams.update({'font.size': 9.5})


    # std_WWD = np.round (np.std (diag1),2)
    # std_D_ACE = np.round (np.std (diag2), 2)
    
    # text_to_display = '$\u03C3_{WWD}$ = %s \n $\u03C3_{ACE}$ = %s' %(std_WWD, std_D_ACE)
        
    # plt.figure(figsize = (10,6))
    # for i in range (len(diag1)):
    #     plt.plot(diag1[i], diag2[i], markers[i], markersize = 11, color = colors[i], label = names_CMIP6[i])
    # plt.xticks (fontsize = 14)
    # plt.yticks (fontsize = 14)
    # plt.xlabel ('$WWD$', size = 17)
    # plt.ylabel ('$D_{ACE}$', size = 17)
    # plt.legend(bbox_to_anchor = (1.01, 1.06), ncol = 1,  fancybox=True, shadow=True)
    # plt.title ('Distance of Large Ensemble SST to observations', style = 'italic', size = 19)
    # plt.tight_layout()
    # plt.text (2.85, 1.85, text_to_display, size = 12, bbox = dict (facecolor = 'wheat', alpha = 0.5))
    # plt.savefig ('Figures/2D_metric_space_WWD_D_ACE', dpi = 300)
    
    ####### Plot PDF = weight density distributions

    # density_weights_netEC_TCR, bins_TCR = get_weight_density_distributions (TCR, weights_combined_for_TCR)
    # density_weights_netEC_ECS, bins_ECS=  get_weight_density_distributions (ECS, weights_combined_for_ECS)
    
    # middle_bins_TCR = (bins_TCR[:-1] + bins_TCR[1:])/2
    # middle_bins_ECS = (bins_ECS[:-1] + bins_ECS[1:])/2
    
    # list_perc = ['perc 0', 'perc 16.6', 'perc 33.3', 'perc 50', 'perc 66.6', 'perc 83.3', 'perc 100']
    
    # plt.figure(figsize = (10,4), dpi= 150)
    # plt.subplot(1,2,1)
    # plt.bar (middle_bins_ECS, density_weights_netEC_ECS, width = np.diff (bins_ECS), color = 'skyblue',edgecolor='black')
    # plt.xlabel('ECS [°C]', size = 12)
    # plt.ylabel('Density', size = 12)
    # plt.xticks (bins_ECS,list_perc, rotation = 60, fontsize = 8)
    # plt.title('Weight distribution function', size = 12)
    # plt.subplot(1,2,2)
    # plt.bar (middle_bins_TCR, density_weights_netEC_TCR, width = np.diff (bins_TCR),  color = 'skyblue',edgecolor='black')
    # plt.xlabel('TCR [°C]', size = 12)
    # plt.ylabel('Density', size = 12)
    # plt.xticks (bins_TCR, list_perc, rotation = 60, fontsize = 8)
    # plt.title('Weight distribution function', size = 12)
    # plt.tight_layout()
    # plt.savefig ('Figures/Weight_distribution_function', dpi = 300)
    
    
    ####### Plot boxplot    
   
    # Compute stats  
    weights_equal =  np.ones((N_models))/N_models
    mean_ref, med_ref, perc17_ref, perc83_ref, perc5_ref, perc95_ref = get_stats(ECS, weights_equal)
    length66_ref = perc83_ref - perc17_ref
    length90_ref = perc95_ref - perc5_ref

    mean, med, perc17, perc83, perc5, perc95 = get_stats(ECS, weights_combined_for_ECS)
    length66 = perc83 - perc17
    length90 = perc95 - perc5
    
       
    print ('TARGET ECS')
    print ('Weights equal :')
    print ('med ref = %s, mean_ref = %s, and [%s, %s]' %(med_ref, mean_ref, perc17_ref, perc83_ref))
    print ('length range 66 ref = %s' %length66_ref.round(2))
    print ('length range 90 ref = %s' %length90_ref.round(2))

    print ('\n')
    print ('med = %s, mean  = %s, and [%s, %s]' %(med, mean, perc17, perc83))
    print ('length range 66 = %s' %length66.round(2))
    print ('length range 90 = %s' %length90.round(2))

    ### TCR 
    mean_ref2, med_ref2, perc17_ref2, perc83_ref2, perc5_ref2, perc95_ref2 = get_stats(TCR, weights_equal)
    length66_ref2 = perc83_ref2 - perc17_ref2
    length90_ref2 = perc95_ref2 - perc5_ref2

    
    mean2, med2, perc17_2, perc83_2, perc5_2, perc95_2 = get_stats(TCR, weights_combined_for_TCR)
    length66_2 = perc83_2 - perc17_2
    length90_2 = perc95_2 - perc5_2
    
    print ('\n')
    print ('TARGET TCR')
    print ('Weights equal :')
    print ('med ref = %s, mean_ref = %s, and [%s, %s]' %(med_ref2, mean_ref2, perc17_ref2, perc83_ref2))
    print ('length range 66 ref = %s' %length66_ref2.round(2))
    print ('length range 90 ref = %s' %length90_ref2.round(2))

    print ('\n')
    print ('med = %s, mean  = %s, and [%s, %s]' %(med2, mean2, perc17_2, perc83_2))
    print ('length range 66 = %s' %length66_2.round(2))
    print ('length range 90 = %s' %length90_2.round(2))
    
    list_performance_weights_ECS = [ weights_equal, weights_combined_for_ECS]
    list_names_w1 = [  'unweighted', 'netEC']
    df1 = plot_one_boxplot(list_performance_weights_ECS, list_names_w1, ECS, 'ECS')
    # plt.savefig ('Figures/Estimate_ECS_boxplot')
    
    list_performance_weights_TCR = [ weights_equal, weights_combined_for_TCR]
    list_names_w2 = [  'unweighted', 'netEC']
    df1 = plot_one_boxplot (list_performance_weights_TCR, list_names_w1, TCR, 'TCR')
    # plt.savefig ('Figures/Estimate_TCR_boxplot')    
   


   
    