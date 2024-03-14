# netEC
[DOI](https://zenodo.org/badge/769948700.svg)

emergent network constraint on climate sensitivity

netEC provides weighted estimates of ECS and TCR based on the evaluation of SST networks reconstructed from CMIP6 historical SST output over the time period 1975-2014

The protocol is the following:

1) Load nc file of CMIP6 model output of SST (or any other SST dataset you have) and pre-process SST data with Climate Data Operator (CDO)
2) Load referent SST files (nc file of HadISST and COBEv2). Pre-processed data are given in Data/ 
4) Load referent network (causal graph) inferred in reference HadISST dataset, which consists of:
   - referent regions of Sea Surface Temperature (SST) inferred with delta-MAPS algorithm (binary maps)
   - referent directed teleconnections inferred with PCMCI 
5) Calculate SST signals (cumulative SST signal anomalies) in referent SST regions
6) Calculate two distance metrics with respect to reference datasets HadISST and COBEv2:
  - Metric Weighted Wasserstein Distance (WWD) which is the weighted mean of the WD between simulated SST signal and referent SST signal in each region
  - Metric Distance Average Causal Effect (D_ACE) which is the distance between maps labeled with the ACE of the region, where ACE is the average of the total causal effect of each region 
6) Calculate performance weights (ClimWIP approach)
7) Calculate and plot weighted estimates of ECS and TCR


# Useful links 

COBEv2 https://psl.noaa.gov/data/gridded/data.cobe2.html
HadISST  https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html
CMIP6 output https://esgf-node.llnl.gov/projects/cmip6/
Region inference and strength estimate with delta-MAPS :  https://github.com/FabriFalasca/delta-MAPS
Causal discovery and causal effects estimate with Tigramite : https://github.com/jakobrunge/tigramite
