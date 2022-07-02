# Confident Sinkhorn Allocation for Pseudo-Labeling
https://arxiv.org/pdf/2206.05880.pdf


# Plot results

> Please specify the following parameters. These parameters will link to the correct files in your result folders

save_dir = 'results_output' # path to the folder store the results
out_file='' 
numTrials=20 # number of repeated trials
numIters=5 # number of used pseudo-iterations
dataset_name='madelon_no' # datasets
list_algorithms=['Pseudo_Labeling','FlexMatch','UPS','CSA'] # list of algorithms to be plotted

> the following parameters to be used to load the correct paths
confidence='ttest' # for CSA 
upper_threshold=0.8 # for Pseudo_Labeling,FlexMatch
low_threshold=0.2 # for UPS
num_XGB_models=10 # for CSA and UPS


```
python plot_results.py
```

# All dataset is collected from UCI and stored in all_data.pickle

> we can load all data with their datasetName as follows:
```
    with open('all_data.pickle', 'rb') as handle:
        [all_data, datasetName_list] = pickle.load(handle)
```

## Credits and References:

Vu Nguyen, Sachin Farfade, and Anton van den Hengel. "Confident Sinkhorn Allocation for Pseudo-Labeling." Workshop on Distribution-Free Uncertainty Quantification at ICML 2022
