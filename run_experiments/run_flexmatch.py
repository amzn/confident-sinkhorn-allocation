import sys
sys.path.append('..')


import numpy as np
import os
import argparse
import logging
import pickle
from tqdm import tqdm
from confident_sinkhorn_allocation.algorithm.pseudo_labeling import Pseudo_Labeling
from confident_sinkhorn_allocation.algorithm.flexmatch import FlexMatch
#from confident_sinkhorn_allocation.algorithm.ups import UPS
#from confident_sinkhorn_allocation.algorithm.csa import CSA


from confident_sinkhorn_allocation.utilities.utils import get_train_test_unlabeled,append_acc_early_termination
from confident_sinkhorn_allocation.utilities.utils import get_train_test_unlabeled_for_multilabel
import warnings
warnings.filterwarnings('ignore')


def run_experiments(args, save_dir):

    out_file = args.output_filename
    numTrials=args.numTrials  
    numIters=args.numIters
    upper_threshold=args.upper_threshold
    verbose=args.verbose
    dataset_name=args.dataset_name
    IsMultiLabel=False # by default

    # in our list of datasets: ['yeast','emotions'] are multi-label classification dataset
    # the rest datasets are multiclassification
    if dataset_name in ['yeast','emotions']: # multi-label
        IsMultiLabel=True

    accuracy = []

    for tt in tqdm(range(numTrials)):
        
        np.random.seed(tt)
       
        # load the data        
        if IsMultiLabel==False: # multiclassification
            x_train,y_train, x_test, y_test, x_unlabeled=get_train_test_unlabeled(dataset_name,path_to_data='all_data.pickle',random_state=tt)
        else: # multi-label classification
            x_train,y_train, x_test, y_test, x_unlabeled=get_train_test_unlabeled_for_multilabel(dataset_name,path_to_data='all_data_multilabel.pickle',random_state=tt)
        
        pseudo_labeller = FlexMatch(x_unlabeled,x_test,y_test, 
                num_iters=numIters,
                upper_threshold=upper_threshold,
                verbose = False,
                IsMultiLabel=IsMultiLabel
            )
        pseudo_labeller.fit(x_train, y_train)
        
        accuracy.append(  append_acc_early_termination(pseudo_labeller.test_acc,numIters) )


    # print and pickle results
    filename = os.path.join(save_dir, '{}_{}_{}_numIters_{}_numTrials_{}_threshold_{}.pkl'.format(out_file, pseudo_labeller.algorithm_name,dataset_name,\
        numIters,numTrials,upper_threshold))
    print('\n* Trial summary: avgerage of accuracy per Pseudo iterations')
    print( np.mean( np.asarray(accuracy),axis=0))
    print('\n* Saving to file {}'.format(filename))
    with open(filename, 'wb') as f:
        pickle.dump([accuracy], f)
        f.close()

def main(args):

    # make save directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = save_dir + '/'  + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run_experiments(args, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for FlexMatch experiments')
    parser.add_argument('--numIters', type=int, default=5, help='number of Pseudo Iterations')
    parser.add_argument('--numTrials', type=int, default=20, help ='number of Trials (Repeated Experiments)' )
    parser.add_argument('--upper_threshold', type=float, default=0.8, help ='threshold in pseudo-labeling' )
    parser.add_argument('--dataset_name', type=str, default='synthetic_control_6c', help='segment_2310_20 | wdbc_569_31 | analcatdata_authorship | synthetic_control_6c | \
        German-credit |  madelon_no | agaricus-lepiota | breast_cancer | digits | yeast | emotions')

    parser.add_argument('--verbose', type=str, default='No', help='verbose Yes or No')
    parser.add_argument('--output_filename', type=str, default='', help='name of output files')
    parser.add_argument('--save_dir', type=str, default='results_output', help='name of save directory')

    args = parser.parse_args()
    main(args)