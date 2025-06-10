import random
import os 
import numpy as np

import torch
import time

from data.cxr import CXR
from data.isic import ISIC
from data.ph2 import PH2
from data.wbc import WBC

from bdcsam import get_eval # from bdcsam2 import get_eval

from segment_anything import sam_model_registry
from utils.utils import process_config

import argparse

def main(args):

    #  ========== add the seed to make sure the results are reproducible ==========

    np.random.seed(args.seed)     # set random seed for numpy
    random.seed(args.seed)        # set random seed for python for image transformations
    torch.manual_seed(args.seed)  # set random seed for CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # set random seed for  GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution
    
    config = process_config(os.path.join(os.path.dirname(__file__), args.config))

    #  ========== data preparation ==========
    
    if args.data == "cxr":
        test_data = CXR(config, 'whole')

    elif args.data == "isic":
        test_data = ISIC(config, 'train')

    elif args.data == 'ph2':
        test_data = PH2(config, 'whole')

    elif args.data == 'wbc':
        test_data = WBC(config, 'whole')

        
    final_avg_dice  = get_eval(test_data, config)
    print(f"Final average dice score: {final_avg_dice}")

  
if __name__ == '__main__':

    # ========== parameters setting ==========
    parser = argparse.ArgumentParser(description='SAM')
    # define arguments
    parser.add_argument('--config', type=str, required=True, help = 'path to config file. This file has all the parameters specificed in ./config folder')
    parser.add_argument('--data', type = str, required = True, help = "specify the name of dataset")
    parser.add_argument('--seed', type = int, required = True, help = "random seed")
    

    args = parser.parse_args()
    now = time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time()))

    print('----------------------------------------------------------------------')
    print('Time: ' + now)
    print('----------------------------------------------------------------------')
    print('                    Now start ...')
    print('----------------------------------------------------------------------')

    main(args)


    print('----------------------------------------------------------------------')
    print('                      All Done!')
    print('----------------------------------------------------------------------')
    print('Start time: ' + now)
    print('Now time: ' + time.strftime('%Y-%m-%d | %H:%M:%S', time.localtime(time.time())))
    print('----------------------------------------------------------------------')


  