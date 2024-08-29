

# left == tableA
# right == tableB

import copy
import warnings
from func import *
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os




STATS = {}
for task in ['Beer','Walmart-Amazon','Amazon-Google','DBLP-ACM',
            'Fodors-Zagat','iTunes-Amazon','DBLP-GoogleScholar','Febrl']:
    


    left_df = pd.read_csv("data/" + task +'/tableA.csv')
    right_df = pd.read_csv("data/" + task +'/tableB.csv')
    match_df = pd.read_csv("data/" + task +'/matches.csv')
    
    if task == 'Fodors-Zagat':
        left_df = left_df.applymap(lambda x: x.strip('`').strip() if isinstance(x, str) else x)
        left_df = left_df.applymap(lambda x: x.strip("'").strip() if isinstance(x, str) else x)

        right_df = right_df.applymap(lambda x: x.strip('`').strip() if isinstance(x, str) else x)
        right_df = right_df.applymap(lambda x: x.strip("'").strip() if isinstance(x, str) else x)

    left_sens = pd.read_csv("data/" + task +'/left_sens.csv')
    right_sens = pd.read_csv("data/" + task +'/right_sens.csv')
    match_sens = pd.read_csv("data/" + task +'/match_sens.csv')
    
    STATS[task] = {}
    SENS = [match_sens.copy(), left_sens.copy(), right_sens.copy()]
    DATA = [match_df.copy(), left_df.copy(), right_df.copy()]

    # ------------------- Auto Encoder methods --------------------------
    print("Auto Encoder")
    candidatesDB = deepBlock(left_df.copy(), right_df.copy(), K = 50, method = 'AE')
    stat, all_tot_sesn, tot_not_sens = make_block_stat(candidatesDB, copy.deepcopy(DATA), task, copy.deepcopy(SENS))
    STATS[task]['AutoEncode_deepBlock'] = stat

    # ------------------- CTT methods --------------------------
    print('CTT')
    candidatesDB = deepBlock(left_df.copy(), right_df.copy(), K = 50, method = 'CTT')
    stat, all_tot_sesn, tot_not_sens = make_block_stat(candidatesDB, copy.deepcopy(DATA), task, copy.deepcopy(SENS))
    STATS[task]['CTT_deepBlock'] = stat
    save_blk_stat(STATS, output = 'classic_deep.txt')
