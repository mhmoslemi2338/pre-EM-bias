

# left == tableA
# right == tableB

import copy
import recordlinkage
import warnings
from func import *
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.block_cleaning import BlockPurging
from pyjedai.block_cleaning import BlockFiltering
from pyjedai.block_building import (
    StandardBlocking,
    QGramsBlocking,
    ExtendedQGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking,
)
from pyjedai.comparison_cleaning import BLAST
import os

from pyjedai.comparison_cleaning import (
    WeightedEdgePruning,
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning,
    BLAST,
    ReciprocalCardinalityNodePruning,
    ReciprocalWeightedNodePruning,
    ComparisonPropagation
)

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from DeepBlocksrc.tuple_embedding_models import AutoEncoderTupleEmbedding
from DeepBlocksrc.deep_blocker import canopy_deep_blocker

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



    # ------------------- Sorted  Neighbourhood --------------------------    
    print('Sorted Neighbourhood')
    indexer = recordlinkage.index.SortedNeighbourhood(block_dict[task][0])#,block_left_on = [block_dict[task][0]], block_right_on = [block_dict[task][0]], window=5)
    res = indexer.index(left_df.copy(), right_df.copy())
    res = pd.DataFrame(np.array(res))
    candidates = pd.DataFrame(res[0].tolist(), columns=['ltable_id', 'rtable_id'])
    # print(task , candidates.shape)
    stat, all_tot_sesn, tot_not_sens = make_block_stat(candidates, copy.deepcopy(DATA), copy.deepcopy(task), copy.deepcopy(SENS))
    STATS[task]['SortedNeighbourhood'] = stat
    save_blk_stat(STATS, output = 'classic_output.txt')


    # ------------------- Canopy --------------------------    
    print('Canopy')
    tuple_embedding_model = AutoEncoderTupleEmbedding()
    db = canopy_deep_blocker(tuple_embedding_model)
    cols_to_block = [col for col in left_df.columns if col != 'id']
    candidate_set_df = db.block_datasets(left_df, right_df, cols_to_block)

    T1 = 0.01
    T2 = 0.5
    candidates = []
    canopies = canopy(candidate_set_df[0], candidate_set_df[1], T1, T2, distance_metric='l2')
    for k in canopies.keys():
        tmp = canopies[k]
        center = tmp['c']
        points = tmp['points']
        for p in points:
            candidates.append((center, p))
    stat, all_tot_sesn, tot_not_sens = make_block_stat(pd.DataFrame(candidates), copy.deepcopy(DATA), task, copy.deepcopy(SENS))
    STATS[task]['Canopy'] = stat
    save_blk_stat(STATS, output = 'classic_output.txt')



    # ------------------- Classic methods --------------------------    
    names = ['StandardBlocking', 'QGramsBlocking', 'ExtendedQGramsBlocking', 'SuffixArraysBlocking', 'ExtendedSuffixArraysBlocking']
    attr = [col for col in left_df.columns if col != 'id']
    for i,bb in enumerate([StandardBlocking(), QGramsBlocking(),ExtendedQGramsBlocking(),SuffixArraysBlocking(),ExtendedSuffixArraysBlocking()]):
        match_df2 = match_df.copy()
        data = Data(dataset_1=left_df.copy(),id_column_name_1='id', dataset_2=right_df.copy(),id_column_name_2='id',
                ground_truth=match_df2.rename(columns={list(match_df2.columns)[0]: 'D1', list(match_df2.columns)[1]: 'D2'}))
        data.clean_dataset(remove_stopwords = False, remove_punctuation = False, remove_numbers = False,remove_unicodes = False)
        blocks = bb.build_blocks(copy.deepcopy(data), attributes_1=attr, attributes_2=attr)
        print(names[i])
        bp = BlockPurging()
        bf = BlockFiltering(ratio=0.8)
        mb = BLAST()
        if task == 'iTunes-Amazon': mb = CardinalityEdgePruning()

        cleaned_blocks = bp.process(blocks, data, tqdm_disable=False,)
        filtered_blocks = bf.process(cleaned_blocks, data, tqdm_disable=False)
        candidate_pairs_blocks = mb.process(cleaned_blocks, data, tqdm_disable=False)
        candidates=mb.export_to_df(candidate_pairs_blocks)
        stat, all_tot_sesn, tot_not_sens = make_block_stat(candidates, copy.deepcopy(DATA), copy.deepcopy(task), copy.deepcopy(SENS))
        STATS[task][names[i]] = stat
        save_blk_stat(STATS, output = 'classic_output.txt')

