import gender_guesser.detector as gender

import numpy as np
import pandas as pd
from DeepBlocksrc.deep_blocker import DeepBlocker
from sklearn.metrics.pairwise import pairwise_distances
from pathlib import Path
import DeepBlocksrc.blocking_utils as blocking_utils
from DeepBlocksrc.tuple_embedding_models import  CTTTupleEmbedding, HybridTupleEmbedding
from DeepBlocksrc.tuple_embedding_models import  AutoEncoderTupleEmbedding
from DeepBlocksrc.vector_pairing_models import ExactTopKVectorPairing

from pyjedai.datamodel import Data
from pyjedai.block_cleaning import BlockPurging, BlockFiltering
from pyjedai.comparison_cleaning import CardinalityEdgePruning,BLAST
from pyjedai.block_building import (
    StandardBlocking,
    ExtendedQGramsBlocking,
    ExtendedSuffixArraysBlocking,
    QGramsBlocking,
    SuffixArraysBlocking
)


block_dict = {
    'Beer': ['Beer_Name', 'Brew_Factory_Name'],
    'Amazon-Google': ['title', 'manufacturer'],  
    'DBLP-ACM': ['title','authors'], 
    'DBLP-GoogleScholar': ['title','venue'],  
    'Fodors-Zagat': ['name','type'],  
    'iTunes-Amazon': ['Song_Name','Genre'],
    'Walmart-Amazon': ['title','category'],
    'Febrl': ['given_name','address_1']
    }
# Define method dictionaries
classic_method_dict = {
                'SB': StandardBlocking(),
                'QG': QGramsBlocking(),
                'EQG': ExtendedQGramsBlocking(),
                'SA': SuffixArraysBlocking(),
                'ESA': ExtendedSuffixArraysBlocking()
                }
classic_method_name = {
    'SB': 'StandardBlocking',
    'EQG': 'ExtendedQGramsBlocking',
    'ESA': 'ExtendedSuffixArraysBlocking',
    'QG': 'QGramsBlocking',
    'SA': 'SuffixArraysBlocking',
    'CTT': 'CTT',
    'AE': 'AUTO',

}


block_dict = {
'Beer': ['Beer_Name', 'Brew_Factory_Name'],
'Amazon-Google': ['title', 'manufacturer'],  
'DBLP-ACM': ['title','authors'], 
'DBLP-GoogleScholar': ['title','venue'],  
'Fodors-Zagat': ['name','type'],  
'iTunes-Amazon': ['Song_Name','Genre'],
'Walmart-Amazon': ['title','category'],
'Febrl': ['given_name','address_1']
}


female_names = ['adriana', 'agma', 'alexandra', 'alice', 'aya', 'barbara', 'betty', 'bhavani', 'carol',
                'carole', 'cass', 'cecilia', 'chia-jung', 'christine', 'clara', 'claudia', 'debra',
                'diane', 'dimitra', 'ebru', 'elaheh', 'elena', 'elisa', 'elke', 'esther', 'fatima',
                'fatma', 'felicity', 'françoise', 'gillian', 'hedieh', 'helen', 'ilaria', 'isabel', 
                'jeanette', 'jeanine', 'jennifer', 'jenny', 'joann', 'julia', 'juliana', 'julie', 
                'kelly', 'kimberly', 'laura', 'letizia', 'ljiljana', 'louiqa', 'lynn', 'maria',
                'marianne', 'melissa', 'meral', 'monica', 'myra', 'pamela', 'patricia', 'paula',
                'pierangela', 'pina', 'rachel', 'sandra', 'sheila', 'sihem', 'silvana', 'sophie',
                'sorana', 'sunita', 'susan', 'teresa', 'tova', 'ulrike', 'vana', 'véronique', 'ya-hui', 'yelena', 'zoé']

# Function to determine gender based on a given name
def gender_rev(name):
    name = name.strip().split()[0].strip()
    d = gender.Detector()
    modified_name = name[0].upper() + name[1:]
    gen_dict = {
        'male': 'male',
        'female': 'female',
        'andy': 'other',
        'mostly_male': 'male',
        'mostly_female': 'female',
        'unknown': 'other'
    }
    return gen_dict[d.get_gender(modified_name)]



sens_dict = {
    'Walmart-Amazon': ['category', 'printers'],  # Equal
    'Beer': ['Beer_Name', 'Red'],  # Continuous
    'Amazon-Google': ['manufacturer', 'Microsoft'],  # Continuous
    'Fodors-Zagat': ['type', 'asian'],  # Equal
    'iTunes-Amazon': ['Genre', 'Dance'],  # Continuous
    'DBLP-GoogleScholar': ['venue', 'vldb j'],  # Continuous
    'DBLP-ACM': ['authors', 'female'],  # Functional
    'COMPAS': ['Ethnic_Code_Text', 'African-American'],
    'Febrl': ['given_name', 'female']
}

def make_sens_vector(df, dataset, sens_dict):
    if dataset in ['Walmart-Amazon', 'Fodors-Zagat', 'COMPAS']:
        df['contains_s'] = df[sens_dict[dataset][0]] == sens_dict[dataset][1]
    elif dataset in ['Beer', 'Amazon-Google', 'iTunes-Amazon', 'DBLP-GoogleScholar']:
        df[sens_dict[dataset][0]] = df[sens_dict[dataset][0]].astype(str)
        df['contains_s'] = df[sens_dict[dataset][0]].str.lower().str.contains(sens_dict[dataset][1].lower())
    else:
        # df[sens_dict[dataset][0]] = df[sens_dict[dataset][0]].astype(str)
        # df['contains_s'] = df[sens_dict[dataset][0]].apply(lambda x: x.replace('&#216;', '').replace('&#214;', '').replace('&#237;', ',').split(',')[-1].strip())
        # df['contains_s'] = df['contains_s'].apply(lambda x: ', '.join([gender_rev(name) for name in x.split(',')]))
        # df['contains_s'] = df['contains_s'].apply(lambda x: 'True' if 'female' in str(x) else 'False')
        # df['contains_s'] = df['contains_s'].apply(lambda x: any(item in x for item in ['True']))
    
        df[sens_dict[dataset][0]] = df[sens_dict[dataset][0]].astype(str)
        df['contains_s'] = df[sens_dict[dataset][0]].apply(lambda x: x.replace('&#216;', '').replace('&#214;', '').replace('&#237;', ',').split(',')[-1].strip())


        sens = [] 
        for x  in list(df['contains_s']):
            try: 
                tmp = str(gender_rev(x))
            except:
                tmp = '?'
            if 'female' in tmp:
                GENDER = True
            else:
                GENDER = False
            sens.append(GENDER)

        df['contains_s'] = sens
        result_vector = np.array(df['contains_s'])
        sens_attr = np.array(result_vector).reshape(-1)
        return sens_attr





    return np.array( df['contains_s']).reshape(-1)





def make_candid_sens(candidate_set_df, left_df, right_df,dataset, sens_dict):
    df = candidate_set_df.copy()


    left_df['Sensitive'] = make_sens_vector(left_df, dataset, sens_dict)
    right_df['Sensitive'] = make_sens_vector(right_df, dataset, sens_dict)

    left_sens_id = list(left_df[left_df['Sensitive'] == True]['id'])
    right_sens_id = list(right_df[right_df['Sensitive'] == True]['id'])

    df['left_Sensitive'] = df['ltable_id'].isin(left_sens_id)
    df['right_Sensitive'] = df['rtable_id'].isin(right_sens_id)

    df['sensitive'] = df['left_Sensitive'] | df['right_Sensitive']
    df = df[['ltable_id','rtable_id','sensitive']]
    return df




def make_block_stat(candidate_set_df_in, DATA , dataset, SENS):
    [match_sens, left_sens , rihgt_sens] = SENS
    [golden_df_in, left_df, right_df] = DATA
    golden_df = golden_df_in.rename(columns={list(golden_df_in.columns)[0]: 'ltable_id', list(golden_df_in.columns)[1]: 'rtable_id'})
    candidate_set_df = candidate_set_df_in.rename(columns={list(candidate_set_df_in.columns)[0]: 'ltable_id', list(candidate_set_df_in.columns)[1]: 'rtable_id'})
    candidate_set_df = candidate_set_df.astype(int)
    statistics_dict = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df)
    
    
    


    candid_sens = make_candid_sens(candidate_set_df, left_df, right_df,dataset, sens_dict)
    


    merged_sens = pd.merge(candidate_set_df, golden_df, on=['ltable_id', 'rtable_id'])
    merged_sens = make_candid_sens(merged_sens, left_df, right_df,dataset, sens_dict)




    PC_minor = np.sum(merged_sens['sensitive'])  / np.sum(match_sens['sensitive'])
    PC_major = np.sum(~merged_sens['sensitive']) / np.sum(~match_sens['sensitive'])



    PQ_minor = np.sum(merged_sens['sensitive'])  / np.sum(candid_sens['sensitive'])
    PQ_major = np.sum(~merged_sens['sensitive'])  / np.sum(~candid_sens['sensitive'])


    a = len(right_df) - np.array(np.sum(rihgt_sens))[0]
    b = len(left_df) - np.array(np.sum(left_sens))[0]

    tot_sesn = len(right_df)*len(left_df) - a*b
    tot_not_sens = a*b
    RR_minor = 1 - np.sum(candid_sens['sensitive'])  / tot_sesn
    RR_major = 1 - np.sum(~candid_sens['sensitive'])  / (a*b)

    statistics_dict['RR_minor'] = RR_minor
    statistics_dict['RR_major'] = RR_major
    statistics_dict['PC_minor'] = PC_minor  
    statistics_dict['PC_major'] = PC_major
    statistics_dict['PQ_minor'] = PQ_minor
    statistics_dict['PQ_major'] = PQ_major

    return statistics_dict, tot_sesn, tot_not_sens






def deepBlock(left_df, right_df, K = 20, method = 'AE',attr  =[]):
    if attr == []:
        cols_to_block = [col for col in left_df.columns if col != 'id']
    else:
        cols_to_block = attr

    
    if method == 'AE':
        tuple_embedding_model = AutoEncoderTupleEmbedding()
    elif method == 'CTT':
        tuple_embedding_model = CTTTupleEmbedding()
    else:
        tuple_embedding_model = HybridTupleEmbedding()


    topK_vector_pairing_model = ExactTopKVectorPairing(K=K)
    db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)
    candidate_set_df, tot_time = db.block_datasets(left_df, right_df, cols_to_block)
    
    # statistics_dict = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df)
    return candidate_set_df, tot_time




def canopy(X1, X2, T1, T2, distance_metric='euclidean', filemap1=None, filemap2=None):
    canopies = dict()
    X_dist = pairwise_distances(X1, X2, metric=distance_metric)
    canopy_points = set(range(X1.shape[0]))
    while canopy_points:
        point = canopy_points.pop()
        i = len(canopies)
        canopies[i] = {"c": point, "points": list(np.where(X_dist[point] < T2)[0])}
        canopy_points = canopy_points.difference(set(np.where(X_dist[point] < T1)[0]))
    if filemap1 and filemap2:
        for canopy_id in canopies.keys():
            canopy = canopies.pop(canopy_id)
            canopy2 = {"c": filemap1[canopy['c']], "points": list()}
            for point in canopy['points']:
                canopy2["points"].append(filemap2[point])
            canopies[canopy_id] = canopy2
    return canopies



def save_blk_stat(STATS, output = 'classic_output.txt'):
    with open(output, 'w') as file:
        for task in STATS.keys():
            for name in STATS[task].keys():
                stat = STATS[task][name]
                file.write(name + ' --> ' + task + '\n')
                file.write(f"{'PC:':<10}{'PC_minor:':<12}{'PC_major:':<10}\n")
                file.write(f"{str(round(100*stat['PC'],2))+' %':<10}{str(round(100*stat['PC_minor'],2))+' %':<12}{str(round(100*stat['PC_major'],2))+' %':<10}\n")
                file.write('\n')
                file.write(f"{'PQ:':<10}{'PQ_minor:':<12}{'PQ_major:':<12}\n")
                file.write(f"{str(round(100*stat['PQ'],2))+' %':<10}{str(round(100*stat['PQ_minor'],2))+' %':<12}{str(round(100*stat['PQ_major'],2))+' %':<10}\n")
                file.write('\n')
                file.write(f"{'RR:':<10}{'RR_minor:':<12}{'RR_major:':<12}\n")
                file.write(f"{str(round(100*stat['RR'],2))+' %':<10}{str(round(100*stat['RR_minor'],2))+' %':<12}{str(round(100*stat['RR_major'],2))+' %':<10}\n")
                file.write('\n\n-------------------------------------\n')
        

import csv

def dynamic_convert_csv_to_txt(input_csv, output_txt):
    with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_txt, mode='w', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        for row in reader:
            parts = {'left': '', 'right': ''}
            for col_name, value in row.items():
                if '_' in col_name:
                    prefix, attribute = col_name.split('_', 1)
                    parts[prefix] += f"COL {attribute} VAL {value} "
            
            left_part = parts.get('left', '').strip()
            right_part = parts.get('right', '').strip()
            label = row.get('label', '').strip()
            outfile.write(f"{left_part} \t{right_part} \t{label}\n")







def calc_bias_block(result_df_in,match_df_in,left_df_in,right_df_in, task , sens_dict, input_match_sens = []):

    result_df  = result_df_in.copy()
    match_df  = match_df_in.copy()
    left_df  = left_df_in.copy()
    right_df  = right_df_in.copy()

    col, criteria = sens_dict[task]
    if input_match_sens ==[]:
        # --------------------------------
        r_sens_col = result_df[col+'_right']
        l_sens_col = result_df[col+'_left']

        a = make_sens_vector(pd.DataFrame({col:r_sens_col}),task, sens_dict)
        b = make_sens_vector(pd.DataFrame({col:l_sens_col}),task, sens_dict)

        result = result_df.copy()
        result['sens'] = np.logical_or(a, b)
        print('sens res done')
    else:
        result = result_df.copy()
        result['sens']= input_match_sens


    # --------------------------------
    try:
        match_df_sens = pd.read_csv('/Users/mohammad/Desktop/IEEE Blocking/MATCH_sens'+task+'.csv')
    except:

        left_M = list(match_df['ltable_id'])
        right_M = list(match_df['rtable_id'])

        left_arr = []
        for id in left_M:
            try: 
                row = list(left_df[left_df['id'] == id][col])[0]
            except:
                row = ''
            left_arr.append(row)


        right_arr = []
        for id in right_M:
            try: 
                row = list(right_df[right_df['id'] == id][col])[0]
            except:
                row = ''
            right_arr.append(row)


        a = make_sens_vector(pd.DataFrame({col:left_arr}),task, sens_dict)
        b = make_sens_vector(pd.DataFrame({col:right_arr}),task, sens_dict)

        match_df_sens = match_df.copy()
        match_df_sens['sens'] = np.logical_or(a, b)
        match_df_sens.to_csv('/Users/mohammad/Desktop/IEEE Blocking/MATCH_sens'+task+'.csv',index = False)
        print('sens match done')

    # # --------------------------------




    C_minor = np.sum(result['sens'] == True)
    C_major = np.sum(result['sens'] == False)

    C_minor_M = np.sum(np.logical_and(result['sens'] ,result['label'] == 1 ))
    C_major_M = np.sum(np.logical_and(~result['sens'] ,result['label'] == 1 ))

    try:
        a = np.array(list(pd.read_csv('/Users/mohammad/Desktop/IEEE Blocking/data/'+task+'/left_sens.csv')['0']))
        b = np.array(list(pd.read_csv('/Users/mohammad/Desktop/IEEE Blocking/data/'+task+'/right_sens.csv')['0']))

    except:
        print('failed')

        a = make_sens_vector(pd.DataFrame({col:left_df[col]}),task, sens_dict)
        b = make_sens_vector(pd.DataFrame({col:right_df[col]}),task, sens_dict)

    P = len(a) * len(b)
    P_major = np.sum(~a) * np.sum(~b)
    P_minor = P - P_major 


    M_minor = np.sum(match_df_sens['sens'] == True)
    M_major = np.sum(match_df_sens['sens'] == False)



    RR_minor = 100*(1 - C_minor/P_minor)
    PC_minor = 100*(C_minor_M / M_minor)
    PQ_minor = 100*(C_minor_M / C_minor)

    RR_major = 100*(1 - C_major/P_major)
    PC_major = 100*(C_major_M / M_major)
    PQ_major = 100*(C_major_M / C_major)


    Fb_minor = 2 * PC_minor * RR_minor / (PC_minor  + RR_minor)
    Fb_major = 2 * PC_major * RR_major / (PC_major  + RR_major)

    return [RR_minor,PC_minor,PQ_minor,Fb_minor ] , [RR_major,PC_major,PQ_major,Fb_major ], [P_major, P_minor, M_major, M_minor]



import time
import copy
def trad_blk(task, method, left_df_in, right_df_in, match_df_in, attr_type = 'all'):
    left_df, right_df, match_df = left_df_in.copy(), right_df_in.copy(), match_df_in.copy()
# Process using classic method if applicable
    bb = classic_method_dict[method]
    if attr_type == 'all':
        attr = [col for col in left_df.columns if col not in ['id']]
    else:
        attr = [col for col in left_df.columns if col not in ['id',block_dict[task][1]]]
    data = Data(
        dataset_1=left_df.copy(), id_column_name_1='id',
        dataset_2=right_df.copy(), id_column_name_2='id',
        ground_truth=match_df.rename(columns={list(match_df.columns)[0]: 'D1', list(match_df.columns)[1]: 'D2'}))
    data.clean_dataset(remove_stopwords=False, remove_punctuation=False, remove_numbers=False, remove_unicodes=True)
    start_time = time.time()
    blocks = bb.build_blocks(copy.deepcopy(data), attributes_1=attr, attributes_2=attr, tqdm_disable=True)
    
    bp = BlockPurging()
    bf = BlockFiltering()
    mb = BLAST('EJS')
    if task == 'iTunes-Amazon': mb = CardinalityEdgePruning()
    # for meta in META:
    cleaned_blocks = bf.process(copy.deepcopy(blocks), data, tqdm_disable=True)
    filtered_blocks = bp.process(cleaned_blocks, data, tqdm_disable=True)
    candidate_pairs_blocks = mb.process(filtered_blocks, data, tqdm_disable=False)
    candidates = mb.export_to_df(candidate_pairs_blocks)
    candidates= candidates.astype(int)
    candidates.rename(columns={'ltable_id': 'id1', 'rtable_id': 'id2'}, inplace=True)
    endtime = time.time()

    left_merged = candidates.merge(left_df, left_on='id1', right_on='id', suffixes=('', '_left'))
    right_merged = left_merged.merge(right_df, left_on='id2', right_on='id', suffixes=('_left', '_right'))
    result_df = right_merged.copy()

    # Merge with match_df to determine labels
    merged_df = result_df.merge(match_df, left_on=['id_left', 'id_right'], right_on=['ltable_id', 'rtable_id'], how='left', indicator=True)
    result_df['label'] = (merged_df['_merge'] == 'both').astype(int)

    return candidates, (endtime- start_time), result_df


def load_blk_data(task):
        # Load datasets
    left_df = pd.read_csv(f"data/{task}/tableA.csv")
    right_df = pd.read_csv(f"data/{task}/tableB.csv")
    match_df = pd.read_csv(f"data/{task}/matches.csv")


    left_df = left_df.replace(r"\\ '", "'", regex=True).replace(r" '", "'", regex=True).replace(r"\\ `", "\\ ", regex=True)
    right_df = right_df.replace(r"\\ '", "'", regex=True).replace(r" '", "'", regex=True).replace(r"\\ `", "\\ ", regex=True)
    
    left_df = left_df.applymap(lambda x: x.strip('`') if isinstance(x, str) else x).applymap(lambda x: x.strip("'") if isinstance(x, str) else x).applymap(lambda x: x.strip() if isinstance(x, str) else x)
    right_df = right_df.applymap(lambda x: x.strip('`') if isinstance(x, str) else x).applymap(lambda x: x.strip("'") if isinstance(x, str) else x).applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return left_df, right_df,match_df