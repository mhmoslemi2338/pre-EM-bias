#GiG
import numpy as np
import pandas as pd
from pathlib import Path
import DeepBlocksrc.blocking_utils as blocking_utils
import warnings
warnings.filterwarnings("ignore")


class DeepBlocker:
    def __init__(self, tuple_embedding_model, vector_pairing_model):
        self.tuple_embedding_model = tuple_embedding_model
        self.vector_pairing_model = vector_pairing_model

    def validate_columns(self):
        #Assumption: id column is named as id
        if "id" not in self.cols_to_block:
            self.cols_to_block.append("id")
        self.cols_to_block_without_id = [col for col in self.cols_to_block if col != "id"]

        #Check if all required columns are in left_df
        check = all([col in self.left_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the left dataset")

        #Check if all required columns are in right_df
        check = all([col in self.right_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the right dataset")


    def preprocess_datasets(self):
        self.left_df = self.left_df[self.cols_to_block]
        self.right_df = self.right_df[self.cols_to_block]

        # self.left_df.fillna(' ', inplace=True)
        # self.right_df.fillna(' ', inplace=True)
        
        self.left_df.loc[:, :] = self.left_df.fillna(' ')
        self.right_df.loc[:, :] = self.right_df.fillna(' ')

        

        self.left_df = self.left_df.astype(str)
        self.right_df = self.right_df.astype(str)


        self.left_df["_merged_text"] = self.left_df[self.cols_to_block_without_id].agg(' '.join, axis=1)
        self.right_df["_merged_text"] = self.right_df[self.cols_to_block_without_id].agg(' '.join, axis=1)

        #Drop the other columns
        self.left_df = self.left_df.drop(columns=self.cols_to_block_without_id)
        self.right_df = self.right_df.drop(columns=self.cols_to_block_without_id)


    def block_datasets(self, left_df, right_df, cols_to_block):
        self.left_df = left_df
        self.right_df = right_df
        self.cols_to_block = cols_to_block

        self.validate_columns()
        self.preprocess_datasets()

        # print("Performing pre-processing for tuple embeddings ")
        all_merged_text = pd.concat([self.left_df["_merged_text"], self.right_df["_merged_text"]], ignore_index=True)
        self.tuple_embedding_model.preprocess(all_merged_text)
        for _ in range(1):
            import time
            start_time = time.time()

            # print("Obtaining tuple embeddings for left table")
            self.left_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.left_df["_merged_text"])
            # print("Obtaining tuple embeddings for right table")
            self.right_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.right_df["_merged_text"])


            # print("Indexing the embeddings from the right dataset")
            self.vector_pairing_model.index(self.right_tuple_embeddings)

            # print("Querying the embeddings from left dataset")
            topK_neighbors = self.vector_pairing_model.query(self.left_tuple_embeddings)

            for i,row in enumerate(topK_neighbors):
                for j,row2 in enumerate(row):
                    topK_neighbors[i][j] = right_df.iloc[row2]['id']


            topK_df = pd.DataFrame(topK_neighbors)
            topK_df["ltable_id"] = left_df['id']
            melted_df = pd.melt(topK_df, id_vars=["ltable_id"])
            melted_df["rtable_id"] = melted_df["value"]
            self.candidate_set_df = melted_df[["ltable_id", "rtable_id"]].fillna(-10)
            self.candidate_set_df = self.candidate_set_df.astype(int)
            end_time = time.time()



        # self.candidate_set_df = blocking_utils.topK_neighbors_to_candidate_set(topK_neighbors)

        return self.candidate_set_df, end_time - start_time





class canopy_deep_blocker:
    def __init__(self, tuple_embedding_model):
        self.tuple_embedding_model = tuple_embedding_model

    def validate_columns(self):
        #Assumption: id column is named as id
        if "id" not in self.cols_to_block:
            self.cols_to_block.append("id")
        self.cols_to_block_without_id = [col for col in self.cols_to_block if col != "id"]

        #Check if all required columns are in left_df
        check = all([col in self.left_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the left dataset")

        #Check if all required columns are in right_df
        check = all([col in self.right_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the right dataset")


    def preprocess_datasets(self):
        self.left_df = self.left_df[self.cols_to_block]
        self.right_df = self.right_df[self.cols_to_block]

        # self.left_df.fillna(' ', inplace=True)
        # self.right_df.fillna(' ', inplace=True)
        
        self.left_df.loc[:, :] = self.left_df.fillna(' ')
        self.right_df.loc[:, :] = self.right_df.fillna(' ')

        

        self.left_df = self.left_df.astype(str)
        self.right_df = self.right_df.astype(str)


        self.left_df["_merged_text"] = self.left_df[self.cols_to_block_without_id].agg(' '.join, axis=1)
        self.right_df["_merged_text"] = self.right_df[self.cols_to_block_without_id].agg(' '.join, axis=1)

        #Drop the other columns
        self.left_df = self.left_df.drop(columns=self.cols_to_block_without_id)
        self.right_df = self.right_df.drop(columns=self.cols_to_block_without_id)


    def block_datasets(self, left_df, right_df, cols_to_block):
        self.left_df = left_df
        self.right_df = right_df
        self.cols_to_block = cols_to_block

        self.validate_columns()
        self.preprocess_datasets()

        # print("Performing pre-processing for tuple embeddings ")
        all_merged_text = pd.concat([self.left_df["_merged_text"], self.right_df["_merged_text"]], ignore_index=True)
        self.tuple_embedding_model.preprocess(all_merged_text)

        # print("Obtaining tuple embeddings for left table")
        self.left_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.left_df["_merged_text"])
        # print("Obtaining tuple embeddings for right table")
        self.right_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.right_df["_merged_text"])




        # self.candidate_set_df = blocking_utils.topK_neighbors_to_candidate_set(topK_neighbors)

        return self.left_tuple_embeddings , self.right_tuple_embeddings
