import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import constant 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn.functional as F
from PIL import Image
import torch
import clip
from IPython.display import display
from tqdm import tqdm
from typing import Optional

logger = logging.getLogger(__name__)
class DataLoader:
    def __init__(self, data_source, model_type = "clip", logger = logger):
        if data_source == "qa":
            self.data_source = "qa"
            self.data_dir = constant.QA_DATA_DIR

        elif data_source == "real":
            self.data_source = "real"
            self.data_dir = constant.REAL_DATA_DIR

        elif data_source == "meld":
            self.data_source = "meld"
            self.data_dir = constant.MELD_DATA_DIR
        
        self.model_type = model_type
        self.logger = logger
        if data_source in ["qa", "real"]:
            self.supercategories = constant.SUPERCATEGORIES
        elif data_source == "meld":
            self.supercategories = constant.MELD_CATEGORIES
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_image_dir = self.data_dir / 'sample'

    def load_dialogue_df(self, split: Optional[str] = 'train'):
        if self.data_source == "real":
            data_path = self.data_dir / 'sample.json'
        elif self.data_source == "qa":
            if self.model_type == "clip":
                data_path = self.data_dir / 'CLIP' / 'qa_dialogs_truncate' / 'qa.json'
                return pd.read_json(data_path)
            elif self.model_type == "blip":
                data_path = self.data_dir / 'BLIP' / 'qa_dialogs' / 'qa_blip_dialog_feature.npy'
                return np.load(data_path, allow_pickle=True)
            
        elif self.data_source == "meld":
            if self.model_type == 'clip':
                data_path = self.data_dir / 'CLIP' / f'{split}_text_features.npy'
                return np.load(data_path, allow_pickle=True)
            elif self.model_type == 'blip':
                data_path = self.data_dir / 'BLIP' / f'{split}_text_features.npy'
                return np.load(data_path, allow_pickle=True)
            
    def load_image_df(self, split: Optional[str] = 'train'):
        if self.data_source == "real":
            data_path = self.data_dir / 'sample.json'
        elif self.data_source == "qa":
            if self.model_type == "clip":
                data_path = self.data_dir / 'CLIP' / 'qa_images' / 'qa.json'
                return pd.read_json(data_path)
            elif self.model_type == "blip":
                data_path = self.data_dir / 'BLIP' / 'qa_image' / 'qa_blip_image_feature.npy'
                return np.load(data_path, allow_pickle=True)
        
        elif self.data_source == "meld":
            if self.model_type == 'clip':
                data_path = self.data_dir / 'CLIP' / f'{split}_img_features.npy'
                return np.load(data_path, allow_pickle=True)
            elif self.model_type == 'blip':
                data_path = self.data_dir / 'BLIP' / f'{split}_img_features.npy'
                return np.load(data_path, allow_pickle=True)

    def load_annotations_df(self, split: Optional[str] = 'train'):
        if self.data_source == "real":
            data_path = self.data_dir / 'sample.json'
        elif self.data_source == "qa":
            data_path = self.data_dir / 'qa.json'
        elif self.data_source == "meld":
            data_path = self.data_dir /  f'clean_{split}.json'
        return pd.read_json(data_path)

    def plot_image(self, caption, image_path):
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis('off') 
        plt.title(caption)
        plt.show()

    def showing_example(self, num):
        df_table = self.load_dialogue_df()
        try:
            self.logger.info(f'Image ID:, {df_table.iloc[num]['image_id']}')
            self.logger.info('*'*50)
            self.logger.info(f'Categories: {df_table.iloc[num]['categories']}')
            self.logger.info(f'Super Categories: {df_table.iloc[num]['supercategories']}')
            self.logger.info('*'*50)
            if self.data_source == "qa":
                self.logger.info(f"{df_table.iloc[num]['dialog_full']}")
                self.logger.info('*'*50)
                self.plot_image(df_table.iloc[num]['caption'], 
                        f"{self.data_image_dir}/COCO_train2014_{df_table.iloc[num]['image_id']:0>12}.jpg")
            elif self.data_source == "real":
                count = 0 
                for i in df_table.iloc[num]['dialog']:
                    if count % 2 == 0:
                        self.logger.info(f'S1: {i}')
                    else:
                        self.logger.info(f'S2: {i}')
                    count += 1
                self.logger.info('*'*50)

                self.logger.info(f"Similarity: {df_table.iloc[num]['score']}")
                self.logger.info('*'*50)
                self.plot_image(f"Image: {df_table.iloc[num]['image_id']}",
                            f"{self.data_image_dir}/{df_table.iloc[num]['image_id']}.jpg")
                self.logger.info('*'*50)
            return True
        except Exception as e:
            self.logger.info(e)
            self.logger.info('Please provide a valid index')
            return False 
        
    def show_clip_similarity(self, num, df_table, model, preprocess, verbose = 0):
        if self.data_source == "qa":
            image_path = f"{self.data_image_dir}/COCO_train2014_{df_table.iloc[num]['image_id']:0>12}.jpg"
        elif self.data_source == "real":
            image_path = f"{self.data_image_dir}/{df_table.iloc[num]['image_id']}.jpg"

        image = Image.open(image_path)
        if verbose:
            display(image)
        preprocessed_image = preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = model.encode_image(preprocessed_image)

        scores = []
        for categories in self.supercategories:
            text = 'This photo contains' + categories
            text_tokens = clip.tokenize([text]).to(self.device)  
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
            cosine_sim = F.cosine_similarity(image_features, text_features)
            self.logger.info(categories + ' CLIP feature similarity: ' + str(cosine_sim.item()))
            scores.append(cosine_sim.item())
        if verbose:
            self.logger.info(f'True labels: {df_table.iloc[num]['supercategories']}')


    def encode_images(self, df_table, preprocess, model, verbose = 0):
        features_list = []
        if self.data_source == "qa":
            df_table['image_id'] = df_table['image_id'].apply(lambda x: f"COCO_train2014_{int(x):012d}")
        
        iterator = tqdm(df_table.iterrows(), total=df_table.shape[0]) if verbose else df_table.iterrows()

        for index, row in iterator:
            image_path = f"{self.data_image_dir}/{row['image_id']}.jpg"
            image = Image.open(image_path)
            preprocessed_image = preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = model.encode_image(preprocessed_image)
                image_features = image_features.cpu().numpy().flatten() 

            features_list.append(image_features)

        features_df = pd.DataFrame(features_list)  
        self.save_clip_features(features_df)
        return features_df
    

    def encode_dialogues(self, df_table, model, verbose = 0):
        #TODO: check the possiblity
        features_list = []
        iterator = tqdm(df_table.iterrows(), total=df_table.shape[0]) if verbose else df_table.iterrows()

        for index, row in iterator:
            dialog = ' '.join(row['dialog_full'])
            text_tokens = clip.tokenize([dialog]).to(self.device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features.cpu().numpy().flatten()
            features_list.append(text_features)

        features_df = pd.DataFrame(features_list)  
        self.save_clip_features(features_df)
        return features_df

    def save_clip_features(self, features_df):
        if self.data_source == "qa":
            features_df.to_json(self.data_dir / 'qa_clip_features.json', index=False)
        elif self.data_source == "real":
            features_df.to_json(self.data_dir / 'real_clip_features.json', index=False)
        self.logger.info('Features saved successfully')

    def load_clip_image_features(self, df_table, model, preprocess, verbose = 0):
        if self.data_source == "qa":
            try:
                return pd.read_json(self.data_dir / 'qa_clip_image_features.json')
            except:
                self.logger.info('CLIP features not found, start encoding images')
                return self.encode_images(df_table, preprocess, model, verbose)
            
        elif self.data_source == "real":
            try:
                return pd.read_json(self.data_dir / 'real_clip_image_features.json')
            except:
                self.logger.info('CLIP features not found, start encoding images')
                return self.encode_images(df_table, preprocess, model, verbose)
            
    def load_clip_dialogue_features(self, df_table, model, verbose = 0):
        if self.data_source == "qa":
            try:
                return pd.read_json(self.data_dir / 'qa_clip_dialogue_features.json')
            except:
                self.logger.info('CLIP features not found, start encoding dialogues')
                return self.encode_dialogues(df_table, model, verbose)
            
        elif self.data_source == "real":
            try:
                return pd.read_json(self.data_dir / 'real_clip_dialogue_features.json')
            except:
                self.logger.info('CLIP features not found, start encoding dialogues')
                return self.encode_dialogues(df_table, model, verbose)
            
    