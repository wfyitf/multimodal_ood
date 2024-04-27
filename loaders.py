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

class DataLoader:
    def __init__(self, data_source):
        if data_source == "qa":
            self.data_source = "qa"
            self.data_dir = constant.QA_DATA_DIR

        elif data_source == "real":
            self.data_source = "real"
            self.data_dir = constant.REAL_DATA_DIR
        
        self.supercategories = constant.SUPERCATEGORIES
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_image_dir = self.data_dir / 'sample'

    def load_dialogue_df(self):
        data_path = self.data_dir / 'sample.json'
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
            print('Image ID:', df_table.iloc[num]['image_id'])
            print('*'*50)
            print('Categories:', df_table.iloc[num]['categories'])
            print('Super Categories:', df_table.iloc[num]['supercategories'])
            print('*'*50)
            if self.data_source == "qa":
                print(df_table.iloc[num]['dialog_full'])
                print('*'*50)
                self.plot_image(df_table.iloc[num]['caption'], 
                        f"{self.data_image_dir}/COCO_train2014_{df_table.iloc[num]['image_id']:0>12}.jpg")
            elif self.data_source == "real":
                count = 0 
                for i in df_table.iloc[num]['dialog']:
                    if count % 2 == 0:
                        print('S1:', i)
                    else:
                        print('S2:', i)
                    count += 1
                print('*'*50)

                print('Similarity:', df_table.iloc[num]['score'])
                print('*'*50)
                self.plot_image(f"Image: {df_table.iloc[num]['image_id']}",
                            f"{self.data_image_dir}/{df_table.iloc[num]['image_id']}.jpg")
                print('*'*50)
            return True
        except Exception as e:
            print(e)
            print('Please provide a valid index')
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
            print(categories + ' similarity: ' + str(cosine_sim.item()))
            scores.append(cosine_sim.item())
        if verbose:
            print(f'True labels: {df_table.iloc[num]['supercategories']}')


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
        return features_df