import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import io
import sys
import random


logger = logging.getLogger(__name__)

class MultiLabelNN(nn.Module):
    def __init__(self, 
                 logger, 
                 input_size=512, 
                 output_size=11
                 ):
        
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256) 
        self.fc2 = nn.Linear(256, 128) 
        self.fc3 = nn.Linear(128, 64) 
        self.fc4 = nn.Linear(64, output_size)    
        self.relu = nn.ReLU()           
        self.sigmoid = nn.Sigmoid()    

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))  
        return x
    

class model_loader:
    def __init__(self, 
                 logger = logger, 
                 input_size=512, 
                 output_size=11, 
                 batch_size=32, 
                 learning_rate=0.001, 
                 proportion = 0.8,
                 num_epochs=20,
                 seed = 42
                 ):
        
        self.model = MultiLabelNN(logger, input_size, output_size)
        self.logger = logger
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.proportion = proportion
        self.random_seed = seed
        self.set_seed()

    def set_seed(self):
        """
        Set the random seed for numpy, random, and TensorFlow
        """
        self.logger.info(f"Setting random seed: {self.random_seed}")
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed) 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    def create_optimizer(self, customised_optimizer = None):
        """
        Create an optimizer for the model
        """
        if customised_optimizer is None:
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            return customised_optimizer(self.model.parameters(), lr=self.learning_rate)
    
    def define_loss(self, loss):
        """
        Define the loss function for the model
        """
        if loss is None:
            return nn.BCELoss()
        else:
            return loss

    def compile_model(self):
        """
        Compile the model with the optimizer and loss function
        """
        self.optimizer = self.create_optimizer()
        self.loss_function = self.define_loss(None)


    def create_dataset(self, data_loader, df_table):
        """
        Create a dataset for the model
        """
        df_ind = df_table[df_table['OOD'] == 0]
        df_ind_train = df_ind.sample(frac=self.proportion)
        df_ind_train = df_ind_train.loc[np.sort(df_ind_train.index)]
        df_test = df_table.drop(df_ind_train.index)

        X_train_image = np.stack(df_ind_train['image_clip'].values)
        X_test_image = np.stack(df_test['image_clip'].values)
        X_train_dialogue = np.stack(df_ind_train['dialogue_clip'].values)
        X_test_dialogue = np.stack(df_test['dialogue_clip'].values)
        Y_train = np.stack(df_ind_train['encoded_label'].values)
        Y_test = np.stack(df_test['encoded_label'].values)

        return df_ind_train, df_test, X_train_image, X_test_image, X_train_dialogue, X_test_dialogue, Y_train, Y_test


    def train_model(self, X_train, Y_train, verbose = 1):
        """
        Train the model
        """
        X_train_tensor = torch.tensor(X_train).float() 
        Y_train_tensor = torch.tensor(Y_train).float()
        dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)  
        self.compile_model()

        if verbose == 1:
            iterator = tqdm(range(self.num_epochs))
        else:
            iterator = range(self.num_epochs)

        for epoch in iterator:
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

            average_loss, average_accuracy = self.evaluate(train_loader, self.loss_function)
            print(f'Epoch {epoch+1}, Train Loss: {average_loss:.4f}, Train Accuracy: {average_accuracy:.4f}')


    def evaluate(self, data_loader, loss_function, score = "energy", return_score = False):
        """
        Evaluate the model
        """
        self.model.eval()  
        total_loss = 0
        total_accuracy = 0
        total_samples = 0

        score_sum = []
        score_max = []

        with torch.no_grad():  
            for inputs, labels in data_loader:
                outputs = self.model(inputs)
                if score == "energy":
                    outputs_np = outputs.cpu().numpy()
                    outputs_energy = -np.log(1+outputs_np/(1-outputs_np))
                    score_sum.append(outputs_energy.sum(axis = 1))
                    score_max.append(outputs_energy.min(axis = 1))

                loss = loss_function(outputs, labels)
                total_loss += loss.item()
                predictions = outputs > 0.5 
                total_accuracy += (predictions == labels.byte()).all(dim=1).float().mean().item()
                total_samples += 1
            
        average_loss = total_loss / total_samples
        average_accuracy = total_accuracy / total_samples

        if return_score:
            score_sum = np.concatenate(score_sum)
            score_max = np.concatenate(score_max)
            return average_loss, average_accuracy, score_sum, score_max
        else:
            return average_loss, average_accuracy
    


