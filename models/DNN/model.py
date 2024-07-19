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
from pathlib import Path

logger = logging.getLogger(__name__)

class MultiLabelNN(nn.Module):
    def __init__(self, 
                 logger, 
                 input_size=512, 
                 output_size=11
                 ):
        
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512) 
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128) 
        self.fc4 = nn.Linear(128, 64) 
        self.fc5 = nn.Linear(64, output_size)    
        self.relu = nn.ReLU()           
        self.sigmoid = nn.Sigmoid()    

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))  
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
                 seed = 42,
                 type = "image"
                 ):
        self.input_size = input_size
        self.output_size = output_size
        self.logger = logger
        self.model = MultiLabelNN(self.logger, self.input_size, self.output_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.proportion = proportion
        self.random_seed = seed
        self.type = type
        self.set_seed()
        self.model_path = Path(__file__).parent / 'models' / 'DNN' 
        if not self.model_path.exists():
            self.model_path.mkdir(parents=True, exist_ok=True)

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
        self.model = MultiLabelNN(self.logger, self.input_size, self.output_size)
        self.optimizer = self.create_optimizer()
        self.loss_function = self.define_loss(None)


    def create_dataset(self, data_loader, df_table, add_mismatch = False, mismatch_num = 10000, used_model = "clip"):
        """
        Create a dataset for the model
        """
        df_ind = df_table[df_table['OOD'] == 1]
        df_ind_train = df_ind.sample(frac=self.proportion)
        df_ind_train = df_ind_train.loc[np.sort(df_ind_train.index)]
        df_test = df_table.drop(df_ind_train.index)

        if add_mismatch:
            if data_loader.data_source == "qa":
                df_sample = df_table.sample(n = mismatch_num, random_state=42).reset_index(drop=True)
                df_dialogue_sample = df_table.sample(n = mismatch_num, random_state=24).reset_index(drop=True)

                while any(df_sample['image_id'] == df_dialogue_sample['image_id']):
                    df_dialogue_sample = df_table.sample(n = mismatch_num, random_state=np.random.randint(0, 10000)).reset_index(drop=True)

                df_sample['dialog'] = df_dialogue_sample['dialog']
                df_sample['dialog_full'] = df_dialogue_sample['dialog_full']
                df_sample[f'dialogue_{used_model}'] = df_dialogue_sample[f'dialogue_{used_model}']
                df_sample['OOD'] = 0

            elif data_loader.data_source == "meld":
                df_sample = df_table.sample(n = mismatch_num, random_state=42).reset_index(drop=True)
                df_dialogue_sample = df_table.sample(n = mismatch_num, random_state=24).reset_index(drop=True)

                while any(df_sample['image_id'] == df_dialogue_sample['image_id']):
                    df_dialogue_sample = df_table.sample(n = mismatch_num, random_state=np.random.randint(0, 10000)).reset_index(drop=True)

                df_sample['utterance'] = df_dialogue_sample['utterance']
                df_sample[f'dialogue_{used_model}'] = df_dialogue_sample[f'dialogue_{used_model}']
                df_sample['OOD'] = 0


        X_train_image = np.stack(df_ind_train[f'image_{used_model}'].values)
        X_test_image = np.stack(df_test[f'image_{used_model}'].values)
        X_train_dialogue = np.stack(df_ind_train[f'dialogue_{used_model}'].values)
        X_test_dialogue = np.stack(df_test[f'dialogue_{used_model}'].values)
        Y_train = np.stack(df_ind_train['encoded_label'].values)
        Y_test = np.stack(df_test['encoded_label'].values)

        if add_mismatch:
            X_test_image = np.concatenate([X_test_image, np.stack(df_sample[f'image_{used_model}'].values)])
            X_test_dialogue = np.concatenate([X_test_dialogue, np.stack(df_sample[f'dialogue_{used_model}'].values)])
            Y_test = np.concatenate([Y_test, np.stack(df_sample['encoded_label'].values)])
            df_test = pd.concat([df_test, df_sample], ignore_index=True)

        return df_ind_train, df_test, X_train_image, X_test_image, X_train_dialogue, X_test_dialogue, Y_train, Y_test


    def load_model(self, ood_category = []):
        """
        Load the model
        """
        path = self.model_path / f'{self.type}_model_{ood_category}_{self.num_epochs}_{self.learning_rate}.pth'
        self.model.load_state_dict(torch.load(path))
        self.logger.info(f'Model loaded from {path}')

    def train_model(self, X_train, Y_train, X_test = None, Y_test = None, verbose = 1, ood_category = [], save_model = True, retrain = True):
        """
        Train the model
        """
        X_train_tensor = torch.tensor(X_train).float() 
        Y_train_tensor = torch.tensor(Y_train).float()
        dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)  
        if retrain:
            self.compile_model()

        if X_test is not None and Y_test is not None:
            X_test_tensor = torch.tensor(X_test).float()
            Y_test_tensor = torch.tensor(Y_test).float()
            test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


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
            test_loss, test_accuracy = self.evaluate(test_loader, self.loss_function) if X_test is not None and Y_test is not None else (None, None)
            if X_test is not None and Y_test is not None:
                self.logger.info(f'Epoch {epoch+1}, Train Loss: {average_loss:.4f}, Train Accuracy: {average_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            else:
                self.logger.info(f'Epoch {epoch+1}, Train Loss: {average_loss:.4f}, Train Accuracy: {average_accuracy:.4f}')

        if save_model:
            path = self.model_path / f'{self.type}_model_{ood_category}_{self.num_epochs}_{self.learning_rate}.pth'
            torch.save(self.model.state_dict(), path)
            self.logger.info(f'Model saved at {path}')

    def evaluate_on_test(self, X_test, Y_test, return_score = False, score_type = "energy", verbose = 1, X_train = None, Y_train = None):
        """
        Evaluate the model on the test set
        """
        X_test_tensor = torch.tensor(X_test).float() 
        Y_test_tensor = torch.tensor(Y_test).float()
        dataset = TensorDataset(X_test_tensor, Y_test_tensor)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)  
        if return_score:
            if score_type == "mahalanobis":
                average_loss, average_accuracy, score_sum, score_max = self.evaluate(test_loader, 
                                                                                 self.loss_function, 
                                                                                 score = score_type, 
                                                                                 return_score = return_score,
                                                                                 data = [X_train, X_test, Y_train, Y_test])                
            else:
                average_loss, average_accuracy, score_sum, score_max = self.evaluate(test_loader, 
                                                                                 self.loss_function, 
                                                                                 score = score_type, 
                                                                                 return_score = return_score)
            self.logger.info(f'Test Loss: {average_loss:.4f}, Test Accuracy: {average_accuracy:.4f}')
            return score_sum, score_max
        else:
            average_loss, average_accuracy = self.evaluate(test_loader, self.loss_function)
            return None, None


    def evaluate(self, data_loader, loss_function, score = "energy", return_score = False, data = None):
        """
        Evaluate the model
        """
        self.model.eval()  
        total_loss = 0
        total_accuracy = 0
        total_samples = 0

        score_sum = []
        score_max = []
        features_mahala_score = []
        
        for inputs, labels in data_loader:
            outputs = self.model(inputs)
            if score == "energy" and return_score:
                with torch.no_grad():
                    outputs_np = outputs.cpu().numpy()
                outputs_energy = np.log(1+outputs_np/(1.0000001-outputs_np))
                score_sum.append(outputs_energy.sum(axis = 1))
                score_max.append(outputs_energy.max(axis = 1))
            if score == "logits" and return_score:
                with torch.no_grad():
                    outputs_prob = outputs.cpu().numpy()
                outputs_logits = np.log(outputs_prob/(1.0000001-outputs_prob))
                score_sum.append(outputs_logits.sum(axis = 1))
                score_max.append(outputs_logits.max(axis = 1))
            if score == "msp" and return_score:
                with torch.no_grad():
                    outputs_prob = outputs.cpu().numpy()
                outputs_logits = np.log(outputs_prob/(1.0000001-outputs_prob))
                outputs_softmax = np.exp(outputs_logits)/np.exp(outputs_logits).sum(axis = 1)[:, None]
                score_max.append(outputs_softmax.max(axis = 1))
            if score == "prob" and return_score:
                with torch.no_grad():
                    outputs_prob = outputs.cpu().numpy()
                score_max.append(outputs_prob.max(axis = 1))
                score_sum.append(outputs_prob.sum(axis = 1))
            if score == "odin" and return_score:
                outputs_prob = self.odin_score(inputs, 0.001, 1)
                outputs_prob = outputs_prob.detach().numpy()
                score_max.append(outputs_prob.max(axis = 1))
                score_sum.append(outputs_prob.sum(axis = 1))
            

            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            predictions = outputs > 0.5 
            total_accuracy += (predictions == labels.byte()).all(dim=1).float().mean().item()
            total_samples += 1
        
        if score == "mahalanobis" and return_score:
            X_train = data[0]
            X_test = data[1]
            Y_train = data[2]
            Y_test = data[3]
            output_score = self.mahalanobis_score(X_train, X_test, Y_train, Y_test)
            score_max.append(-output_score.min(axis = 1))
            score_sum.append(-output_score.sum(axis = 1))
            
        
        average_loss = total_loss / total_samples
        average_accuracy = total_accuracy / total_samples

        if return_score and score in ["energy","logits","prob", "odin", "mahalanobis"]:
            score_sum = np.concatenate(score_sum)
            score_max = np.concatenate(score_max)
            return average_loss, average_accuracy, score_sum, score_max
        elif return_score and score in ["msp"]:
            score_max = np.concatenate(score_max)
            return average_loss, average_accuracy, None, score_max
        else:
            return average_loss, average_accuracy
    

    def odin_score(self, input, epsilon, temperature):

        input.requires_grad = True
        outputs = self.model(input)
        outputs = torch.log(outputs/(1.0000001-outputs))
        outputs = F.sigmoid(outputs / temperature)
        labels = torch.round(outputs)
        loss = nn.BCELoss()(outputs, labels)
        loss.backward()
        perturbation = epsilon * input.grad.sign()
        perturbed_input = input + perturbation
        outputs = self.model(perturbed_input)
        outputs = torch.log(outputs/(1.0000001-outputs))
        outputs = F.sigmoid(outputs / temperature)
        self.model.zero_grad()
        return outputs
    

    def mahalanobis_score(self, X_train, X_test, Y_train, Y_test):
        """
        Calculate the Mahalanobis score
        """

        num_classes = Y_train.shape[1]
        num_features = X_train.shape[1]
        
        mean_list = np.zeros((num_classes, num_features))
        cov_list = np.zeros((num_classes, num_features, num_features))
        
        for i in range(num_classes):
            X_train_class = X_train[Y_train[:, i] == 1]
            mean_list[i] = X_train_class.mean(axis=0)
            cov_list[i] = np.cov(X_train_class, rowvar=False)
            
        inv_cov_list = np.linalg.inv(cov_list + np.eye(num_features) * 1e-6) 

        diff = X_test[:, np.newaxis, :] - mean_list 
        mahalanobis_distances = np.einsum('ijk,jkl,ijl->ij', diff, inv_cov_list, diff)
        
        return mahalanobis_distances