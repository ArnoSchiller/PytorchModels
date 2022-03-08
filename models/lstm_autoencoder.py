import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
from pytorch_model import TrainParams
from pytorch_model import PytorchModel

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from arff2pandas import a2p # pip install arff2pandas

from pydantic import BaseModel

batch_size = 1
epochs = 20
learning_rate = 1e-3

seq_len = 60
n_features = 2

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class AutoencoderBuildParams(BaseModel):
    sequence_length: int
    num_features: int
    hidden_layer_size: int

class PytorchLSTMAutoencoder(PytorchModel):

  
    def __init__(self, model_info):
        super().__init__(model_info)

        # create an optimizer object
        # Adam optimizer with learning rate 1e-3
        self.optimizer = torch.optim.Adam
        # mean-squared error loss
        self.loss_fn = nn.L1Loss
        
    def build_model(self, build_params):
        if not self.update_build_params(build_params):
            return False
        self._model = RecurrentAutoencoder( self._build_params.sequence_length, 
                                            self._build_params.num_features, 
                                            self._build_params.hidden_layer_size)
        return super().build_model()


    def update_build_params(self, build_params):
        try: 
            self._build_params = AutoencoderBuildParams(**build_params)
            self._input_size = (1, self._build_params.sequence_length, self._build_params.num_features)

            return True
        except Exception as e:
            logging.error("Failed to parse AutoencoderBuildParams. Details: " + e.trace())
            return False

    def update_train_params(self, train_params):
        try: 
            self._train_params = TrainParams(**train_params)
        except Exception as e:
            logging.error("Failed to parse TrainParams. Details: " + e.trace())



    def load_dataset(self):
        pass
    

    #def train_model(self, train_dataset, val_dataset, n_epochs):
        """
        history = dict(train=[], val=[])
        best_model_wts = copy.deepcopy(self._model.state_dict())
        best_loss = 10000.0

        for epoch in range(1, n_epochs + 1):
            self._model = self._model.train()
            train_losses = []

            for seq_true in train_dataset:
                self._optimizer.zero_grad()
                seq_true = seq_true.to(self.device)
                seq_pred = self._model(seq_true)
                loss = self._loss_fn(seq_pred, seq_true)
                loss.backward()
                self._optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            self._model = self._model.eval()

            with torch.no_grad():
                for seq_true in val_dataset:
                    seq_true = seq_true.to(self.device)
                    seq_pred = self._model(seq_true)
                    loss = self._loss_fn(seq_pred, seq_true)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self._model.state_dict())

            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        self._model.load_state_dict(best_model_wts)
        return self._model.eval(), history
        """

    def validate_model(self):
        pass

    def save_model(self):
        pass



class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):

    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim)
    self.decoder = Decoder(seq_len, embedding_dim, n_features)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):

    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features

    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):

    x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((self.n_features, self.embedding_dim))

class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):

    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim

    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):

    x = x.repeat(self.seq_len, self.n_features)

    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)

    x, (hidden_n, cell_n) = self.rnn2(x)

    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)


def create_dataset(df):
  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features



"""
if __name__ == '__main__':

    with open('data/ECG5000_TRAIN.arff') as f:
        train = a2p.load(f)
    with open('data/ECG5000_TEST.arff') as f:
        test = a2p.load(f)

    df = train.append(test)
    df = df.sample(frac=1.0)
    df.shape

    CLASS_NORMAL = 1
    class_names = ['Normal','R on T','PVC','SP','UB']
 
    new_columns = list(df.columns)
    new_columns[-1] = 'target'
    df.columns = new_columns

    normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)
    print(normal_df.shape)
    anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)
    print(anomaly_df.shape)

    train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(val_df, test_size=0.33, random_state=RANDOM_SEED)

    train_dataset, seq_len, n_features = create_dataset(train_df)
    val_dataset, _, _ = create_dataset(val_df)
    test_normal_dataset, _, _ = create_dataset(test_df)
    test_anomaly_dataset, _, _ = create_dataset(anomaly_df)



    ### 
        

    model = PytorchLSTMAutoencoder("")
    print("Starting build")
    model.build_model()
    model.train_model(
        training_loader,
        val_dataset,
        validation_loader=1)
"""