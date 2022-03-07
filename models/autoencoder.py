import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

batch_size = 512
epochs = 20
learning_rate = 1e-3

class PytorchAutoencoder():
    def __init__(self, model_info):
        self.info = model_info
        
    def build_model(self):
        #  use gpu if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create a model from `AE` autoencoder class
        # load it to the specified device, either gpu or cpu
        self.model = AE(input_shape=784).to(self.device)

        # create an optimizer object
        # Adam optimizer with learning rate 1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # mean-squared error loss
        self.criterion = nn.MSELoss()

    def load_dataset(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        train_dataset = torchvision.datasets.MNIST(
            root="~/torch_datasets", train=True, transform=transform, download=True
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        test_dataset = torchvision.datasets.MNIST(
            root="~/torch_datasets", train=False, transform=transform, download=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=10, shuffle=False
        )
    

    def train_model(self):
        for epoch in range(epochs):
            loss = 0
            for batch_features, _ in self.train_loader:
                # reshape mini-batch data to [N, 784] matrix
                # load it to the active device
                batch_features = batch_features.view(-1, 784).to(self.device)
                
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                self.optimizer.zero_grad()
                
                # compute reconstructions
                outputs = self.model(batch_features)
                
                # compute training reconstruction loss
                train_loss = self.criterion(outputs, batch_features)
                
                # compute accumulated gradients
                train_loss.backward()
                
                # perform parameter update based on current gradients
                self.optimizer.step()
                
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
            
            # compute the epoch training loss
            loss = loss / len(self.train_loader)
            
            # display the epoch training loss
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

    def validate_model(self):
        test_examples = None

        with torch.no_grad():
            for batch_features in self.test_loader:
                batch_features = batch_features[0]
                test_examples = batch_features.view(-1, 784)
                reconstruction = self.model(test_examples)
                break

        with torch.no_grad():
            number = 10
            plt.figure(figsize=(20, 4))
            for index in range(number):
                # display original
                ax = plt.subplot(2, number, index + 1)
                plt.imshow(test_examples[index].numpy().reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display reconstruction
                ax = plt.subplot(2, number, index + 1 + number)
                plt.imshow(reconstruction[index].numpy().reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()

    def save_model(self):
        pass



class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed



if __name__ == "__main__":
    model = PytorchAutoencoder("")
    model.build_model()
    model.load_dataset()
    model.train_model()
    model.validate_model()