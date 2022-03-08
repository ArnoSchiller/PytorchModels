import json
import logging
from pathlib import Path
from types import SimpleNamespace

from autoencoder import PytorchAutoencoder
from lstm_autoencoder import PytorchLSTMAutoencoder

logging.basicConfig(level=logging.DEBUG)

class ModelHandler():

    model_info = None

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_info_path = Path(model_dir, "model_info.json")
        self.load_model_info()

        if self.model_info.framework.lower() != "pytorch": 
            logging.warning("You are trying to load a model with a different framework than pytorch")
            return 
        
        if self.model_info.type.lower() == "autoencoder":
            self.model = PytorchLSTMAutoencoder(self.model_info)
        elif self.model_info.type.lower() == "objectdetector":
            self.model = PytorchObjectDetector(self.model_info)
        self.model_info.status = "created"
        

    def build_model(self, build_params):
        if self.model_info.status != "created":
            logging.error("You can only build created models")
            return

        if self.model.build_model(build_params):
            self.model_info.status = "build"
        print("model_info.status" + self.model_info.status)

        self.model_info.summary = self.model.summary()

    def train_model(self, training_loader, validation_loader, train_params):
        if self.model_info.status != "build":
            logging.error("You can only train build models")
            return 

        self.model.train_model(training_loader, validation_loader, train_params)
        self.model_info.status = "trained"


    def load_model_info(self):
        if not self.model_info_path.is_file(): 
            logging.error("Path does not exist")
            return None

        with open(self.model_info_path, 'r') as f:
            self.model_info = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))
            
    def save_model_info(self):
        with open(self.model_info_path, 'w') as f:
            f.write(json.dumps(self.model_info.__dict__))







class PytorchObjectDetector():
    def __init__(self, model_info):
        self.info = model_info




if __name__ == '__main__':
    modelHandler = ModelHandler("test_model") 

    import torch
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=2)

    json_params = {
        "sequence_length": 3136,
        "num_features": 1,
        "hidden_layer_size": 128
    }

    modelHandler.build_model(json_params)

    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    json_train_params = {
        "num_epochs": 1
    }

    print(modelHandler.model_info.summary)
    # modelHandler.train_model(training_loader, validation_loader, json_train_params)
