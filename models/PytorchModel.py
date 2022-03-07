import json
import logging
from pathlib import Path
from types import SimpleNamespace

from autoencoder import PytorchAutoencoder

class ModelHandler():

    model_info = None

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_info_path = Path(model_dir, "model_info.json")
        self.load_model_info()

        if self.model_info.framework.lower() != "pytorch": 
            logging.warning("You are trying to load a model with a different framework than pytorch")
            return False
        
        if self.model_info.type.lower() == "autoencoder":
            self.model = PytorchAutoencoder(self.model_info)
        elif self.model_info.type.lower() == "objectdetector":
            self.model = PytorchObjectDetector(self.model_info)

        return True


    def build_model(self):
        if self.model_info.status != "created":
            logging.error("You can only build created models")
            return 

        self.model.build_model()
        self.model_info.status = "build"

    def train_model(self):
        if self.model_info.status != "build":
            logging.error("You can only train build models")
            return 

        self.model.build_model()
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
    model = PytorchModel("test_model")
    print(model.info)


