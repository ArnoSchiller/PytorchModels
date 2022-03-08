
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import logging

import torch

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

learning_rate = 1e-3
reduction = 'sum'


class TrainParams(BaseModel):
    num_epochs: int

class PytorchModel():
    
    _model = None
    _device = None
    _writer = None

    loss_fn = None
    optimizer = None

    _loss_fn = None
    _optimizer = None

    _build_params = None
    _train_params = None

    _input_size = None
    
    _training_loader = None
    _validation_loader = None

    def __init__(self, model_info):
        self.info = model_info

        self.model_name = self.info.name
        self.model_path = self.info.dir_path
        self.train_path = Path(self.info.dir_path, "train")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self.device)
        

    def build_model(self):
        if not self.validate_build_params():
            return False
        
        logging.info("Building model ...")
        #  use gpu if available
        self._model.to(self._device)

        self._optimizer = self.optimizer(self._model.parameters(), lr=learning_rate)
        self._loss_fn = self.loss_fn(reduction=reduction).to(self._device)

        logging.info("Building model done")
        return True

    def train_model(self, training_loader, val_dataloader, train_params):

        self.update_train_params(train_params)

        logging.info("Training model ...")
        self._training_loader = training_loader
        self._validation_loader = val_dataloader

        return self._train_model()

    def _train_model(self):
        self.validate_training_params()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(Path(self.train_path, "{}_{}".format(self.model_name, timestamp)))

        for epoch in range(self._train_params.num_epochs):
            logging.info('EPOCH {}:'.format(epoch + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self._model.train(True)
            avg_loss = self.train_one_epoch(epoch)

            # We don't need gradients on to do reporting
            self._model.train(False)


            running_vloss = 0.0
            for i, vdata in enumerate(self._validation_loader):
                vinputs, vground_truths = vdata
                voutputs = self._model(vinputs)
                vloss = self._loss_fn(voutputs, vground_truths)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            logging.info('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            self.writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch + 1)
            self.writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = Path(self.model_path, 'model_{}_{}'.format(timestamp, epoch))
                torch.save(self._model.state_dict(), model_path)

        logging.info("Training model done")
        return True

    def train_one_epoch(self, epoch_index):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self._training_loader):
            # Every data instance is an input + label pair
            inputs, ground_truths = data

            # Zero your gradients for every batch!
            self._optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self._model(inputs)

            # Compute the loss and its gradients
            loss = self._loss_fn(outputs, ground_truths)
            loss.backward()

            # Adjust learning weights
            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                logging.info('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.training_loader) + i + 1
                self._writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def summary(self):
        print( self._model)
        return #summary(self._model, (1,6272))# , batch_size=1, device=self.device) 

    def update_train_params(self, train_params):
        pass

    def validate_training_params(self):
        if not self.validate_build_params():
            return False

        if self._training_loader is None or self._validation_loader is None: 
            logging.error("Initialise your dataloader! Retry again.")
            return False
            
        return True

        
    def update_build_params(self, build_params):
        pass

    def validate_build_params(self):
        if self._build_params is None: 
            logging.error("Initialise your build params! Retry again.")
            return False

        if self._model is None: 
            logging.error("Initialise your model first! Retry again.")
            return False
            
        return True