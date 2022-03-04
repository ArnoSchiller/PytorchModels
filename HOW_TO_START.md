# Use torchserve to serve a model in docker
Source: https://towardsdatascience.com/deploy-models-and-create-custom-handlers-in-torchserve-fc2d048fbe91

## Prepare system
I am using the Lunix Server from SWMS via Putty. Create the directory:
```bash
mkdir torchserve_getting_started
cd torchserve_getting_started
```

## Setup the environment 
Check [the docs](https://github.com/pytorch/serve/tree/v0.5.3) to enable GPU support.
```bash
python -m venv torchserve
source torchserve/bin/activate

## maybe u need to install wheel first 
# pip install wheel
git clone https://github.com/pytorch/serve.git
cd serve
python ./ts_scripts/install_dependencies.py
cd model-archiver
pip install .
cd ../..
```

## Export a model 
Torchserve expects a .mar file to be provided. In a nutshell, the file is just your model and all the dependencies packed together. To create one need to first export our trained model.

### Export the model using trace
There are three ways to export your model for torchserve. The best way that I have found so far is to trace the model and store the results. By doing so we do not need to add any additional files to torchserve.

Copy this code to the file ```export.py``` and run the file ```python export.py``` (Using python 3.8.10)
```python
import torch
from torchvision.models import resnet34

model = resnet34(pretrained=True)
example_input = torch.rand(1, 3, 224, 224)
# following issue https://github.com/pytorch/serve/issues/364 you need to explicitly set the model to .eval
model.eval()
traced_model = torch.jit.trace(model, example_input)
traced_model.save("./resnet34.pt")
```
Now there should be a file named ```resnet34.pt```.
Also make sure to create a file called ```index_to_name.json``` containing the mapping of the Resnet Detector. You can find the content of the file [here](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/torchserve-tryout/master/index_to_name.json)
### Archive the model 
```bash 
mkdir model-store
torch-model-archiver --model-name resnet34 --version 1.0 --serialized-file resnet34.pt --extra-files ./index_to_name.json,./MyHandler.py --handler my_handler.py  --export-path model-store -f
ls model-store
```
In this directory there should be the archived model named ```resnet34.mar```.

## Serve the model using docker 
```bash
# download the image 
docker pull pytorch/torchserve:0.5.3-cpu
# run the container
docker run --rm -it -p 3000:8080 -p 3001:8081 -v $(pwd)/model-store:/home/model-server/model-store pytorch/torchserve:latest torchserve --start --model-store model-store --models resnet34=resnet34.mar
```