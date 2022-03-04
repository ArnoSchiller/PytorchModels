FROM pytorch/torchserve:0.5.3-cpu

# Copy needed files 
COPY ./my_handler.py /home/model-server/my_handler.py 
COPY ./MyHandler.py /home/model-server/MyHandler.py 
COPY ./export.py /home/model-server/export.py 
COPY ./export_model.sh /home/model-server/export_model.sh 

# Download the class mapping
RUN curl -O https://raw.githubusercontent.com/FrancescoSaverioZuppichini/torchserve-tryout/master/index_to_name.json

# Export and archive the model
USER root
RUN pip install torch-model-archiver -q
RUN chmod +x /home/model-server/export_model.sh 
RUN /home/model-server/export_model.sh
EXPOSE 8080 8081
USER model-server
