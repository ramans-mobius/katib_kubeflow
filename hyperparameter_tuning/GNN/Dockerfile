FROM nikhilv215/nesy-factory:v2

WORKDIR /app
COPY train_model.py .
RUN pip install --no-cache-dir \
    tqdm \
    kubernetes\
    kubeflow-katib
# RUN pip install dgl==1.1.1 -f https://data.dgl.ai/wheels/torch-2.0/repo.html
# RUN pip install --no-cache-dir pykeen==1.10.1 class-resolver==0.3.10

# RUN pip install sentence-transformers

RUN pip install --no-cache-dir pandas

# If extra deps needed:
# RUN pip install torch-geometric ...

ENTRYPOINT ["python", "train_model.py"]
