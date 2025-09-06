import argparse, json, time, os
import torch
from sklearn.metrics import classification_report
import pickle

# --- Models/utilities from your package
from nesy_factory.GNNs import create_model, RGCN, PinSAGE, create_boxe, BoxE, create_pinsage, GCN, create_gcn
from nesy_factory.utils import get_config_by_name, train_model_with_config
# --- PyG datasets
from torch_geometric.datasets import FB15k_237, EllipticBitcoinDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help="Model type: rgcn, tgcn, gat, gcn, ...")
parser.add_argument('--process_data_url', type=str, required=True)
parser.add_argument('--weights_url', type=str, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--hidden_dim', type=int, required=True)
parser.add_argument('--dropout', type=float, required=True)
args = parser.parse_args()

import requests, pickle, io
import numpy as np


class DataWrapper:
    def __init__(self, data_dict):
        self.__dict__.update(data_dict)

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()

# --- Download & Load ---
dataset_url = args.process_data_url.replace("_$_V1", "_$$_V1")
print(dataset_url)
resp = requests.get(dataset_url)
resp.raise_for_status()

data = pickle.load(io.BytesIO(resp.content))


weights_url = args.weights_url.replace("_$_V1", "_$$_V1")
print(weights_url)
# fetch the file
resp = requests.get(weights_url)
resp.raise_for_status()
# parse JSON
weights = json.loads(resp.text)

def ensure_edge_splits(data, train_ratio=0.8, valid_ratio=0.1):
    if not hasattr(data, "train_edge_index"):
        num_edges = data.edge_index.size(1)

        train_size = int(train_ratio * num_edges)
        valid_size = int(valid_ratio * num_edges)
        test_size = num_edges - train_size - valid_size

        perm = torch.randperm(num_edges)
        train_idx = perm[:train_size]
        valid_idx = perm[train_size:train_size+valid_size]
        test_idx = perm[train_size+valid_size:]

        # Train edges
        data.train_edge_index = data.edge_index[:, train_idx]
        data.train_edge_type = data.edge_type[train_idx]

        # Validation edges (match nesy_factory naming: 'valid_*')
        data.valid_edge_index = data.edge_index[:, valid_idx]
        data.valid_edge_type = data.edge_type[valid_idx]

        # Test edges
        data.test_edge_index = data.edge_index[:, test_idx]
        data.test_edge_type = data.edge_type[test_idx]

    return data

if args.model_name.lower() == 'rgcn':
    dataset = FB15k_237(root='/tmp/fb15k237')
    data = dataset[0]
    data = ensure_edge_splits(data)

    config=get_config_by_name("basic_rgcn", "configs/rgcn_configs.yaml")
    # config = {
    # "input_dim": 1,
    # "output_dim": 1,
    # **config
    # }

    # config["hidden_dim"]=args.hidden_dim
    # config["learning_rate"]=args.learning_rate
    # config["dropout"]=args.dropout
    config = {
        # BaseGNN may expect these even if unused by RGCN
        "input_dim": 1, "output_dim": 1,
        "num_nodes": 14541,
        "num_relations": 237,
        "embed_dim": 64,
        "hidden_dim": args.hidden_dim,
        "num_layers": 2,
        "dropout": args.dropout,
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-4,
        "epochs": 3,
        "neg_samples": 1,
        "task_type": "link_prediction",
    }

    # Create model
    config = dict(config)
    model = RGCN(config)

    # Training
    loss = model.train_step(data, neg_samples=config["neg_samples"])
    print("Train loss:", loss)

    # Evaluation
    val_metrics = model.eval_step(data)
    print("Validation metrics:", val_metrics)
    
    # Dummy value for global_step
    global_step = 1
    trial_id = "0"  # Can also pass via env var or leave as default

    timestamp = time.time()

    metrics = [
        {
            "accuracy": f"{val_metrics['hit_rate']:.4f}",
            "checkpoint_path": "",
            "global_step": str(global_step),
            "timestamp": timestamp,
            "trial": trial_id
        }
    ]

    with open('/katib/mnist.json', "a", encoding="utf-8") as f:
        for metric in metrics:
            json.dump(metric, f)
            f.write("\n")


elif args.model_name.lower() == 'tgcn':

    # dataset = EllipticBitcoinDataset(root='/tmp/elliptic')
    # data = dataset[0]
    # config=get_config_by_name("basic_tgcn", "configs/tgcn_configs.yaml")

    config={"input_dim": 165,   "output_dim": 3,   "hidden_dim": args.hidden_dim,   "num_layers": 2,   "dropout": args.dropout,   "optimizer": "adam",   "learning_rate": args.learning_rate,   "weight_decay": 0.0,   "epochs": 2,   "task_type": "classification",   "use_focal_loss": True,   "focal_loss_gamma": 1.0,   "use_class_weights": False,   "ignore_index": 2 }
    # weight={"focal_loss_alpha": [0.11580919248009634, 0.8841908075199036, 0.0],   "class_weights": [0.5654888014527845, 4.317446562680532, 0.0] }
    config['focal_loss_alpha'] = weights['focal_loss_alpha']
    config['class_weights'] = weights['class_weights']
    
    model = create_model(args.model_name, config)
    
    epochs=config["epochs"]
    for epoch in range(epochs):
        loss = model.train_step(data, data.train_mask)
        if epoch % 2 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
    print("Finished Model Training ")


    eval_metrics = model.eval_step(data, data.test_mask)
    print(f"Test Accuracy: {eval_metrics['accuracy']:.4f}")

    model.eval()

    with torch.no_grad():
        out = model.predict(data)
        pred = out.argmax(dim=1)
        
        test_pred = pred[data.test_mask].cpu().numpy()
        test_true = data.y[data.test_mask].cpu().numpy()
        
        known_mask = test_true != 2
        test_pred_known = test_pred[known_mask]
        test_true_known = test_true[known_mask]
        
        print("\\nClassification Report:")
        report = classification_report(test_true_known, test_pred_known, 
                                        target_names=['Licit', 'Illicit'], 
                                        labels=[0, 1], digits=4, output_dict=True)
        print(classification_report(test_true_known, test_pred_known, 
                                    target_names=['Licit', 'Illicit'], 
                                    labels=[0, 1], digits=4))
            
        # Prepare metrics output
        metrics_output = {
            'test_accuracy': eval_metrics['accuracy'],
            'classification_report': report,
            'test_predictions': test_pred_known.tolist(),
            'test_true_labels': test_true_known.tolist()
        }

        # Dummy value for global_step
        global_step = 1
        trial_id = "0"  # Can also pass via env var or leave as default

        timestamp = time.time()

        metrics = [
            {
                "accuracy": f"{metrics_output['test_accuracy']:.4f}",
                "checkpoint_path": "",
                "global_step": str(global_step),
                "timestamp": timestamp,
                "trial": trial_id
            }
        ]

        with open('/katib/mnist.json', "a", encoding="utf-8") as f:
            for metric in metrics:
                json.dump(metric, f)
                f.write("\n")
                

elif args.model_name.lower() == 'pinsage':
    
    from torch_geometric.datasets import MovieLens

    dataset = MovieLens(root="/tmp/MovieLens")
    data = dataset[0]
    

    print("Movie features:", data['movie'].x.shape)  # should be [num_movies, 128]
    print("Users:", data['user'].num_nodes)
    print("Edges:", data['user', 'rates', 'movie'].edge_index.shape)

    num_movies = data['movie'].num_nodes
    train_mask = torch.zeros(num_movies, dtype=torch.bool)
    val_mask = torch.zeros(num_movies, dtype=torch.bool)
    test_mask = torch.zeros(num_movies, dtype=torch.bool)

    train_mask[:int(0.6 * num_movies)] = True
    val_mask[int(0.6 * num_movies):int(0.8 * num_movies)] = True
    test_mask[int(0.8 * num_movies):] = True

    data['movie'].train_mask = train_mask
    data['movie'].val_mask = val_mask
    data['movie'].test_mask = test_mask

    
    config=get_config_by_name('basic_pinsage', 'configs/pinsage_configs.yaml')
    print(config)

    config = {
    "input_dim": 128,
    "hidden_dim": [args.hidden_dim, args.hidden_dim, args.hidden_dim], 
    "output_dim": 32,
    "num_layers": 3,              # since hidden_dim has 3 layers
    "dropout": args.dropout,               # you can keep args.dropout if you want to tune it
    "optimizer": "adam",
    "learning_rate": args.learning_rate,        # or args.learning_rate
    "weight_decay": 5e-4,
    "epochs": 1,                 # usually >1 for MovieLens
    "task_type": "node_classification",
    "num_samples": 12,
    "walk_length": 4,
    "num_walks": 50,
    }


    # Create model
    model = PinSAGE(config)

    # Training
    loss = model.train_step(data, mask=data['movie'].train_mask)
    print("Train loss:", loss)

    # Evaluation
    val_metrics = model.eval_step(data, mask=data['movie'].val_mask)
    print("Validation metrics:", val_metrics)

    # Prediction (optional)
    preds = model.predict(data, batch_nodes=None)
    print("Predictions shape:", preds.shape)

elif args.model_name.lower() == 'boxe':
    
    config=get_config_by_name('default_boxe', 'configs/boxe_configs.yaml')
    print(config)

    # Build config (taken from your YAML)
    config = {
        "dataset": "wn18rr",  # PyKEEN dataset name
        "input_dim": 0,       # not used, but required for BaseGNN
        "output_dim": 0,
        "hidden_dim": args.hidden_dim,
        "model_kwargs": {"embedding_dim": 2},
        "optimizer": "Adam",
        "optimizer_kwargs": {"lr": args.learning_rate},
        "training_kwargs": {"num_epochs": 1, "batch_size": 128},  # reduced for quick test
        "random_seed": 42,
        "device": "cuda",  # change to "cuda" if GPU available
        "dropout": args.dropout,
    }

    # Create BoxE model wrapper
    model = BoxE(config)

    # --- Training ---
    print("Training BoxE model with PyKEEN...")
    model.train_model()  # runs the PyKEEN pipeline internally
    print("Training completed")

    # --- Access trained embeddings ---
    embeddings = model.forward()
    print("Entity embeddings shape:", embeddings.shape)

    # --- Evaluation ---
    print("Evaluation metrics:")
    print(model.pykeen_result.metric_results.to_dict())
    mrr = model.pykeen_result.metric_results.get_metric("mrr")
    print(f"Final MRR={mrr}")


    # --- Example triple scoring ---
    # Pick random head/relation/tail IDs
    h = torch.tensor([0, 1, 2])  # head entity IDs
    r = torch.tensor([0, 1, 2])  # relation IDs
    t = torch.tensor([1, 2, 3])  # tail entity IDs

    scores = model.decode(h, r, t)
    print("Triple scores:", scores)
    
    # Dummy value for global_step
    global_step = 1
    trial_id = "0"  # Can also pass via env var or leave as default

    timestamp = time.time()

    metrics = [
        {
            "accuracy": f"{mrr:.4f}",
            "checkpoint_path": "",
            "global_step": str(global_step),
            "timestamp": timestamp,
            "trial": trial_id
        }
    ]

    with open('/katib/mnist.json', "a", encoding="utf-8") as f:
        for metric in metrics:
            json.dump(metric, f)
            f.write("\n")
    
    

elif args.model_name.lower() == 'gcn':
    
    config=get_config_by_name('basic_gcn', 'configs/gcn_configs.yaml')
    from torch_geometric.datasets import Planetoid
    from models import create_gcn
    import torch.optim as optim

    # 1. Load dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]  # graph data

    # 2. Config based on dataset
    config = {
        "input_dim": dataset.num_features,
        "hidden_dim": 64,
        "output_dim": dataset.num_classes,
        "num_layers": 2,
        "dropout": 0.5,
        "normalize": True,
        "add_self_loops": True,
        "bias": True,
    }

    # 3. Create model
    model = create_gcn(config)

    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)   # forward
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(model, data, mask):
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        return acc
    
    for epoch in range(1, 201):  # 200 epochs
        loss = train(model, data, optimizer)
        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")



