#This code has been tested only for urls with stgnn and default datasets for stgnn, gcn, rgcn, tgcn, boxe and pinsage
import argparse, json, time, os, pickle, io, yaml, requests, torch, urllib.parse
from sklearn.metrics import classification_report
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import Data
import random
from nesy_factory.GNNs import create_model, RGCN, STGNN, PinSAGE, BoxE, create_gcn
from nesy_factory.utils import get_config_by_name
from torch_geometric.datasets import FB15k_237, Planetoid
import base64
import sys

# -------------------------------
# Helpers
# -------------------------------
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
        self.xs, self.ys = self.xs[permutation], self.ys[permutation]

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()

class ImportanceBasedSampler:
    """Importance-based neighbor sampler for PinSAGE"""

    def __init__(self, edge_index, num_nodes, walk_length=5, num_walks=50):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
        self.importance_scores = self._compute_importance_scores()

    def _random_walk(self, start_node, length):
        if start_node not in self.G: 
            return [start_node]
        
        walk = [start_node]
        current = start_node

        for _ in range(length - 1):
            neighbors = list(self.G.neighbors(current))
            if not neighbors: 
                break
            current = random.choice(neighbors)
            walk.append(current)

        return walk

    def _compute_importance_scores(self):
        importance = {}
        for node in range(self.num_nodes):
            visit_counts = {}
            for _ in range(self.num_walks):
                walk = self._random_walk(node, self.walk_length)
                for visited_node in walk[1:]:
                    visit_counts[visited_node] = visit_counts.get(visited_node, 0) + 1
            total_visits = sum(visit_counts.values())
            node_importance = {k: v / total_visits for k, v in visit_counts.items()} if total_visits > 0 else {}
            importance[node] = node_importance
        return importance

    def sample_neighbors(self, nodes, num_samples=10):
        sampled_neighbors = {}
        importance_weights = {}
        for node in nodes:
            node = int(node)
            neighbors_scores = self.importance_scores.get(node, {})
            if neighbors_scores:
                sorted_neighbors = sorted(neighbors_scores.items(), key=lambda x: x[1], reverse=True)
                selected = sorted_neighbors[:min(num_samples, len(sorted_neighbors))]
                if selected:
                    neighbors, weights = zip(*selected)
                    sampled_neighbors[node] = list(neighbors)
                    importance_weights[node] = list(weights)
                else:
                    sampled_neighbors[node] = []
                    importance_weights[node] = []
            else:
                sampled_neighbors[node] = []
                importance_weights[node] = []
        return sampled_neighbors, importance_weights

def normalize_url(url: str) -> str:
    url = urllib.parse.unquote(url)
    url = url.replace("_$_", "_$$").replace("_-", "_$$").replace("__DOLLARS__", "$$")
    return url

def load_pickle_url(url: str):
    print(f"Downloading pickle dataset from {url} ...")
    resp = requests.get(url)
    resp.raise_for_status()
    return pickle.load(io.BytesIO(resp.content))

def load_json_url(url: str):
    print(f"Downloading JSON weights/config from {url} ...")
    resp = requests.get(url)
    resp.raise_for_status()
    return json.loads(resp.text)

def save_metrics(metrics: dict):
    """Enhanced metrics saving with proper error handling and exit codes"""
    global_step = 1
    trial_id = "0"
    timestamp = time.time()
    
    print("=== SAVING METRICS ===")
    print(f"Input metrics: {metrics}")
    
    # Determine which metric to save
    if "accuracy" in metrics:
        metric_name = "accuracy"
        metric_value = f"{metrics['accuracy']:.4f}"
    elif "accuracy_percentage" in metrics:
        metric_name = "accuracy"
        metric_value = f"{metrics['accuracy_percentage']:.4f}"
    elif "mrr" in metrics:
        metric_name = "mrr"  # Katib expects 'accuracy' as objective
        metric_value = f"{metrics['mrr']:.4f}"
    elif "hit_rate" in metrics:
        metric_name = "hit_rate"
        metric_value = f"{metrics['hit_rate']:.4f}"
    else:
        # Pick the first remaining metric
        remaining_metrics = {k: v for k, v in metrics.items() if k not in ["checkpoint_path"]}
        if remaining_metrics:
            metric_name, metric_value = next(iter(remaining_metrics.items()))
        else:
            print("WARNING: No valid metrics found, using default accuracy=0.0")
            metric_name = "No valid metric"
            metric_value = 0.0
    """
    # Ensure metric value is a valid number
    try:
        metric_value = float(metric_value)
        if np.isnan(metric_value) or np.isinf(metric_value):
            print(f"WARNING: Invalid metric value {metric_value}, setting to 0.0")
            metric_value = 0.0
    except (ValueError, TypeError):
        print(f"ERROR: Cannot convert metric value {metric_value} to float, setting to 0.0")
        metric_value = 0.0

    result = {"accuracy": metric_value}
    print(f"Final result to save: {result}")
    
    # Ensure directory exists
    metrics_dir = "/katib"
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = "/katib/mnist.json"    # mnist.json
    try:
        with open(metrics_file, "w") as f:
            json.dump(result, f)
        print(f"✓ Metrics saved successfully to {metrics_file}")
        
        # Verify file was written
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                saved_data = json.load(f)
            print(f"✓ Verified saved data: {saved_data}")
        else:
            print("ERROR: Metrics file was not created!")
            
    except Exception as e:
        print(f"ERROR saving metrics: {e}")
        # Don't fail the trial, but log the error
    """
    record = {
        metric_name: metric_value, 
        "checkpoint_path": "",
        "global_step": str(global_step),
        "timestamp": timestamp,
        "trial": trial_id,
    } 
    print("Record being saved:", record)
    with open("/katib/mnist.json", "a", encoding="utf-8") as f:  #I am changing this from appending to writing
        json.dump(record, f)
        f.write("\n")
        
    print("=== METRICS SAVING COMPLETE ===")

def ensure_edge_splits(data, train_ratio=0.8, valid_ratio=0.1):
    if not hasattr(data, "train_edge_index"):
        num_edges = data.edge_index.size(1)
        train_size = int(train_ratio * num_edges)
        valid_size = int(valid_ratio * num_edges)
        test_size = num_edges - train_size - valid_size
        perm = torch.randperm(num_edges)
        data.train_edge_index = data.edge_index[:, perm[:train_size]]
        data.train_edge_type = data.edge_type[perm[:train_size]]
        data.valid_edge_index = data.edge_index[:, perm[train_size:train_size+valid_size]]
        data.valid_edge_type = data.edge_type[perm[train_size:train_size+valid_size]]
        data.test_edge_index = data.edge_index[:, perm[train_size+valid_size:]]
        data.test_edge_type = data.edge_type[perm[train_size+valid_size:]]
    return data

# -------------------------------
# Default configs
# -------------------------------
DEFAULT_CONFIGS = {
    # Same as your previous DEFAULT_CONFIGS
    "rgcn": {"input_dim":1,"output_dim":1,"num_nodes":14541,"num_relations":237,"embed_dim":64,"hidden_dim":16,"num_layers":2,"dropout":0.1,"optimizer":"adam","learning_rate":0.001,"weight_decay":1e-4,"epochs":2,"task_type":"link_prediction","neg_samples":1},
    "tgcn": {"input_dim":165,"output_dim":3,"hidden_dim":16,"num_layers":2,"dropout":0.1,"optimizer":"adam","learning_rate":0.001,"epochs":2,"task_type":"classification"},
    "pinsage": {"input_dim":1433,"hidden_dim":[64,64],"output_dim":7,"num_layers":2,"dropout":0.5,"optimizer":"adam","learning_rate":0.01,"weight_decay":5e-4,"epochs":2,"task_type":"node_classification","num_samples":10,"walk_length":3,"num_walks":20,"optimizer_kwargs":{"lr":0.01,"weight_decay":5e-4}},
    "boxe": {"dataset":"wn18rr","dropout":0.1,"hidden_dim":16,"input_dim":0,"model_kwargs":{"embedding_dim":2},"optimizer":"Adam","learning_rate":0.001,"output_dim":0,"optimizer_kwargs":{"lr":0.001},"epochs":2,"task_type":"link_prediction","training_kwargs":{"num_epochs":30,"batch_size":128},"random_seed":42,"device":"cuda"},
    "stgnn": {"gcn_true":True,"buildA_true":True,"num_nodes":6,"hidden_dim":16,"dropout":0.1,"seq_out_len":1,"subgraph_size":6,"propalpha":0.1,"tanhalpha":20,"node_dim":256,"gcn_depth":2,"dilation_exponential":1,"conv_channels":16,"residual_channels":16,"skip_channels":32,"end_channels":64,"layers":2,"in_dim":1,"batch_size":64,"epochs":2,"learning_rate":0.0003,"weight_decay":0.0001,"seq_in_len":100,"error_percentage_threshold":0.1,"error_absolute_threshold":0.05,"optimizer":"adam","task_type":"regression","clip":10,"step_size1":2500},
    "gcn": {"input_dim":1433,"hidden_dim":64,"output_dim":7,"dropout":0.5,"learning_rate":0.01,"epochs":2,"optimizer":"adam","task_type":"classification","normalize":True,"add_self_loops":True,"bias":True},
}

# -------------------------------
# Config merging
# -------------------------------
def build_config(ModelName, unknown, process_data_url=None, weights_url=None, data=None, cfg=None):
    # 1. Load config file if available
    config_key = f"basic_{ModelName}"
    config_file = f"configs/{ModelName}_configs.yaml"
    file_config = {}
    try: 
        file_config = get_config_by_name(config_key, config_file) or {}
        print(f"[CONFIG FILE] Loaded {config_file}")
    except Exception as e: 
        print(f"[CONFIG FILE] not found/failed: {e}")
        file_config = {}

    merged = file_config.copy()
    source = {k:"config_file" for k in file_config}

    # 2. Apply hardwired defaults
    for k,v in DEFAULT_CONFIGS.get(ModelName,{}).items():
        merged[k] = v
        source[k] = "hardwired_defaults"

    # 3. Merge weights_url
    if weights_url:
        try:
            weights_dict = load_json_url(normalize_url(weights_url)) or {}
            for k,v in weights_dict.items():
                merged[k] = v
                source[k] = "weights_url"
        except Exception as e:
            print(f"[WEIGHTS] load failed: {e}")

    # 4. Merge cfg from --config_json (user JSON input)
    if cfg:
        try:
            weights_dict = load_json_url(normalize_url(weights_url)) or {}
            for k,v in cfg.items():
                merged[k] = v
                source[k] = "config_json"
        except Exception as e:
            print(f"[CONFIG] load failed: {e}")
        

    # 5. Apply CLI overrides (highest priority)
    cli_overrides = {}
    i=0
    while i < len(unknown):
        tok = unknown[i]
        if tok.startswith("--"):
            key = tok.lstrip("-")
            if "=" in key:
                key, sval = key.split("=",1)
                try: 
                    val = json.loads(sval)
                except: 
                    val = sval
                cli_overrides[key] = val
                i += 1
            else:
                if i+1 < len(unknown) and not unknown[i+1].startswith("--"):
                    sval = unknown[i+1]
                    try: 
                        val = json.loads(sval)
                    except: 
                        val = sval
                    cli_overrides[key] = val
                    i += 2
                else:
                    cli_overrides[key] = True
                    i += 1
        else: 
            i += 1

    for k,v in cli_overrides.items():
        merged[k] = v
        source[k] = "cli_override"

    # 6. Override input/output dims from dataset if available
    if data is not None:
        if hasattr(data,"num_features") and "input_dim" in merged:
            merged["input_dim"] = data.num_features
            source["input_dim"] = "dataset"
        if hasattr(data,"num_classes") and "output_dim" in merged:
            merged["output_dim"] = data.num_classes
            source["output_dim"] = "dataset"

    epochs = int(merged.get("epochs",1))

    print("\n[FINAL CONFIG]")
    for k in sorted(merged): 
        print(f"  {k}: {merged[k]} (from {source.get(k,'unknown')})")

    return merged, epochs

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--process_data_url", type=str, default=None)
        parser.add_argument("--weights_url", type=str, default=None)
        parser.add_argument("--config", type=str, required=True)

        args, unknown = parser.parse_known_args()

        # Treat --config as base64 string, not a file path
        try:
            fixed_config = base64.b64decode(args.config).decode("utf-8")
            config = json.loads(fixed_config)
        except Exception as e:
            raise ValueError(f"Failed to decode config: {e}")

        print("=== STARTING TRIAL ===")
        print("Input config (base64):", args.config[:50] + "..." if len(args.config) > 50 else args.config)
        print("Fixed Config:", fixed_config)
        print("=== Training Config Loaded ===")
        print(json.dumps(config, indent=2))

        cfg = config
        ModelName = args.model_name.lower()
        data = None

        print(f"Model: {ModelName}")
        print(f"Process data URL: {args.process_data_url}")
        print(f"Weights URL: {args.weights_url}")

        # Dataset loading
        if args.process_data_url:
            data = load_pickle_url(normalize_url(args.process_data_url))
            print(f"Dataset has been loaded from url: {normalize_url(args.process_data_url)}")
        else:
            print("Default Dataset has been used.")
            
        config, epochs = build_config(ModelName, unknown, args.process_data_url, args.weights_url, data, cfg)
        print(f"Training for {epochs} epochs")

        # Model-specific training blocks
        if ModelName == "pinsage":
            print("=== TRAINING PINSAGE MODEL ===")
            dataset = Planetoid(root="data/Cora", name="Cora", transform=ToUndirected()) if data is None else None
            data = dataset[0] if data is None else data
            
            config["input_dim"] = data.num_node_features
            config["output_dim"] = dataset.num_classes if dataset else config["output_dim"]

            model = PinSAGE(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            criterion = nn.CrossEntropyLoss()

            x, edge_index = data.x, data.edge_index
            train_mask, test_mask, y = data.train_mask, data.test_mask, data.y

            if hasattr(model,"forward") and "sampler" in model.forward.__code__.co_varnames:
                sampler = ImportanceBasedSampler(edge_index, x.size(0), walk_length=config.get("walk_length",3), num_walks=config.get("num_walks",20))
                def full_forward(x, edge_index): return model(x,sampler,torch.arange(x.size(0)))
                model.forward = full_forward

            for epoch in range(config["epochs"]):
                model.train()
                optimizer.zero_grad()
                out = model(x, edge_index)
                loss = criterion(out[train_mask], y[train_mask])
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch}, Loss={loss.item():.4f}")

            model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
                pred = out.argmax(dim=1)
                acc = (pred[test_mask]==y[test_mask]).sum().item()/test_mask.sum().item()
                print(f"Test Accuracy: {acc}")
            save_metrics({"accuracy": acc})

        elif ModelName == "rgcn":
            print("=== TRAINING RGCN MODEL ===")
            dataset = FB15k_237(root="/katib/fb15k237") if data is None else None
            data = dataset[0] if data is None else data
            data = ensure_edge_splits(data)

            model = RGCN(config)
            for e in range(epochs):
                loss = model.train_step(data, neg_samples=config.get("neg_samples",1))
                print(f"Epoch {e+1}/{epochs} Loss {loss}")

            eval_metrics = {}
            if hasattr(model,"eval_step"):
                eval_metrics = model.eval_step(data)
            
            if "accuracy" in eval_metrics: 
                save_metrics({"accuracy": eval_metrics["accuracy"]})
            elif "accuracy_percentage" in eval_metrics: 
                save_metrics({"accuracy": eval_metrics["accuracy_percentage"]})
            elif "hit_rate" in eval_metrics: 
                save_metrics({"accuracy": eval_metrics["hit_rate"]})
            else: 
                save_metrics({"accuracy": 0.0})

        elif ModelName == "gcn":
            print("=== TRAINING GCN MODEL ===")
            dataset = Planetoid(root="/katib/Cora", name="Cora") if data is None else None
            data = dataset[0] if data is None else data

            input_dim = config.pop("input_dim")
            hidden_dim = config.pop("hidden_dim")
            output_dim = config.pop("output_dim")
            model = create_gcn(input_dim, hidden_dim, output_dim, **config)

            optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate",0.01), weight_decay=config.get("weight_decay",0.0))
            
            def train(model,data):
                model.train()
                optimizer.zero_grad()
                out = model(data.x,data.edge_index)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                return loss.item()
            
            def eval_acc(mask):
                model.eval()
                out=model(data.x,data.edge_index)
                pred = out.argmax(dim=1)
                return (pred[mask]==data.y[mask]).sum().item()/mask.sum().item()
            
            for epoch in range(1,epochs+1):
                loss = train(model,data)
                acc = eval_acc(data.test_mask)
                print(f"Epoch {epoch}, Loss {loss:.4f}, Test Acc {acc:.4f}")
            
            final_acc = eval_acc(data.test_mask)
            save_metrics({"accuracy": final_acc})

        elif ModelName == "tgcn":
            print("=== TRAINING TGCN MODEL ===")
            if data is None:
                num_nodes, num_features, num_classes, num_timesteps = 200, 165, 2, 100
                x = torch.randn((num_timesteps, num_nodes, num_features))
                edge_index = torch.tensor([
                    [i for i in range(num_nodes)] + [(i+1) % num_nodes for i in range(num_nodes)],
                    [(i+1) % num_nodes for i in range(num_nodes)] + [i for i in range(num_nodes)]
                ], dtype=torch.long)

                y = torch.randint(0, num_classes, (num_nodes,))
                for c in range(num_classes):
                    y[c] = c

                data = Data(x=x[-1], edge_index=edge_index, y=y)

                indices = torch.randperm(num_nodes)
                train_size = int(0.7 * num_nodes)
                val_size = int(0.1 * num_nodes)
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                train_mask[indices[:train_size]] = True
                val_mask[indices[train_size:train_size+val_size]] = True
                test_mask[indices[train_size+val_size:]] = True
                data.train_mask = train_mask
                data.val_mask = val_mask
                data.test_mask = test_mask

            model = create_model(ModelName, config)
            for e in range(epochs):
                if hasattr(model, "train_step"):
                    loss = model.train_step(data)
                    print(f"Epoch {e+1}/{epochs} Loss {loss:.4f}")

            model.eval()
            with torch.no_grad():
                out = model.predict(data)
                pred = out.argmax(dim=1)

                test_pred = pred[data.test_mask].cpu().numpy()
                test_true = data.y[data.test_mask].cpu().numpy()

                rep = classification_report(
                    test_true, test_pred,
                    target_names=['Licit', 'Illicit'],
                    labels=[0, 1],
                    digits=4,
                    output_dict=True,
                    zero_division=0
                )
                print(rep)
            save_metrics({"accuracy": rep.get("accuracy", 0.0)})

        elif ModelName == "boxe":
            print("=== TRAINING BOXE MODEL ===")
            dataset = FB15k_237(root="/katib/fb15k237") if data is None else None
            data = dataset[0] if data is None else data
            data = ensure_edge_splits(data)

            model = BoxE(config)
            print("Training BoxE model with PyKEEN...")
            model.train_model()
            print("Training completed")

            embeddings = model.forward()
            print("Entity embeddings shape:", embeddings.shape)
            print("Evaluation metrics:")
            print(model.pykeen_result.metric_results.to_dict())
            mrr = model.pykeen_result.metric_results.get_metric("mrr")
            print(f"Final MRR={mrr}")

            h = torch.tensor([0, 1, 2])
            r = torch.tensor([0, 1, 2])
            t = torch.tensor([1, 2, 3])
            scores = model.decode(h, r, t)
            print("Triple scores:", scores)
            save_metrics({"mrr": mrr})

        elif ModelName == "stgnn":
            print("=== TRAINING STGNN MODEL ===")
            num_nodes = 6
            num_timesteps = 200
            num_features = 1
            seq_in_len = config.get("seq_in_len", 100)
            seq_out_len = config.get("seq_out_len", 1)
            config["num_nodes"] = num_nodes

            if data is None:
                t = torch.linspace(0, 20, num_timesteps)
                x = torch.sin(t).unsqueeze(-1).unsqueeze(-1).repeat(1, num_nodes, num_features) \
                    + 0.1 * torch.randn(num_timesteps, num_nodes, num_features)

                xs, ys = [], []
                for i in range(num_timesteps - seq_in_len - seq_out_len):
                    xs.append(x[i:i + seq_in_len].numpy())
                    ys.append(x[i + seq_in_len:i + seq_in_len + seq_out_len].numpy())

                xs, ys = np.array(xs), np.array(ys)
                mean, std = xs.mean(), xs.std()
                scaler = StandardScaler(mean, std)
                xs = scaler.transform(xs)

                batch_size = config.get("batch_size", 64)
                train_size = int(0.8 * len(xs))
                val_size = int(0.1 * len(xs))

                train_x, val_x, test_x = np.split(xs, [train_size, train_size + val_size])
                train_y, val_y, test_y = np.split(ys, [train_size, train_size + val_size])

                train_loader = DataLoaderM(train_x, train_y, batch_size)
                val_loader = DataLoaderM(val_x, val_y, batch_size)
                test_loader = DataLoaderM(test_x, test_y, batch_size)

                class STGNNDataWrapper:
                    pass

                data = STGNNDataWrapper()
                data.train_loader = train_loader
                data.val_loader = val_loader
                data.test_loader = test_loader
                data.scaler = scaler

            model = STGNN(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001),
                                        weight_decay=config.get("weight_decay", 0.0))
            criterion = nn.MSELoss()

            for e in range(epochs):
                if hasattr(model, "train_step"):
                    loss = model.train_step(data)
                    print(f"Epoch {e+1}/{epochs} Loss {loss}")

            eval_metrics = model.eval_step(data, getattr(data, "test_mask", None))
            print("Eval metrics:", eval_metrics)

            if "accuracy" in eval_metrics:
                save_metrics({"accuracy": eval_metrics["accuracy"]})
            elif "accuracy_percentage" in eval_metrics:
                save_metrics({"accuracy": eval_metrics["accuracy_percentage"]})
            else:
                save_metrics({"accuracy": 0.0})

        else:
            raise ValueError(f"Unknown model: {ModelName}")

        print("=== TRAINING COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        print(f"=== TRAINING FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Still save a metric so Katib doesn't hang
        save_metrics({"accuracy": 0.0})
        
        # Exit with error code
        sys.exit(1)
    
    print("=== TRIAL COMPLETED ===")
    # Ensure successful exit
    sys.exit(0)

"""
def save_metrics(metrics: dict):
    global_step = 1
    trial_id = "0"
    timestamp = time.time()
    print("These are the metrics that are being passed on to the save_metrics function")
    print(metrics)
    # Determine which metric to save
    if "accuracy" in metrics:
        metric_name = "accuracy"
        metric_value = metrics["accuracy"]
        print("Accuracy is the version")
    elif "accuracy_percentage" in metrics:
        metric_name = "accuracy"
        metric_value = metrics["accuracy_percentage"]
    else:
        # Pick the first remaining metric
        remaining_metrics = {k: v for k, v in metrics.items() if k not in ["checkpoint_path"]}
        if remaining_metrics:
            metric_name, metric_value = next(iter(remaining_metrics.items()))
        else:
            metric_name, metric_value = "accuracy", 0.0

    result = {"accuracy": float(metric_value)}
    print("Result:", result)
    with open("/katib/mnist.json", "w") as f:
        json.dump(result, f)

    
    #record = {
    #    "accuracy": float(metric_value),                                  #I am changing this to float value
    #    "checkpoint_path": metrics.get("checkpoint_path", ""),
    #    "global_step": str(global_step),
    #    "timestamp": timestamp,
    #    "trial": trial_id,
    #}
    #print('Record:',record)

    #save_path = "/katib/mnist.json"
    #os.makedirs(os.path.dirname(save_path), exist_ok=True)

    #with open(save_path, "w", encoding="utf-8") as f:                              #I am changing this from appending to writing
    #    json.dump(record, f)
    #    f.write("\n")
    #print("The metrics has been stored.")
    
"""
