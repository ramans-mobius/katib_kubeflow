# train_slm.py
import argparse
import base64
import json
import os
import time
import sys

# Import exactly as shown in the working train_slm_probir.py
from nesy_factory.language_model.text_exporter import TextExporter
from nesy_factory.language_model.tokenize import ByteLevelBPETokenizer
from nesy_factory.language_model.gemma import Gemma3Builder
from nesy_factory.language_model.train import GemmaTrainer

# ------------------ DEFAULT CONFIG ------------------
DEFAULT_CONFIG = {
    "gemma": {
        # String parameters
        "dataset": "roneneldan/TinyStories",
        "split": "train", 
        "text_fields": "text",
        "layer_pattern": "S*3,F*1,S*2",
        "streaming": "true",
        "add_bos_eos": "true",
        "special_tokens": "[PAD],[UNK],[BOS],[EOS]",
        
        # Integer parameters  
        "n_layers": 6,
        "warmup_steps": 1,
        "max_iters": 50,
        "batch_size": 8,
        "block_size": 64,
        "grad_accum": 1,
        "eval_interval": 1,
        "eval_iters": 5,
        "num_proc": 4,
        "vocab_size": 50257,
        "min_frequency": 2,
        
        # String parameters that represent numbers
        "learning_rate": "0.0001",
        "min_lr": "0.00001", 
        "weight_decay": "0.1",
        "beta2": "0.95",
        "clip_grad_norm": "0.5",
        "val_fraction": "0.1",
        "max_rows_dataset": "1000",
        "max_rows_train": "1000",
        
        # File paths
        "train_corpus": "/katib/train.txt",
        "tokenizer_json": "/katib/tokenizer.json", 
        "model_config": "/katib/gemma3_config.json",
        "model_weights": "/katib/gemma3.pt",
        "model_py_in": "/katib/gemma3_model.py",
        "model_py_out": "/katib/artifacts/gemma3_model.py",
        "best_weights": "/katib/artifacts/best.pt",
        "final_weights": "/katib/artifacts/final.pt",
        "training_report": "/katib/artifacts/training_report.json",
        "loss_curve_csv": "/katib/artifacts/loss_curve.csv"
    }
}

# ------------------ UTILITY ------------------
def _auto_cast(v):
    """Enhanced auto-cast that handles all types properly"""
    if isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, str):
        v_lower = v.lower().strip()
        
        # Handle boolean strings
        if v_lower in ("true", "false"):
            return v_lower == "true"
        # Handle None/null
        if v_lower in ("none", "null", "nan", "unavailable"):
            return None
        # Handle integer strings
        if v.isdigit() or (v.startswith('-') and v[1:].isdigit()):
            return int(v)
        # Handle float strings
        try:
            return float(v)
        except ValueError:
            # Return as string if it can't be converted to number
            return v
    return v

def ensure_directory_for_path(file_path):
    """Ensure the directory for a file path exists"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def ensure_paths_exist(cfg):
    """Ensure all directory paths in config exist"""
    print("=== ENSURING PATHS EXIST ===")
    
    # List of path keys that need directories
    path_keys = [
        "train_corpus", "tokenizer_json", "model_config", "model_weights", 
        "model_py_in", "model_py_out", "best_weights", "final_weights",
        "training_report", "loss_curve_csv"
    ]
    
    for key in path_keys:
        if key in cfg:
            ensure_directory_for_path(cfg[key])
            print(f"✓ Path ensured for: {key} -> {cfg[key]}")

# ------------------ ARG PARSING ------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--config", type=str, help="Base64 encoded config JSON")
    args, unknown = parser.parse_known_args()

    print("========INPUTS==========")
    print("Model Name:", args.model_name)
    print("Base64 encoded string:", args.config)

    # Get model-specific defaults from nested structure
    model_name_lower = args.model_name.lower()
    model_defaults = DEFAULT_CONFIG.get(model_name_lower, {})
    
    # Auto-cast ALL default values first
    cfg = {}
    for k, v in model_defaults.items():
        cfg[k] = _auto_cast(v)
    
    source = {k: "hardwired_defaults" for k in cfg}

    print(f"[CONFIG] Loaded defaults for model: {model_name_lower}")

    # Base64 config
    if args.config:
        try:
            decoded = base64.b64decode(args.config).decode("utf-8")
            b64_cfg = json.loads(decoded)
            print('======Configuration decoded from base64 config=====')
            print(b64_cfg)
            # Auto-cast all values from base64 config and update
            casted_b64_cfg = {}
            for k, v in b64_cfg.items():
                casted_b64_cfg[k] = _auto_cast(v)
            cfg.update(casted_b64_cfg)
            for k in casted_b64_cfg:
                source[k] = "config_json"
            print("[CONFIG] Loaded and casted base64 config")
        except Exception as e:
            print(f"[WARNING] Failed to decode base64 config: {e}")

    # CLI overrides with auto-casting
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if tok.startswith("--"):
            key = tok.lstrip("-")
            if "=" in key:
                k, sval = key.split("=", 1)
                val = _auto_cast(sval)
                cfg[k] = val
                source[k] = "cli_override"
                i += 1
            else:
                if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                    val = _auto_cast(unknown[i + 1])
                    cfg[key] = val
                    source[key] = "cli_override"
                    i += 2
                else:
                    cfg[key] = True
                    source[key] = "cli_override"
                    i += 1
        else:
            i += 1

    # Ensure all paths exist
    ensure_paths_exist(cfg)

    # Print final config with types
    print("\n[FINAL CONFIG WITH TYPES]")
    for k in sorted(cfg):
        value = cfg[k]
        value_type = type(value).__name__
        print(f"  {k}: {value} (type: {value_type}, from {source.get(k, 'unknown')})")

    return args, cfg

# ------------------ FILTER TRAINER KEYS ------------------
def filter_trainer_keys(cfg):
    trainer_keys = [
        "tokenizer_json", "train_corpus", "model_config", "model_weights", "model_py_in", "model_py_out",
        "learning_rate", "min_lr", "warmup_steps", "max_iters", "batch_size", "block_size",
        "grad_accum", "eval_interval", "eval_iters", "weight_decay", "beta2", "clip_grad_norm",
        "val_fraction", "num_proc", "best_weights", "final_weights", "training_report", "loss_curve_csv"
    ]
    filtered_cfg = {}
    for k in trainer_keys:
        if k in cfg:
            # Apply auto-casting again to ensure all numeric values are properly converted
            filtered_cfg[k] = _auto_cast(cfg[k])
    
    return filtered_cfg

# ------------------ TEXT EXPORTER INTEGRATION ------------------
def export_dataset(cfg):
    """Export dataset using the provided configuration"""
    print("=== EXPORTING DATASET ===")
    
    # Convert string parameters to appropriate types for TextExporter
    max_rows = int(cfg.get("max_rows_dataset", 10000)) if cfg.get("max_rows_dataset") else None
    streaming = cfg.get("streaming", "true")
    
    # Handle text_fields - always convert to comma-separated string for TextExporter
    text_fields = cfg.get("text_fields", "text")
    if isinstance(text_fields, list):
        text_fields_str = ",".join(text_fields)
    else:
        text_fields_str = str(text_fields)
    
    print(f"[DATASET] Using text fields: {text_fields_str}")
    print(f"[DATASET] Max rows: {max_rows}, Streaming: {streaming}")
    
    meta = TextExporter().run(
        dataset=cfg.get("dataset"),
        split=cfg.get("split"),
        output_file=cfg["train_corpus"],
        max_rows=max_rows,
        streaming=streaming,
        text_fields=text_fields_str  # Always pass as string
    )
    print(f"[INFO] Exported {meta['exported_rows']} rows to {meta['output_file']}")
    return meta

# ------------------ TOKENIZER INTEGRATION ------------------
def setup_tokenizer(cfg):
    """Setup tokenizer using the provided configuration"""
    print("=== SETTING UP TOKENIZER ===")
    
    # Create tokenizer instance exactly like in train_slm_probir.py
    tok = ByteLevelBPETokenizer()
    
    if not os.path.exists(cfg["tokenizer_json"]):
        print(f"[INFO] Tokenizer {cfg['tokenizer_json']} not found. Training...")
        
        # Convert parameters to appropriate types
        vocab_size = int(cfg.get("vocab_size", 32000))
        min_frequency = int(cfg.get("min_frequency", 2))
        add_bos_eos = cfg.get("add_bos_eos", "true")
        
        # Handle special_tokens - always convert to comma-separated string
        special_tokens = cfg.get("special_tokens", "[PAD],[UNK],[BOS],[EOS]")
        
        print(f"[TOKENIZER] Using special tokens: {special_tokens}")
        
        # Call tokenizer training exactly like in train_slm_probir.py
        tok.run(
            text_file=cfg["train_corpus"],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,  # Always pass as string
            add_bos_eos=add_bos_eos
        )
        tok.save(cfg["tokenizer_json"])
        print(f"[INFO] Tokenizer saved at {cfg['tokenizer_json']}")
    else:
        tok.load(cfg["tokenizer_json"])
        print(f"[INFO] Loaded existing tokenizer from {cfg['tokenizer_json']}")

    # Test tokenizer exactly like in train_slm_probir.py
    ids = tok.encode("hello world")
    text = tok.decode(ids)
    print("[DEBUG] tokenizer test →", ids, "→", text)
    return tok

# ------------------ METRICS SAVING ------------------
def save_metrics(metric_name, metric_value):
    """Save metrics using the actual metric name (best_val_loss)"""
    global_step = 1
    trial_id = "0"
    timestamp = time.time()
    print(f"=== SAVING METRICS ===")
    print(f"Metric: {metric_name} = {metric_value}")
    
    # Use the actual metric name (best_val_loss) instead of converting to accuracy
    result = {metric_name: f"{metric_value:.4f}","checkpoint_path": "","global_step": str(global_step),"timestamp": timestamp,"trial": trial_id,}
    
    metrics_path = "/katib/mnist.json"
    
    # Ensure directory exists
    ensure_directory_for_path(metrics_path)
    
    # OVERWRITE the file (not append)
    with open("/katib/mnist.json", "a", encoding="utf-8") as f:  #I am changing this from appending to writing
        json.dump(result, f)
        f.write("\n")
    
    print(f"✓ Metrics saved to {metrics_path}")
    print(f"Result: {result}")
    
    # Verify the file was written
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            content = f.read()
            print(f"✓ File contents: {content}")
    else:
        print("❌ ERROR: Metrics file was not created!")

# ------------------ MAIN ------------------
def main():
    try:
        try:
            args, cfg = parse_args()
            print("=== CONFIG VALIDATION ===")
            
            # Validate critical numeric parameters
            critical_params = ['learning_rate', 'batch_size', 'max_iters', 'block_size']
            for param in critical_params:
                if param in cfg:
                    value = cfg[param]
                    print(f"VALIDATE: {param} = {value} (type: {type(value).__name__})")
                    if not isinstance(value, (int, float)):
                        print(f"WARNING: {param} is not numeric: {value}")
                        
        except Exception as e:
            print(f"=== CONFIG PARSING ERROR ===")
            print(f"Error: {e}")
            print("This usually means a parameter type mismatch")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        if args.model_name.lower() == "gemma":
            print("=== STARTING GEMMA TRAINING ===")
            
            # ------------------ Dataset ------------------
            export_dataset(cfg)
            
            # ------------------ Tokenizer ------------------
            tok = setup_tokenizer(cfg)

            # ------------------ Model ------------------
            print("=== BUILDING MODEL ===")
            
            # Convert model parameters to appropriate types
            n_layers = int(cfg.get("n_layers", 6))
            layer_pattern = cfg.get("layer_pattern", "S*3,F*1,S*2")
            
            # Build model exactly like in train_slm_probir.py
            summary = Gemma3Builder().run(
                tokenizer_json=cfg["tokenizer_json"],
                n_layers=n_layers,
                layer_pattern=layer_pattern,
                model_weights_out=cfg["model_weights"],
                model_config_out=cfg["model_config"],
                model_py_out=cfg["model_py_in"],
            )
            print("======SUMMARY======")
            print(summary)

            # ------------------ Training ------------------
            print("=== STARTING TRAINING ===")
            trainer_cfg = filter_trainer_keys(cfg)
            
            # Train model exactly like in train_slm_probir.py
            trainer = GemmaTrainer()
            report = trainer.run(**trainer_cfg)
            
            print("[INFO] Training finished. Report:", json.dumps(report, indent=2))

            # ------------------ Save metrics for Katib ------------------
            print("=== SAVING METRICS FOR KATIB ===")
            
            # Get the metric value from training report - use best_val_loss directly
            objective_metric_name = "best_val_loss"
            metric_value = report.get("best_val_loss", report.get("final_val_loss", 0.0))
            
            print(f"Training metric - {objective_metric_name}: {metric_value}")
            
            # Save using the actual metric name (best_val_loss)
            save_metrics(objective_metric_name, metric_value)

        else:
            raise ValueError(f"Unknown model_name {args.model_name}")
        
        print("=== TRAINING COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        print(f"=== TRAINING FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save a default metric so Katib doesn't hang - but use best_val_loss
        print("Saving default metrics due to failure...")
        save_metrics("best_val_loss", 100.0)  # High loss value for failure
        
        # Exit with error code
        sys.exit(1)
    
    print("=== TRIAL COMPLETED ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
