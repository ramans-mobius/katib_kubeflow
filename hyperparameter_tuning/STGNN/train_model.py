if args.model_name.lower() == 'stgnn':
  base_data_dir = "data"
  smd_data_dir = os.path.join(base_data_dir, "machine-1-1")
  if os.path.exists(smd_data_dir) and os.path.exists(os.path.join(smd_data_dir, 'train.pkl')):
      print("Data already exists.")
  else:
      print("Downloading and preparing data...")
      workdir = "interfusion_temp"
      repo_url = "https://github.com/zhhlee/InterFusion.git"
      
      if os.path.exists(workdir):
          shutil.rmtree(workdir)
      os.makedirs(workdir)
      
      try:
          subprocess.run(["git", "clone", repo_url, os.path.join(workdir, "InterFusion")], check=True, capture_output=True)
          
          source_dir = os.path.join(workdir, "InterFusion", "data", "processed")
          
          os.makedirs(smd_data_dir, exist_ok=True)
  
          shutil.copy(os.path.join(source_dir, 'machine-1-1_train.pkl'), os.path.join(smd_data_dir, 'train.pkl'))
          shutil.copy(os.path.join(source_dir, 'machine-1-1_test.pkl'), os.path.join(smd_data_dir, 'test.pkl'))
          shutil.copy(os.path.join(source_dir, 'machine-1-1_test_label.pkl'), os.path.join(smd_data_dir, 'test_label.pkl'))
          
          print("Data prepared successfully.")
      finally:
          if os.path.exists(workdir):
              shutil.rmtree(workdir)
      
  #return smd_data_dir
  #smd_data_dir = dataset_dir
  
  print("\nCreating STGNN model from config...")
      config = get_config_by_name('basic_stgnn', 'configs/stgnn_configs.yaml')
  
  model = STGNN(config)
  
  print("Model created successfully.")
  
  # --- 5. Train Model ---
  print("\n--- Starting Model Training ---")
  for epoch in range(model.epochs):
      loss = model.train_step(data, data.train_mask)
      if epoch % 1 == 0:
          print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
  print("--- Finished Model Training ---\n")
  
  
  # --- 6. Evaluate Model ---
  print("--- Starting Model Evaluation ---")
  eval_metrics = model.eval_step(data, data.test_mask)
  print(f"Test Accuracy: {eval_metrics['accuracy']:.4f}")
  
  print("--- Finished Model Evaluation ---")
  
  # Dummy value for global_step
  global_step = 1
  trial_id = "0"  # Can also pass via env var or leave as default
  
  timestamp = time.time()
  
  # --- 7. Single Inference and Anomaly Detection ---
  print("\n--- Starting Single Inference and Anomaly Detection ---")
  infer_and_detect_anomaly(model, data, config)
  print("--- Finished Single Inference and Anomaly Detection ---")
  
  metrics = [
      {
          "accuracy": f"{eval_metrics['accuracy']:.4f}",
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



  
      
  
  
