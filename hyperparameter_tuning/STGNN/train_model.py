if args.model_name.lower() == "stgnn":
        # --- Data + Helper Classes ---
        class DataWrapper:
            def __init__(self, data_dict):
                self.__dict__.update(data_dict)

        class StandardScaler:
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

        def create_sliding_windows(data, seq_in_len, seq_out_len):
            x, y = [], []
            num_samples = len(data)
            for i in range(num_samples - seq_in_len - seq_out_len + 1):
                x.append(data[i : i + seq_in_len])
                y.append(data[i + seq_in_len : i + seq_in_len + seq_out_len])
            return np.array(x), np.array(y)

        def load_dataset(dataset_dir, config, scaling_required=True):
            batch_size = config['batch_size']
            seq_in_len = config['seq_in_len']
            seq_out_len = config['seq_out_len']

            with open(os.path.join(dataset_dir, 'train.pkl'), 'rb') as f:
                train_raw = pickle.load(f)
            with open(os.path.join(dataset_dir, 'test.pkl'), 'rb') as f:
                test_raw = pickle.load(f)

            x_train, y_train = create_sliding_windows(train_raw, seq_in_len, seq_out_len)
            x_test, y_test = create_sliding_windows(test_raw, seq_in_len, seq_out_len)

            x_train = np.expand_dims(x_train, axis=-1)
            y_train = np.expand_dims(y_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)
            y_test = np.expand_dims(y_test, axis=-1)

            num_train = int(len(x_train) * 0.8)
            x_val, y_val = x_train[num_train:], y_train[num_train:]
            x_train, y_train = x_train[:num_train], y_train[:num_train]

            data = {}
            data['x_train'], data['y_train'] = x_train, y_train
            data['x_val'], data['y_val'] = x_val, y_val
            data['x_test'], data['y_test'] = x_test, y_test

            scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

            if scaling_required:
                for category in ['train', 'val', 'test']:
                    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

            data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
            data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], batch_size)
            data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], batch_size)
            data['scaler'] = scaler

            data['train_mask'] = torch.ones(len(data['x_train']), dtype=torch.bool)
            data['test_mask'] = torch.ones(len(data['x_test']), dtype=torch.bool)

            return data

        # --- Anomaly Detection Class ---
        class anomaly_dd:
            def __init__(self, train_obs, val_obs, test_obs,
                         train_forecast, val_forecast, test_forecast,
                         window_length=None, batch_size=512, root_cause=False):
                self.train_obs = train_obs
                self.val_obs = val_obs
                self.test_obs = test_obs
                self.train_forecast = train_forecast
                self.val_forecast = val_forecast
                self.test_forecast = test_forecast
                self.root_cause = root_cause
                if window_length is None:
                    self.window_length = len(train_obs) + len(val_obs)
                else:
                    self.window_length = window_length
                self.batch_size = batch_size

                if self.root_cause:
                    self.val_re_full = None
                    self.test_re_full = None

            def pca_model(self, val_error, test_error, dim_size=1):
                from sklearn.decomposition import PCA
                pca = PCA(n_components=dim_size, svd_solver='full')
                pca.fit(val_error)
                transf_val_error = pca.inverse_transform(pca.transform(val_error))
                transf_test_error = pca.inverse_transform(pca.transform(test_error))
                val_re_full = np.absolute(transf_val_error - val_error)
                val_re = val_re_full.sum(axis=1)
                test_re_full = np.absolute(transf_test_error - test_error)
                test_re = test_re_full.sum(axis=1)
                return val_re, test_re, val_re_full, test_re_full

            def scorer(self, num_components):
                val_abs = np.absolute(self.val_obs - self.val_forecast)
                full_obs = np.concatenate((self.train_obs, self.val_obs, self.test_obs), axis=0)
                full_forecast = np.concatenate((self.train_forecast, self.val_forecast, self.test_forecast), axis=0)
                full_abs = np.absolute(full_obs - full_forecast)
                val_norm = (val_abs - np.median(val_abs, axis=0)) / (np.quantile(val_abs, 0.75, axis=0) - np.quantile(val_abs, 0.25, axis=0) + 1e-2)
                test_norm = (full_abs - np.median(full_abs, axis=0)) / (np.quantile(full_abs, 0.75, axis=0) - np.quantile(full_abs, 0.25, axis=0) + 1e-2)
                val_re, test_re, val_re_full, test_re_full = self.pca_model(val_norm, test_norm, num_components)
                if self.root_cause:
                    self.val_re_full = val_re_full
                    self.test_re_full = test_re_full
                realtime_indicator = test_re
                anomaly_prediction = test_re > val_re.max()
                return realtime_indicator, anomaly_prediction

        def infer_and_detect_anomaly(model, data, config):
            model.eval()
            x_test = data.x_test
            y_test = data.y_test
            scaler = data.scaler
            inference_idx = 100
            if len(x_test) <= inference_idx:
                print(f"Test set has less than {inference_idx + 1} data points.")
                return
            single_x = torch.Tensor(x_test[inference_idx:inference_idx+1]).to(model.device)
            single_y_true = y_test[inference_idx:inference_idx+1].squeeze()
            with torch.no_grad():
                pred_y_raw = model(single_x)
                pred_y = pred_y_raw.transpose(1, 3)
            pred_y_unscaled = scaler.inverse_transform(pred_y)
            pred_y_numpy = pred_y_unscaled.squeeze().cpu().detach().numpy()
            print(f"\n--- Inference on Data Point at Index {inference_idx} ---")
            print("Predicted Value:\n", pred_y_numpy)
            print("True Value:\n", single_y_true)
            val_outputs = []
            for i, (x, y) in enumerate(data.val_loader.get_iterator()):
                val_x = torch.Tensor(x).to(model.device)
                with torch.no_grad():
                    preds_raw = model(val_x)
                val_outputs.append(preds_raw.transpose(1, 3))
            val_yhat = torch.cat(val_outputs, dim=0)
            val_realy = torch.Tensor(data.y_val).to(model.device)
            val_yhat = val_yhat[:val_realy.size(0), ...]
            val_pred_unscaled = scaler.inverse_transform(val_yhat)
            val_pred_numpy = val_pred_unscaled.squeeze().cpu().detach().numpy()
            val_label_numpy = val_realy.squeeze().cpu().detach().numpy()
            anomaly_detector = anomaly_dd(
                train_obs=np.array([]).reshape(0, val_label_numpy.shape[1]),
                val_obs=val_label_numpy,
                test_obs=np.array([single_y_true]),
                train_forecast=np.array([]).reshape(0, val_pred_numpy.shape[1]),
                val_forecast=val_pred_numpy,
                test_forecast=np.array([pred_y_numpy]),
                window_length=None,
                root_cause=True
            )
            indicator, prediction = anomaly_detector.scorer(num_components=8)
            print("\n--- Anomaly Detection Result ---")
            print(f"Anomaly Score: {indicator[0]}")
            print(f"Result: {'Anomaly Detected' if prediction[0] else 'No Anomaly Detected'}")
            if prediction[0]:
                error_contribution = anomaly_detector.test_re_full[0]
                sorted_indices = np.argsort(error_contribution)[::-1]
                print("\n--- Root Cause Analysis (Top 3) ---")
                for i in range(min(3, len(sorted_indices))):
                    param_index = sorted_indices[i]
                    print(f"Parameter Index: {param_index}, Contribution: {error_contribution[param_index]:.4f}")

        # --- Data Preparation ---
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

        # --- Model Creation ---
        print("\nCreating STGNN model from config...")
        config = get_config_by_name('basic_stgnn', 'configs/stgnn_configs.yaml')
        model = STGNN(config)
        print("Model created successfully.")

        # --- Load Data ---
        data_dict = load_dataset(smd_data_dir, config)
        data = DataWrapper(data_dict)

        # --- Training ---
        print("\n--- Starting Model Training ---")
        for epoch in range(model.epochs):
            loss = model.train_step(data, data.train_mask)
            if epoch % 1 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
        print("--- Finished Model Training ---\n")

        # --- Evaluation ---
        print("--- Starting Model Evaluation ---")
        eval_metrics = model.eval_step(data, data.test_mask)
        print(f"Test Accuracy: {eval_metrics['accuracy']:.4f}")
        print("--- Finished Model Evaluation ---")

        # --- Anomaly Inference ---
        print("\n--- Starting Single Inference and Anomaly Detection ---")
        infer_and_detect_anomaly(model, data, config)
        print("--- Finished Single Inference and Anomaly Detection ---")

        # --- Metrics Logging (Katib style) ---
        global_step = 1
        trial_id = "0"
        timestamp = time.time()
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

  
