import csv
import sys
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.append("../rasscol_src")

from vina_utils import VinaJob

class RASSCCoL:

    @staticmethod
    def run_batch_job(ids, pocket_encodeds, vols, config, cleanup=True):
        result = []
        out_dir = Path(config["output_directory"]) / 'out'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for id, pocket_encoded, vol in zip(ids, pocket_encodeds, vols):

                vina_job = VinaJob(id, pocket_encoded, vol, config)
                result.append(vina_job.run_single_job(cleanup))  

        except Exception as e:
            print(f"Job {id} failed: {e}")
            result = [{'id': id, 'error': str(e)}]

        return result

    def run_docking_job_star(self, args):
        return self.run_batch_job(*args)
    
    @staticmethod
    def split_subbatches(ids, seqs, vols, num_splits):
        indices = np.array_split(np.arange(len(ids)), num_splits)
        return [(ids[i], seqs[i], vols[i]) for i in indices]

    @staticmethod
    def write_chunk_csv_append(results, output_dir):
        out_file = output_dir / "docking_results.csv"
        file_exists = out_file.exists()

        with open(out_file, "a", newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'id', 'vina_score', 'vina_score_norm',
                    'distance', 'sc_volume', 'cavity_volume', 'cavity_ligand_frac'
                ]
            )
            if not file_exists:
                writer.writeheader()
            for res in results:
                if 'error' not in res:
                    writer.writerow(res)
                    

    def batched_jobs(self, ids, seqs, config, cleanup=True, save_to_csv=True):

        total = ids.shape[0]
    
        batch_size = min(total, config['batch_size'])
        num_cpus = config['num_cpus']
        output_dir = Path(config['output_directory'])
        
        for i in range(0, total, batch_size):
            ids_batch = ids[i:i+batch_size]
            seqs_batch = seqs['sequences'][ids_batch]
            vols_batch = seqs['volumes'][ids_batch]

            # split into sub-batches for each CPU
            subbatches = self.split_subbatches(ids_batch, seqs_batch, vols_batch, num_cpus)
            args = [
                (sub_ids, sub_seqs, sub_vols, config, cleanup)
                for sub_ids, sub_seqs, sub_vols in subbatches
            ]

            # run this batch in parallel
            with Pool(processes=num_cpus) as pool:
                nested_results = pool.map(self.run_docking_job_star, args)

            # flatten and write to CSV
            if save_to_csv:
                all_results = [r for batch in nested_results for r in batch]
                self.write_chunk_csv_append(all_results, output_dir)

                print(f"\nCompleted batch {i // batch_size + 1}, wrote {len(all_results)} results.\n")
                
            else:
                print(f"\nCompleted batch {i // batch_size + 1}.\n")


    def run_active_sampling(self, seqs, config, stopping_patience = None, verbose = 0):
        
        import tensorflow as tf
        import tensorflow_decision_forests as tfdf
        
        output_dir = Path(config["output_directory"])
        output_dir.mkdir(parents=True, exist_ok=True)

        num_cpus = config['num_cpus']
        final_num_designs = config['gradient_boosted_top_sequences']
        steps = config['gradient_boosted_steps']
        step_size = config['gradient_boosted_step_size']

        N = len(seqs["sequences"])
        docking = np.zeros(N)

        seed = 42
        np.random.seed(seed)

        shuffled_ids = np.random.permutation(seqs["ids"])

        # shuffled_ids should be a list of unique indices 
        step_ids = shuffled_ids[:step_size]

        # ---------------------- Run initial data set ---------------------- #

        self.batched_jobs(step_ids, seqs, config)

        # load results
        results_df = pd.read_csv(output_dir / 'docking_results.csv').drop_duplicates(subset='id')

        docking[step_ids] = results_df['vina_score'].values

        df_rf = pd.DataFrame(seqs['sequences'], columns=[f's{i}' for i in range(seqs['sequences'].shape[1])])
        df_rf['Docking'] = docking
        train_index = shuffled_ids[:round(step_size*0.8)]
        mse_index = shuffled_ids[round(step_size*0.8):step_size]
        
        num_models = 10  # Number of models to train in the ensemble

        # ------------------------- Warnings and Initialization -------------------------

        mse_list = []

        # Set up logging 
        sampling_log_file = Path(config['output_directory']) / 'RASSCoL_sampling.log'
        open_sampling_log = open(sampling_log_file, 'w')

        if stopping_patience is None:
            warnings.warn('stopping_patience=None, no validation set will be merged with the training set')

        # ------------------------- Main Loop -------------------------


        for i in range(steps):
            print(f'\nStarting Step {i + 1}/{steps}')

            # Prepare training and test datasets
            df_train = df_rf.loc[train_index]
            df_mse = df_rf.loc[mse_index]
            
            df_test = df_rf.drop(np.concatenate((train_index, mse_index)))
            
            # Drop the "Predictions" column if it exists
            features = [col for col in df_train.columns if col not in ['Docking', 'Predictions', 'Predictions_std']]

            # Convert data to TensorFlow datasets
            train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_train[features + ['Docking']], label='Docking', task=tfdf.keras.Task.REGRESSION)
            test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_test[features + ['Docking']], label='Docking', task=tfdf.keras.Task.REGRESSION)
            mse_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_mse[features + ['Docking']], label='Docking', task=tfdf.keras.Task.REGRESSION)

            # ------------------------- Model Training -------------------------

            print('Training Gradient Boosted Trees models...')
            tuner = tfdf.tuner.RandomSearch(num_trials=20, use_predefined_hps=True)
            models = []
            
            from concurrent.futures import ThreadPoolExecutor

            def train_model(seed):
                model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner, task=tfdf.keras.Task.REGRESSION, random_seed=seed, verbose=verbose)
                model.fit(train_ds)
                return model

            with ThreadPoolExecutor(max_workers=num_cpus) as executor:
                models = list(executor.map(train_model, range(num_models)))

            print(f'{num_models} models trained successfully.')

            # ------------------------- Combine Models for Ensemble -------------------------

            class CombinedModel(tf.keras.Model):
                        """Ensemble model combining predictions from multiple Gradient Boosted Trees models."""
                        def call(self, inputs):
                            predictions = tf.concat([submodel(inputs) for submodel in models], axis=1)
                            return tf.math.reduce_mean(predictions, axis=1), tf.math.reduce_std(predictions, axis=1)

            combined_model = CombinedModel()
            train_predictions, _ = combined_model.predict(train_ds)

            # ------------------------- Evaluate on MSE Dataset -------------------------

            mse_predictions_scaled, mse_predictions_std = combined_model.predict(mse_ds)
            mse = np.mean(np.square(mse_predictions_scaled - df_mse['Docking'].values))

            mse_list.append(mse)
            open_sampling_log.write(f'Step {i + 1} validation MSE: {mse:.4f}')

            # Check for early stopping
            if stopping_patience is not None:

                if i >= stopping_patience and all(x > mse for x in mse_list[-stopping_patience:]):
                    print(f'Stopping early after {i + 1} steps due to no improvement in MSE.')
                    break
                
            # ------------------------- Active Sampling -------------------------
            
            test_predictions_scaled, prediction_std = combined_model.predict(test_ds)

            # Save predictions
            df_test['Predictions'] = test_predictions_scaled
            df_test['Predictions_std'] = prediction_std

            # Select samples for the next round
            worst_index = df_test.nlargest(round(step_size * 0.25), 'Predictions_std').index  # High uncertainty
            best_index = df_test.nsmallest(round(step_size * 0.25), 'Predictions').index  # Best docking scores

            # Combine worst and best indices, ensuring no duplicates
            combined_index = worst_index.union(best_index)

            # Exclude indices already in combined_index to avoid duplicates
            allowed_sample_space = df_test.index.difference(combined_index)

            # Randomly sample from the allowed sample space, sometime best and least certain data points overlap - random fraction needs to be adjusted
            random_index = allowed_sample_space.to_series().sample(n=round(step_size-len(combined_index)), random_state=seed).index

            # Final combined index with worst, best, and random samples
            final_combined_index  = pd.Index(combined_index).union(random_index)
            #sanity check final_combined_index = new_test_index
            new_test_index = np.array(final_combined_index.difference(results_df.index))
            print(len(final_combined_index),len(new_test_index))

            # ------------------------- Dock New Samples -------------------------

            print(f'Docking {len(new_test_index)} new samples...')
            self.batched_jobs(new_test_index, seqs, config)
            
            results_df = pd.read_csv(output_dir / 'docking_results.csv').drop_duplicates(subset='id')

            docking[new_test_index] = results_df.query("id in @new_test_index")['vina_score'].values
            df_rf.loc[new_test_index, 'Docking'] = docking[new_test_index]

            train_index = np.union1d(train_index, new_test_index)
            
        # ----------------------------- Dock Best predictions -----------------------

        # prevent redocking already sampled sequences
        best_index = df_test.nsmallest(final_num_designs, 'Predictions').index
        self.batched_jobs(best_index, seqs, config)
        
        df_test.to_csv(Path(config['output_directory']) / 'RASSCoL_DF_test.csv')

        df_mse['Predictions'] = mse_predictions_scaled
        df_mse['Predictions_std'] = mse_predictions_std
        df_mse.to_csv(Path(config['output_directory']) / 'RASSCoL_DF_validate.csv')

        open_sampling_log.close()

        print('\nActive learning process completed.')

