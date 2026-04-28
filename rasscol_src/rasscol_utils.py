import csv
import sys
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rasscol_src.vina_utils import VinaJob

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


    def run_active_sampling(self, seqs, config, verbose=0):

        import lightgbm as lgb
        from scipy.stats import spearmanr
        from sklearn.preprocessing import OrdinalEncoder

        output_dir = Path(config["output_directory"])
        output_dir.mkdir(parents=True, exist_ok=True)

        step_size = config["gradient_boosted_step_size"]
        steps = config["gradient_boosted_steps"]
        final_num_designs = config["gradient_boosted_top_sequences"]
        num_cpus = config["num_cpus"]

        seed = config.get("seed", 42)
        rng = np.random.default_rng(seed)

        test_size = 5 * step_size
        val_size = 5 * step_size
        pool_multiplier = 100
        min_delta =  5e-4
        stopping_patience = config.get("stopping_patience", 5)

        eps = 1e-6

        # ------------------------- helper functions -------------------------

        def vina_to_target(vina_scores):
            """
            Convert docking scores to target where higher == better.
            Assumes vina_score values are negative.
            """
            vina_scores = np.asarray(vina_scores, dtype=float)
            affinity = np.clip(-vina_scores, eps, None)
            return np.log(affinity)

        def spearman_metric(y_true, y_pred):
            corr = spearmanr(y_true, y_pred).correlation
            if np.isnan(corr):
                corr = 0.0
            return "spearman", corr, True

        def top5_recall(y_true, y_pred, k=0.01):
            """
            Recall@5%, assuming higher values are better.
            """
            n = len(y_true)
            top_n_5 = max(1, int(n * k * 5))

            true_top_5 = np.argsort(-y_true)[:top_n_5]
            pred_top_5 = np.argsort(-y_pred)[:top_n_5]

            return len(set(true_top_5) & set(pred_top_5)) / top_n_5

        def mean_and_uncertainty(lgb_model, X):
            alpha = 0.5
            total_iterations = lgb_model.booster_.current_iteration()

            preds = []
            for _ in range(10):
                pred = lgb_model.booster_.shuffle_models().predict(
                    X,
                    num_iteration=int(alpha * total_iterations)
                )
                preds.append(pred)

            preds = np.stack(preds)
            mu = preds.mean(axis=0)
            sigma = preds.std(axis=0)

            return mu, sigma

        def load_docking_results_into_df(df):
            results_file = output_dir / "docking_results.csv"
            if not results_file.exists():
                return df

            results_df = (
                pd.read_csv(results_file)
                .drop_duplicates(subset="id", keep="last")
            )

            id_to_score = dict(zip(results_df["id"], results_df["vina_score"]))
            matched = df["id"].isin(id_to_score)

            df.loc[matched, "vina_score"] = df.loc[matched, "id"].map(id_to_score)
            df.loc[matched, "target"] = vina_to_target(
                df.loc[matched, "vina_score"].values
            )

            return df

        # ------------------------- dataframe -------------------------

        df = pd.DataFrame({
            "id": seqs["ids"],
            "pocket_seq": list(seqs["sequences"]),
            "cavity_vol": seqs["volumes"],
        })

        df["vina_score"] = np.nan
        df["target"] = np.nan

        # ------------------------- global categorical encoding -------------------------

        pos_cols = [list(seq) for seq in df["pocket_seq"]]

        encoder = OrdinalEncoder(
            dtype=int,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        X_seq = encoder.fit_transform(pos_cols)
        X_full = np.hstack([
            X_seq,
            df["cavity_vol"].values.reshape(-1, 1)
        ])

        seq_len = X_seq.shape[1]
        categorical_features = list(range(seq_len))

        N = X_full.shape[0]

        if N < test_size + val_size + step_size:
            raise ValueError(
                f"Dataset too small. Need at least {test_size + val_size + step_size} "
                f"sequences for test, validation, and initial sampling."
            )

        all_idx = rng.permutation(N)

        test_idx = all_idx[:test_size]
        val_idx = all_idx[test_size:test_size + val_size]
        pool_idx = all_idx[test_size + val_size:]

        # ------------------------- dock test + validation -------------------------

        heldout_idx = np.concatenate([test_idx, val_idx])

        print(f"Docking held-out test/validation set: {len(heldout_idx)} sequences")
        self.batched_jobs(heldout_idx, seqs, config)
        df = load_docking_results_into_df(df)

        if df.loc[test_idx, "target"].isna().any():
            raise RuntimeError("Missing docking results for test set.")

        if df.loc[val_idx, "target"].isna().any():
            raise RuntimeError("Missing docking results for validation set.")

        X_test = X_full[test_idx]
        y_test = df.loc[test_idx, "target"].values

        X_val = X_full[val_idx]
        y_val = df.loc[val_idx, "target"].values

        # ------------------------- initial active sample -------------------------

        sampled_mask = np.zeros(len(pool_idx), dtype=bool)

        initial_rel = rng.choice(len(pool_idx), size=step_size, replace=False)
        sampled_mask[initial_rel] = True

        initial_idx = pool_idx[initial_rel]

        print(f"Docking initial active-sampling set: {len(initial_idx)} sequences")
        self.batched_jobs(initial_idx, seqs, config)
        df = load_docking_results_into_df(df)

        # ------------------------- logging -------------------------

        log_file = output_dir / "RASSCoL_sampling.log"
        log_f = open(log_file, "w")

        history = []
        best_recall5 = -np.inf
        patience_counter = 0
        final_model = None

        # ------------------------- main loop -------------------------

        for step in range(steps):
            print(f"\nStep {step + 1}/{steps}")

            sampled_rel = np.flatnonzero(sampled_mask)
            sampled_abs = pool_idx[sampled_rel]

            if df.loc[sampled_abs, "target"].isna().any():
                raise RuntimeError("Some sampled sequences are missing docking results.")

            X_train = X_full[sampled_abs]
            y_train = df.loc[sampled_abs, "target"].values

            model = lgb.LGBMRegressor(
                boosting_type="gbdt",
                n_estimators=1000,
                num_leaves=96,
                max_bin=63,
                bagging_fraction=0.8,
                bagging_freq=1,
                feature_fraction=0.9,
                n_jobs=num_cpus,
                random_state=seed,
                verbose=verbose,
            )

            model.fit(
                X_train,
                y_train,
                categorical_feature=categorical_features,
                eval_set=[(X_val, y_val)],
                eval_metric=spearman_metric,
                callbacks=[
                    lgb.reset_parameter(
                        learning_rate=lambda i: 0.01 + (0.09 * np.power(0.99, i))
                    ),
                    lgb.early_stopping(
                        stopping_rounds=10,
                        min_delta=min_delta,
                        verbose=bool(verbose),
                    ),
                ],
            )

            final_model = model

            # ------------------------- evaluate pipeline on test set -------------------------

            mu_test, _ = mean_and_uncertainty(model, X_test)
            recall5 = top5_recall(y_test, mu_test)

            train_pred = model.predict(X_train)
            train_spearman = spearmanr(y_train, train_pred).correlation
            if np.isnan(train_spearman):
                train_spearman = 0.0

            num_trees = model.booster_.current_iteration()

            print(
                f"Train size: {len(sampled_abs)} | "
                f"Recall@5%: {recall5:.4f} | "
                f"Train Spearman: {train_spearman:.4f} | "
                f"Trees: {num_trees}"
            )

            history.append({
                "step": step + 1,
                "sampled": len(sampled_abs),
                "fraction_sampled": len(sampled_abs) / len(pool_idx),
                "recall5": recall5,
                "train_spearman": train_spearman,
                "num_trees": num_trees,
            })

            log_f.write(
                f"Step {step + 1}, "
                f"sampled={len(sampled_abs)}, "
                f"fraction_sampled={len(sampled_abs) / len(pool_idx):.6f}, "
                f"recall5={recall5:.6f}, "
                f"train_spearman={train_spearman:.6f}, "
                f"num_trees={num_trees}\n"
            )
            log_f.flush()

            # ------------------------- active-sampling early stopping -------------------------

            if recall5 > best_recall5 + 0.01:
                best_recall5 = recall5
                patience_counter = 0
            else:
                patience_counter += 1

            if stopping_patience is not None and patience_counter >= stopping_patience:
                print("Stopping active sampling: no improvement in Recall@5%.")
                break

            # ------------------------- pool-based acquisition -------------------------

            remaining_rel = np.flatnonzero(~sampled_mask)

            if len(remaining_rel) == 0:
                print("No remaining sequences to sample.")
                break

            subset_size = min(len(remaining_rel), pool_multiplier * step_size)
            subset_rel = rng.choice(remaining_rel, size=subset_size, replace=False)
            subset_abs = pool_idx[subset_rel]

            X_subset = X_full[subset_abs]

            mu, sigma = mean_and_uncertainty(model, X_subset)

            # Higher == better
            acquisition = mu + sigma

            k = min(step_size, len(subset_rel))
            selected_local = np.argpartition(-acquisition, k - 1)[:k]
            selected_rel = subset_rel[selected_local]
            selected_abs = pool_idx[selected_rel]

            sampled_mask[selected_rel] = True

            print(f"Docking {len(selected_abs)} newly selected sequences")
            self.batched_jobs(selected_abs, seqs, config)
            df = load_docking_results_into_df(df)

        # ------------------------- final selection -------------------------

        if final_model is None:
            raise RuntimeError("No model was trained.")

        print("\nPredicting over full active-sampling pool for final selection")

        X_candidates = X_full[pool_idx]
        mu_final, sigma_final = mean_and_uncertainty(final_model, X_candidates)

        pred_df = df.loc[
            pool_idx,
            ["id", "pocket_seq", "cavity_vol", "vina_score", "target"]
        ].copy()

        pred_df["prediction"] = mu_final
        pred_df["prediction_std"] = sigma_final

        pred_df = pred_df.sort_values("prediction", ascending=False)

        best_abs = pred_df.head(final_num_designs).index.values

        not_yet_docked = df.loc[best_abs, "vina_score"].isna()
        best_to_dock = best_abs[not_yet_docked.values]

        print(f"Docking {len(best_to_dock)} final top-predicted sequences")

        if len(best_to_dock) > 0:
            self.batched_jobs(best_to_dock, seqs, config)
            df = load_docking_results_into_df(df)

        # ------------------------- save outputs -------------------------

        history_df = pd.DataFrame(history)

        history_df.to_csv(output_dir / "RASSCoL_sampling_history.csv", index=False)
        pred_df.to_csv(output_dir / "RASSCoL_predictions.csv", index=True)
        df.to_csv(output_dir / "RASSCoL_active_sampling_results.csv", index=False)

        log_f.close()

        print("\nActive sampling completed.")

        return df, pred_df, history_df

