import argparse, itertools, json, pickle, time
import numpy as np
import pandas as pd

import baseline as baseline

# ---------------------------------------------------------------------
# Hyper‑parameter grids
# ---------------------------------------------------------------------

GRIDS = dict(
    mean_teacher_regression = dict(
        lr          = [1e-4, 3e-4, 1e-3],
        w_max       = [0.1, 0.5, 1.0],
        alpha       = [0.95, 0.99, 0.995],
        ramp_len    = [1, 10, 50],
    ),

    fixmatch_regression = dict(
        lr                = [1e-4, 3e-4, 1e-3],
        batch_size        = [32, 64],
        alpha_ema         = [0.99, 0.999],
        lambda_u_max      = [0.5, 1.0],
        rampup_length     = [10, 30],
        conf_threshold    = [0.05, 0.1],
    ),

    laprls_regression = dict(
        lam   = [1e-5, 1e-3, 1e-1],
        gamma = [1e-3, 1e-1, 1],
        k     = [5, 10, 20],
        sigma = [0.5, 1.0, 2.0],
    ),

    tnnr_regression = dict(
        rep_dim = [32, 64, 128],
        beta    = [0.01, 0.1, 1.0],
        lr      = [1e-4, 3e-4, 1e-3],
    ),

    tsvr_regression = dict(
        C                   = [0.1, 1, 10],
        epsilon             = [0.01, 0.1],
        gamma               = [0.1, 1],
        self_training_frac  = [0.1, 0.2, 0.5],
    ),

    ucvme_regression = dict(
        lr      = [1e-4, 3e-4, 1e-3],
        w_unl   = [1, 5, 10],
        mc_T    = [5, 10],
    ),

    rankup_regression = dict(
        hidden_dim   = [128, 256, 512],
        alpha_rda    = [0.01, 0.05, 0.1],
        temperature  = [0.5, 0.7, 1.0],
        tau          = [0.8, 0.9, 0.95],
        lr           = [1e-4, 1e-3],
    ),

    gcn_regression = dict(
        hidden   = [32, 64, 128],
        dropout  = [0.0, 0.1, 0.3],
        lr       = [1e-3, 3e-3],
    ),

    kernel_mean_matching_regression = dict(
        alpha         = [1e-2, 1e-1],
        kmm_B         = [100, 1000],
        kmm_eps       = [1e-3, 1e-2],
        sigma         = [0.5, 1.0],
    ),

    em_regression = dict(
        n_components  = [2, 3],
        max_iter      = [100, 200],
        tol           = [1e-3, 1e-4],
        eps           = [1e-3, 1e-4],
    ),

    eot_barycentric_regression = dict(
        eps = [1e-3, 1e-2, 1e-1, 1, 10],  # or flexible according to scale of X, Y, which is what we eventually use
        ridge_alpha = [1e-2, 1e-3, 1e-4],
        tol         = [1e-5, 1e-7, 1e-9],
    ),
    gw_metric_alignment_regression = dict(
        max_iter    = [200, 400, 800],
        tol         = [1e-5, 1e-7, 1e-9],
    )
)

# ---------------------------------------------------------------------
def mse(y_pred, y_true):
    return ((y_pred - y_true)**2).mean()


def product_dict(**kwargs):
    keys = kwargs.keys()
    for values in itertools.product(*kwargs.values()):
        yield dict(zip(keys, values))

def swap_columns(df):
        df["tmp"]               = df["morph_coordinates"]
        df["morph_coordinates"] = df["gene_coordinates"]
        df["gene_coordinates"]  = df.pop("tmp")
        return df

def eval_one(rev, name, sup_df, inf_df, out_df, **kwargs):
    # if rev:
    #     name = 'reversed_' + name
    #     print(sup_df.columns)
    #     sup_rev = sup_df.rename(columns={'yv': 'morph_coordinates',
    #                                  'x':  'gene_coordinates'}).copy()
    #     inf_rev = inf_df.rename(columns={'yv': 'morph_coordinates',
    #                                  'x':  'gene_coordinates'}).copy()
    #     out_rev = out_df.rename(columns={'yv': 'morph_coordinates',
    #                                     'x':  'gene_coordinates'}).copy()
    #     func = getattr(baseline, name)
    #     preds, truth = func(
    #         image_df=out_rev,
    #         gene_df=inf_rev,
    #         supervised_df=sup_rev,
    #         inference_df=inf_rev,
    #         **kwargs
    #     )

    func = getattr(baseline, name)
    
    if name in ['kernel_mean_matching_regression','em_regression','eot_barycentric_regression','gw_metric_alignment_regression']:
        print("yes")
        preds, truth = func(
            image_df=inf_df,
            gene_df=out_df,
            supervised_df=sup_df,
            inference_df=inf_df,
            **kwargs
        )      
    else:
        # generic signature: sup_df, inf_df, **kwargs
        preds, truth = func(sup_df, inf_df, **kwargs)
    return mse(preds, truth)

# ---------------------------------------------------------------------
def load_data(data_dir):
    with open(f'{data_dir}/supervised_df.pkl', 'rb') as f:
        sup = pickle.load(f)
    with open(f'{data_dir}/inference_df.pkl', 'rb') as f:
        inf = pickle.load(f)
    with open(f'{data_dir}/output_df.pkl', 'rb') as f:
        out = pickle.load(f)
    return sup, inf, out

# ---------------------------------------------------------------------
import numpy as np

def _jsonify(obj):
    """
    Convert numpy types (and arrays) into native Python types for JSON.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Let the default encoder raise for other types
    raise TypeError(f"Type {type(obj)} not serializable")


# ---------------------------------------------------------------------
def grid_search(rev, data_dir, out_file):
    sup_df, inf_df, out_df = load_data(data_dir)
    results = []
    for name, grid in GRIDS.items():
        for cfg in product_dict(**grid):
            start = time.time()
            try:
                score = eval_one(rev, name, sup_df, inf_df, out_df, **cfg)
            except Exception as e:
                print(f'⚠️  {name} failed on {cfg}: {e}')
                continue
            runtime = time.time() - start
            row = dict(model=name, mse=score, runtime_sec=runtime, **cfg)
            results.append(row)
            print(f'{name} {cfg} → {score:.4f}   ({runtime:.1f}s)')
    # save
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=_jsonify)

# ---------------------------------------------------------------------


import os
import json
import numpy as np
from collections import defaultdict

def aggregate_best_hyperparams(n_trials, results_root):
    # 1) Detect score key and verify 'model' key
    sample_path = os.path.join(results_root, "group_1", "grid_search_results.json")
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"Expected to find sample JSON at {sample_path}")
    with open(sample_path, 'r') as f:
        sample = json.load(f)
    if not isinstance(sample, list) or not sample:
        raise ValueError(f"{sample_path} is empty or not a JSON list")
    sample_keys = set(sample[0].keys())

    # Prioritized list of possible score keys (lower-is-better)
    for candidate in ('mse', 'val_score', 'score', 'error'):
        if candidate in sample_keys:
            score_key = candidate
            break
    else:
        raise KeyError(f"No score key found in sample keys: {sample_keys}")

    model_key = 'model'
    if model_key not in sample_keys:
        raise KeyError(f"No '{model_key}' key found in sample keys: {sample_keys}")

    # Keys to exclude when collecting hyperparameters
    reserved_keys = {model_key, score_key, 'runtime_sec', 'runtime'}

    # 2) Accumulate scores per (model, params-tuple)
    acc = defaultdict(list)
    for i in range(1,94):
        group_dir = os.path.join(results_root, f"group_{i}")
        json_path = os.path.join(group_dir, "grid_search_results.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Missing results file: {json_path}")

        with open(json_path, 'r') as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(f"{json_path} does not contain a JSON list")

        for entry in entries:
            # sanity check
            if model_key not in entry or score_key not in entry:
                raise KeyError(
                    f"Entry missing '{model_key}' or '{score_key}' in {json_path}: {entry}"
                )

            model = entry[model_key]
            score = float(entry[score_key])

            # collect all other keys as hyperparameters
            hp_items = tuple(
                sorted(
                    (k, entry[k])
                    for k in entry.keys()
                    if k not in reserved_keys
                )
            )
            acc[(model, hp_items)].append(score)

    # 3) For each model, pick the hyperparam tuple with lowest average score
    best = {}
    for (model, hp_items), scores in acc.items():
        avg_score = float(np.mean(scores))
        if model not in best or avg_score < best[model][0]:
            best[model] = (avg_score, dict(hp_items))

    return best


if __name__ == '__main__':
    n_trials = 98
    results_root = "result_data_wiki"
    rev = False

    # 1) Optionally re-run your grid_search calls here
    for i in range(1, n_trials+1):
        data_dir = os.path.join(results_root, f"group_{i}")
        out_file = os.path.join(data_dir, "grid_search_results.json")
        grid_search(rev, data_dir, out_file)

    # # 2) Aggregate and pick best
    best_hps = aggregate_best_hyperparams(n_trials, results_root)

    # 3) Print summary
    print("\nBest hyperparameters averaged over all groups:\n")
    for model, (avg_score, params) in best_hps.items():
        print(f"{model:<20}  avg score = {avg_score:.6f}")
        for k, v in params.items():
            print(f"    • {k:15} = {v}")
        print()