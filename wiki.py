import itertools
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from k_means_constrained import KMeansConstrained
from tqdm import tqdm
from scipy.stats import gaussian_kde
import os
from collections import Counter
from baseline import (
    gcn_regression,
    fixmatch_regression,
    laprls_regression,
    tsvr_regression,
    tnnr_regression,
    ucvme_regression,
)

#slience warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import pairwise_distances

def _assign_by_centroids(X, km):
    """Nearest-centroid assignment (avoid KMeansConstrained.predict on small batches)."""
    if len(X) == 0:
        return np.array([], dtype=int)
    centers = km.cluster_centers_
    D = pairwise_distances(X, centers)
    return D.argmin(axis=1)

### Columns of this dataset:
# 'x' - image embedding (numpy array)
# 'y' - text description (string)
# 'yv' - text embedding (numpy array)
# 'z' - page description (string, for generating clusters only)
# 'zv' - page embedding (numpy array, for generating clusters only)


df = pd.read_csv("data/wiki_df.csv")
df['x'] = df['x'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()  # Convert string back to numpy array
df['yv'] = df['yv'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()  # Convert string back to numpy array
df['zv'] = df['zv'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()  # Convert string back to numpy array

X = np.vstack(df['zv'].values)  # shape: (n_samples, embedding_dim)
db = DBSCAN(eps=0.36, min_samples=12, metric='cosine').fit(X)
labels = db.labels_
df = df.assign(cluster=labels)
df_valid = df[df['cluster'] != -1].copy()
cluster_sizes = df_valid['cluster'].value_counts()
eligible = cluster_sizes[cluster_sizes >= 12].index

# define a set of stopwords to ignore
stop_words = {
    'on','in','of','to','for','with','a','an','the','and','or','but',
    'is','are','be','as','by','at','from','that','this','these','those',
    # add more as needed...
}

pruned_clusters = []
for cl in eligible:
    sub = df_valid[df_valid['cluster'] == cl]
    # split z into words, lowercase, filter out stopwords & non-alpha tokens
    word_lists = sub['z'].str.split(',').apply(
        lambda shards: [
            w.lower()
            for shard in shards
            for w in shard.strip().split()
            if w.isalpha() and w.lower() not in stop_words
        ]
    )
    # count only the filtered words
    word_counts = Counter(w for words in word_lists for w in words)
    if not word_counts:
        # no valid words in this cluster
        continue

    most_common_word, count = word_counts.most_common(1)[0]
    print(f"Most common non-stopword in cluster {cl}: {most_common_word} (count: {count})")

    # keep only rows containing that word
    mask = word_lists.apply(lambda words: most_common_word in words)
    pruned = sub[mask]
    if len(pruned) >= 12:
        pruned_clusters.append(pruned)

# concatenate and re-compute eligibility
df_pruned       = pd.concat(pruned_clusters, ignore_index=True)
pruned_counts   = df_pruned['cluster'].value_counts()

def get_data(df, supervised_ratio, output_only_ratio, K=None, seed=None,
             mode="transductive", test_frac=0.2):
    """
    Split each cluster into four disjoint subsets:
      - sup_df:        supervised (used with labels)
      - input_only_df: unlabeled input pool (used only during training)
      - out_df:        output-only pool (for Y-side clustering)
      - test_df:       held-out evaluation set (never used in training)

    If mode == "transductive": test_df == input_only_df (duplicate view).
    If mode == "inductive":    split the remainder into input_only vs test by test_frac.

    Returns:
      sup_df, input_only_df, out_df, test_df, X_for_clustering, Y_for_clustering
      where X_for_clustering = input_only_df ∪ sup_df
            Y_for_clustering = out_df        ∪ sup_df
    """
    rng = np.random.default_rng(seed)

    sup_list, in_only_list, out_list, test_list = [], [], [], []

    for _, group in df.groupby('cluster'):
        n = len(group)
        n_sup = max(1, int(np.floor(supervised_ratio * n)))
        n_out = int(np.floor(output_only_ratio   * n))
        if n_sup + n_out >= n:  # keep at least 1 for remainder
            n_sup = max(1, n - 2)
            n_out = 1
        n_rem = n - n_sup - n_out

        perm = group.sample(frac=1, random_state=int(rng.integers(0,2**32))).reset_index(drop=True)

        sup = perm.iloc[:n_sup]
        out = perm.iloc[n_sup:n_sup+n_out]
        rem = perm.iloc[n_sup+n_out:]  # remainder to be split into input_only/test

        if mode == "transductive":
            in_only = rem
            test    = rem
        else:
            n_test = max(1, int(np.floor(test_frac * len(rem)))) if len(rem) else 0
            test   = rem.iloc[:n_test]
            in_only = rem.iloc[n_test:]

        sup_list.append(sup)
        out_list.append(out)
        in_only_list.append(in_only)
        test_list.append(test)

    sup_df        = pd.concat(sup_list, ignore_index=True)
    input_only_df = pd.concat(in_only_list, ignore_index=True)
    out_df        = pd.concat(out_list, ignore_index=True)
    test_df       = pd.concat(test_list, ignore_index=True)

    # pools for constrained KMeans
    X_for_clustering = pd.concat([input_only_df, sup_df], ignore_index=True)
    Y_for_clustering = pd.concat([out_df,        sup_df], ignore_index=True)

    return sup_df, input_only_df, out_df, test_df, X_for_clustering, Y_for_clustering


def perform_clustering(X_for_clustering, Y_for_clustering, K):
    """
    Perform size‐constrained KMeans clustering on image and gene samples.
    Returns the cluster assignments for X and Y.
    """
    # --- X clustering ---
    X_matrix = np.vstack(X_for_clustering["x"].values)
    n_samples_x = X_matrix.shape[0]
    # enforce roughly equal cluster sizes
    size_min_x = n_samples_x // K
    size_max_x = int(np.ceil(n_samples_x / K))
    x_kmc = KMeansConstrained(
        n_clusters=K,
        size_min=size_min_x,
        size_max=size_max_x,
        random_state=42
    ).fit(X_matrix)
    x_clusters = x_kmc.labels_

    # --- Y clustering ---
    Y_matrix = np.vstack(Y_for_clustering["yv"].values)
    n_samples_y = Y_matrix.shape[0]
    size_min_y = n_samples_y // K
    size_max_y = int(np.ceil(n_samples_y / K))
    y_kmc = KMeansConstrained(
        n_clusters=K,
        size_min=size_min_y,
        size_max=size_max_y,
        random_state=42
    ).fit(Y_matrix)
    y_clusters = y_kmc.labels_

    return x_clusters, y_clusters,x_kmc, y_kmc

def clustering_quality_metrics(X_for_clustering, x_clusters, Y_for_clustering, y_clusters):
    """
    Compute Silhouette Coefficient and Davies–Bouldin Index
    for both input-space and output-space cluster assignments.
    """
    # Prepare feature matrices
    X_matrix = np.vstack(X_for_clustering["x"].values)
    Y_matrix = np.vstack(Y_for_clustering["yv"].values)
    
    # Input-space clustering metrics
    sil_x = silhouette_score(X_matrix, x_clusters)
    db_x  = davies_bouldin_score(X_matrix, x_clusters)
    
    # Output-space clustering metrics
    sil_y = silhouette_score(Y_matrix, y_clusters)
    db_y  = davies_bouldin_score(Y_matrix, y_clusters)
    
    return {
        "input_silhouette": sil_x,
        "input_davies_bouldin": db_x,
        "output_silhouette": sil_y,
        "output_davies_bouldin": db_y
    }

def decisionVector(sample, x_column, y_column, dim=5):

    # Check if the specified columns exist in the DataFrame
    if x_column not in sample.columns:
        raise KeyError(f"Column '{x_column}' not found in the DataFrame.")
    if y_column not in sample.columns:
        raise KeyError(f"Column '{y_column}' not found in the DataFrame.")

    # Create association matrix
    association_matrix = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            association_matrix[i, j] = np.sum((sample[x_column] == i) & (sample[y_column] == j))
    
    # Initialize decision array (this could be improved based on specific logic for decision making)
    decision = np.zeros(dim, dtype=int)
    
    for i in range(dim):
        decision[i] = np.argmax(association_matrix[i, :])

    return decision

def build_decision_matrix(supervised_samples, x_clusters, y_clusters, K):
    """
    Build the decision matrix (association vector) using the supervised samples.
    """
    N_sup = len(supervised_samples)
    supervised_samples['x_cluster'] = x_clusters[-N_sup:]
    supervised_samples['y_cluster'] = y_clusters[-N_sup:]
    
    decision_matrix = decisionVector(supervised_samples, x_column='x_cluster', y_column='y_cluster', dim=K)

    return decision_matrix


def build_true_decision_vector(Xc, Yc, x_clusters, y_clusters, K):
    """
    Build an oracle decision vector for Script 1 by majority‐voting
    over the *entire* Xc/Yc (inference+supervised and output-only+supervised).

    Xc           : DataFrame (must include original df['cluster'])
    Yc           : DataFrame (must include original df['cluster'])
    x_clusters   : array-like of length len(Xc)
    y_clusters   : array-like of length len(Yc)
    K            : number of bridged clusters

    Returns
    -------
    true_vec : np.ndarray, shape (K,)
               For each bridged image-cluster i, the y-cluster whose majority
               z-cluster matches i’s majority z-cluster.
    """
    # attach the bridged assignments
    Xc2 = Xc.copy()
    Xc2['x_cluster'] = x_clusters
    Yc2 = Yc.copy()
    Yc2['y_cluster'] = y_clusters

    # 1) image_cluster → majority original z-cluster
    image_to_z = (
        Xc2
        .groupby('x_cluster')['cluster']
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    # 2) y_cluster → majority original z-cluster
    y_to_z = (
        Yc2
        .groupby('y_cluster')['cluster']
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    # 3) invert y_to_z → z_to_y
    z_to_y = { z: y for y, z in y_to_z.items() }

    # 4) assemble the vector
    true_vec = np.full(K, -1, dtype=int)
    for i in range(K):
        z = image_to_z.get(i)
        if z is not None:
            true_vec[i] = z_to_y.get(z, -1)

    return true_vec

def compute_y_centroids(Y_for_clustering, y_clusters, K):
    Y = Y_for_clustering.copy()
    Y['y_cluster'] = y_clusters
    centroids, text_prototypes = [], []
    for c in range(K):
        cluster_data = Y[Y['y_cluster'] == c]
        if len(cluster_data) > 0:
            yvs = np.stack(cluster_data['yv'].values)
            centroid = np.mean(yvs, axis=0)
            dists = np.linalg.norm(yvs - centroid, axis=1)
            idx = np.argmin(dists)
            prototype_text = cluster_data['y'].values[idx]
        else:
            centroid = np.zeros(Y['yv'].iloc[0].shape)
            prototype_text = ""
        centroids.append(centroid)
        text_prototypes.append(prototype_text)
    return np.array(centroids), text_prototypes

def perform_inference(inference_samples, image_clusters, decision_matrix, centroids, text_prototypes):
    inf = inference_samples.copy()
    inf['x_cluster'] = image_clusters[:len(inf)]
    inf['predicted_y_cluster'] = inf['x_cluster'].apply(lambda x: decision_matrix[x])
    inf['predicted_yv'] = inf['predicted_y_cluster'].apply(lambda j: centroids[j])
    inf['predicted_text'] = inf['predicted_y_cluster'].apply(lambda j: text_prototypes[j])
    return inf

def knn_regression(supervised_df, inference_df, n_neighbors=10):
    X_train = np.vstack(supervised_df['x'].values)
    y_train = np.vstack(supervised_df['yv'].values)
    texts_train = supervised_df['y'].values.tolist()
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    X_test = np.vstack(inference_df['x'].values)
    pred_emb = knn.predict(X_test)
    act_emb = np.vstack(inference_df['yv'].values)
    actual_texts = inference_df['y'].tolist()
    pred_texts = []
    for emb in pred_emb:
        dists = np.linalg.norm(y_train - emb, axis=1)
        idx = np.argmin(dists)
        pred_texts.append(texts_train[idx])
    return pred_emb, act_emb, pred_texts, actual_texts

def evaluate_regression_loss(predictions, actuals):
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    return mae, mse

def _wrap_baseline(baseline_fn, sup_df, input_only_df, test_df):
    """
    Call baseline_fn(supervised, inference) on embeddings, then
    recover actual_texts and predicted_texts via nearest‐neighbor lookup
    in sup_df['y'].
    """
    # 1) rename to Script 0’s expected columns
    sup = sup_df.rename(columns={'x':'morph_coordinates', 'yv':'gene_coordinates'})
    ino = input_only_df.rename(columns={'x':'morph_coordinates', 'yv':'gene_coordinates'})
    tst = test_df.rename(columns={'x':'morph_coordinates', 'yv':'gene_coordinates'})

    # 2) run the numeric regressor

    if baseline_fn.__name__ == 'gcn_regression':
        preds, actuals = gcn_regression       (sup, ino, tst,dropout=0.1, hidden=32,lr=0.001)
    elif baseline_fn.__name__ == 'fixmatch_regression':
        preds, actuals = fixmatch_regression  (sup, ino, tst,alpha_ema=0.999,batch_size=64,conf_threshold=0.1,lambda_u_max=0.5,lr=0.0003,rampup_length=30)
    elif baseline_fn.__name__ == 'laprls_regression':
        preds, actuals = laprls_regression    (sup, ino, tst,gamma=0.001,k=5,lam=0.1,sigma=2.0)
    elif baseline_fn.__name__ == 'tsvr_regression':
        preds, actuals = tsvr_regression      (sup, ino, tst, C=10, epsilon=0.01, gamma='scale', self_training_frac=0.1)
    elif baseline_fn.__name__ == 'tnnr_regression':
        preds, actuals = tnnr_regression     (sup, ino, tst, beta=0.1,lr=0.001, rep_dim=128)
    elif baseline_fn.__name__ == 'ucvme_regression':
        preds, actuals = ucvme_regression    (sup, ino, tst,lr=0.001,mc_T=5,w_unl=10)
    else:
        raise ValueError(f"Unknown baseline function: {baseline_fn.__name__}")

    return preds, actuals


##### KMM Section

from baseline import kernel_mean_matching_regression, reversed_kernel_mean_matching_regression
from baseline import em_regression, reversed_em_regression
from baseline import eot_barycentric_regression, reversed_eot_barycentric_regression
from baseline import gw_metric_alignment_regression, reversed_gw_metric_alignment_regression


def run_experiment(
    df: pd.DataFrame,
    supervised_ratio: float = 0.05,
    output_only_ratio: float = 0.5,
    K: int = 100,
    knn_neighbors: int = 10,
    seed: int = None,
    mode: str = "transductive"  # or "inductive"
):
    # 1) split
    sup_df, input_only_df, out_df, test_df, Xc, Yc = get_data(
        df, supervised_ratio, output_only_ratio, K, seed,
        mode=mode,  # or expose via CLI if you like
        test_frac=0.2
    )

    # 2) clustering
    x_cl, y_cl, x_km, y_km = perform_clustering(Xc, Yc, K)

    # 3) clustering quality
    ami_x = adjusted_mutual_info_score(Xc["cluster"], x_cl)
    ami_y = adjusted_mutual_info_score(Yc["cluster"], y_cl)
    decision = build_decision_matrix(sup_df, x_cl, y_cl, K)
    true_decision = build_true_decision_vector(Xc, Yc, x_cl, y_cl, K)
    accuracy = np.mean(decision == true_decision)

    # 4) bridged inference
    cents, texts = compute_y_centroids(Yc, y_cl, K)
    # inf_res = perform_inference(inf_df, x_cl, decision, cents, texts)
    test_x = np.vstack(test_df["x"].values) if len(test_df) else np.zeros((0, np.vstack(Xc["x"].values).shape[1]))
    test_x_clusters = _assign_by_centroids(test_x, x_km)
    inf_res = perform_inference(test_df, test_x_clusters, decision, cents, texts)


    # 5) BKM baseline
    bkm_pred_emb = np.vstack(inf_res["predicted_yv"].values)
    bkm_act_emb  = np.vstack(inf_res["yv"].values)
    bkm_mae, bkm_mse = evaluate_regression_loss(bkm_pred_emb, bkm_act_emb)

    # 6) KNN baseline
    knn_pred_emb, knn_act_emb, knn_pred_texts, knn_act_texts = knn_regression(sup_df, test_df, knn_neighbors)
    knn_mae, knn_mse = evaluate_regression_loss(knn_pred_emb, knn_act_emb)

    # 7) _wrap_baseline methods
    fm_pred, fm_act = _wrap_baseline(fixmatch_regression, sup_df, input_only_df, test_df)
    lap_pred, lap_act = _wrap_baseline(laprls_regression, sup_df, input_only_df, test_df)
    tsvr_pred, tsvr_act = _wrap_baseline(tsvr_regression, sup_df, input_only_df, test_df)
    tnnr_pred, tnnr_act = _wrap_baseline(tnnr_regression, sup_df, input_only_df, test_df)
    ucv_pred, ucv_act = _wrap_baseline(ucvme_regression, sup_df, input_only_df, test_df)
    gcn_pred, gcn_act = _wrap_baseline(gcn_regression, sup_df, input_only_df, test_df)

    fm_mae, fm_mse       = evaluate_regression_loss(fm_pred, fm_act)
    lap_mae, lap_mse     = evaluate_regression_loss(lap_pred, lap_act)
    tsvr_mae, tsvr_mse   = evaluate_regression_loss(tsvr_pred, tsvr_act)
    tnnr_mae, tnnr_mse   = evaluate_regression_loss(tnnr_pred, tnnr_act)
    ucv_mae, ucv_mse     = evaluate_regression_loss(ucv_pred, ucv_act)
    gcn_mae, gcn_mse     = evaluate_regression_loss(gcn_pred, gcn_act)

    # 8) KMM forward on full marginals
    X_kmm = pd.concat([input_only_df, sup_df], ignore_index=True).rename(columns={'x':'morph_coordinates'})
    Y_kmm = pd.concat([out_df, sup_df], ignore_index=True).rename(columns={'yv':'gene_coordinates'})
    sup_kmm = sup_df.rename(columns={'x':'morph_coordinates','yv':'gene_coordinates'})
    inf_kmm = test_df.rename(columns={'x':'morph_coordinates','yv':'gene_coordinates'})
    kmm_pred_emb, kmm_act_emb = kernel_mean_matching_regression(
        image_df      = X_kmm,
        gene_df       = Y_kmm,
        supervised_df = sup_kmm,
        inference_df  = inf_kmm,
        alpha=0.01,
        kmm_B=100,
        kmm_eps=0.001,
        sigma=0.5
    )
    kmm_mae, kmm_mse = evaluate_regression_loss(kmm_pred_emb, kmm_act_emb)

    # 9) EM forward on full marginals
    em_pred_emb, em_act_emb = em_regression(
        supervised_df = sup_kmm,
        image_df      = X_kmm,
        gene_df       = Y_kmm,
        inference_df  = inf_kmm,
        n_components  = K,
        eps=0.001,
        max_iter= 2000,
        tol= 1e-4
    )
    em_mae, em_mse = evaluate_regression_loss(em_pred_emb, em_act_emb)

    # 10) EOT Barycentric Regression
    eot_pred_emb, eot_act_emb = eot_barycentric_regression(
        supervised_df = sup_kmm,
        image_df      = X_kmm,
        gene_df       = Y_kmm,
        inference_df  = inf_kmm,
        max_iter        = 2000,
        eps             = 10,
        ridge_alpha     = 0.1,
        tol             = 1e-09
    )
    eot_mae, eot_mse = evaluate_regression_loss(eot_pred_emb, eot_act_emb)

    # 11) GW Metric Alignment Regression
    gw_pred_emb, gw_act_emb = gw_metric_alignment_regression(
        supervised_df = sup_kmm,
        image_df      = X_kmm,
        gene_df       = Y_kmm,
        inference_df  = inf_kmm,
        max_iter        = 2000,
        tol             = 1e-07
    )
    gw_mae, gw_mse = evaluate_regression_loss(gw_pred_emb, gw_act_emb)

    # 10) package metrics
    metrics = {
        'clustering': {
            'AMI_X': ami_x,
            'AMI_Y': ami_y,
            'Bridging Accuracy': accuracy
        },
        'regression': {
            'BKM':        {'MAE': bkm_mae,  'MSE': bkm_mse},
            'KNN':        {'MAE': knn_mae,  'MSE': knn_mse},
            'FixMatch':   {'MAE': fm_mae,   'MSE': fm_mse},
            'LapRLS':     {'MAE': lap_mae,  'MSE': lap_mse},
            'TSVR':       {'MAE': tsvr_mae,'MSE': tsvr_mse},
            'TNNR':       {'MAE': tnnr_mae,'MSE': tnnr_mse},
            'UCVME':      {'MAE': ucv_mae, 'MSE': ucv_mse},
            'GCN':        {'MAE': gcn_mae, 'MSE': gcn_mse},
            'KMM':        {'MAE': kmm_mae, 'MSE': kmm_mse},
            'EM':         {'MAE': em_mae,  'MSE': em_mse},
            'EOT':        {'MAE': eot_mae, 'MSE': eot_mse},
            'GW':         {'MAE': gw_mae,  'MSE': gw_mse}
        },
    }
    # print mse
    print("MSE Scores:")
    mse_scores = {k: v['MSE'] for k, v in metrics['regression'].items()}
    print(mse_scores)
    return metrics


def run_reversed_experiment(
    df: pd.DataFrame,
    supervised_ratio: float = 0.05,
    output_only_ratio: float = 0.5,
    K: int = 100,
    knn_neighbors: int = 10,
    seed: int = None,
    mode: str = "transductive"  # or "inductive"
):
    """
    Mirror of run_experiment
    """
    from collections import Counter

    # 1) split
    sup_df, input_only_df, out_df, test_df, Xc_rev, Yc_rev = get_data(
        df, supervised_ratio, output_only_ratio, K, seed,
        mode=mode, test_frac=0.2
    )

    # 3) cluster on text (yv)
    T = np.vstack(Xc_rev["yv"].values)
    size_min_t = len(T) // K
    size_max_t = int(np.ceil(len(T) / K))
    km_text = KMeansConstrained(n_clusters=K, size_min=size_min_t, size_max=size_max_t, random_state=42).fit(T)
    text_clusters = km_text.labels_

    # 4) cluster on image (x)
    I = np.vstack(Yc_rev["x"].values)
    size_min_i = len(I) // K
    size_max_i = int(np.ceil(len(I) / K))
    km_img = KMeansConstrained(n_clusters=K, size_min=size_min_i, size_max=size_max_i, random_state=42).fit(I)
    img_clusters = km_img.labels_

    # 5) cluster quality
    ami_text  = adjusted_mutual_info_score(Xc_rev["cluster"], text_clusters)
    ami_image = adjusted_mutual_info_score(Yc_rev["cluster"], img_clusters)

    # 6) build text→image decision vector
    sup_block = sup_df.copy()
    sup_block["text_cluster"] = text_clusters[-len(sup_df):]
    sup_block["img_cluster"]  = img_clusters[-len(sup_df):]
    decision_vec = decisionVector(sup_block, "text_cluster", "img_cluster", dim=K)

    # 7) oracle for reversed mapping
    true_vec  = build_true_decision_vector(Xc_rev, Yc_rev, text_clusters, img_clusters, K)
    oracle_rev = np.full(K, -1, dtype=int)
    for img_c, txt_c in enumerate(true_vec):
        if txt_c >= 0:
            oracle_rev[txt_c] = img_c
    decision_acc = (decision_vec == oracle_rev).mean()

    # 8) compute image‐cluster centroids
    Yc_rev = Yc_rev.copy()
    Yc_rev["img_cluster"] = img_clusters
    Yc_rev["x"]           = np.vstack(Yc_rev["x"].values).tolist()
    img_cents = []
    for c in range(K):
        pts = (
            np.stack(Yc_rev[Yc_rev["img_cluster"] == c]["x"].values)
            if (Yc_rev["img_cluster"] == c).any()
            else np.zeros(I.shape[1])
        )
        img_cents.append(pts.mean(axis=0))
    img_cents = np.vstack(img_cents)

    if len(test_df):
        T_test = np.vstack(test_df["yv"].values)
        centers_t = km_text.cluster_centers_
        D_t = pairwise_distances(T_test, centers_t)
        test_text_clusters = D_t.argmin(axis=1)
    else:
        test_text_clusters = np.array([], dtype=int)

    # 9) bridged‐reversed inference
    inf = test_df.copy()
    inf["text_cluster"]             = test_text_clusters
    inf["predicted_img_cluster"]    = inf["text_cluster"].map(lambda t: decision_vec[t] if t >= 0 else -1)
    inf["pred_x"]                   = inf["predicted_img_cluster"].map(lambda c: img_cents[c] if c >= 0 else np.zeros(img_cents.shape[1]))
    bridged_preds  = np.vstack(inf["pred_x"].values) if len(inf) else np.zeros((0, img_cents.shape[1]))
    bridged_actual = np.vstack(inf["x"].values)       if len(inf) else np.zeros((0, img_cents.shape[1]))

    # 10) prepare marginals for KMM & EM
    sup_rev = sup_df.rename(columns={'yv': 'morph_coordinates', 'x': 'gene_coordinates'}).copy()
    ino_rev = input_only_df.rename(columns={'yv': 'morph_coordinates', 'x': 'gene_coordinates'}).copy()
    tst_rev = test_df.rename(columns={'yv': 'morph_coordinates', 'x': 'gene_coordinates'}).copy()

    if len(sup_rev) and len(tst_rev):
        knn = KNeighborsRegressor(n_neighbors=knn_neighbors)
        knn.fit(np.vstack(sup_rev["morph_coordinates"]), np.vstack(sup_rev["gene_coordinates"]))
        knn_preds = knn.predict(np.vstack(tst_rev["morph_coordinates"]))
    else:
        knn_preds = np.zeros((0, np.vstack(sup_rev["gene_coordinates"]).shape[1] if len(sup_rev) else 0))
    y_te = np.vstack(tst_rev["gene_coordinates"]) if len(tst_rev) else np.zeros_like(knn_preds)

    gc_preds, gc_actuals = gcn_regression       (sup_rev, ino_rev, tst_rev, hidden=32, dropout=0.1, lr=0.001)
    fx_preds, fx_actuals = fixmatch_regression  (sup_rev, ino_rev, tst_rev, alpha_ema=0.999, batch_size=32, conf_threshold=0.05, lambda_u_max=0.5, lr=3e-4, rampup_length=10)
    lp_preds, lp_actuals = laprls_regression    (sup_rev, ino_rev, tst_rev, gamma=0.1, k=20, lam=0.001, sigma=2.0)
    ts_preds, ts_actuals = tsvr_regression      (sup_rev, ino_rev, tst_rev, C=10, epsilon=0.01, gamma='scale', self_training_frac=0.5)
    tn_preds, tn_actuals = tnnr_regression      (sup_rev, ino_rev, tst_rev, beta=1.0, lr=0.001, rep_dim=128)
    uv_preds, uv_actuals = ucvme_regression     (sup_rev, ino_rev, tst_rev, lr=3e-4, mc_T=5, w_unl=1.0)

    gene_df_rev  = Xc_rev.rename(columns={'yv': 'gene_coordinates'}).copy()   # text side
    image_df_rev = Yc_rev.rename(columns={'x':  'morph_coordinates'}).copy()  # image side
    sup_rev_reg  = sup_df.rename(columns={'yv': 'gene_coordinates', 'x': 'morph_coordinates'}).copy()
    tst_rev_reg  = test_df.rename(columns={'yv': 'gene_coordinates', 'x': 'morph_coordinates'}).copy()

    # ── reversed KMM ──
    kmm_rev_pred_emb, kmm_rev_act_emb = reversed_kernel_mean_matching_regression(
        gene_df       = gene_df_rev,
        image_df      = image_df_rev,
        supervised_df = sup_rev_reg,
        inference_df  = tst_rev_reg,
        random_state  = 0,
        alpha=0.01,
        kmm_B=100,
        kmm_eps=0.001,
        sigma=0.5
    )

    # ── reversed EM ──
    em_rev_pred_emb, em_rev_act_emb = reversed_em_regression(
        gene_df       = gene_df_rev,
        image_df      = image_df_rev,
        supervised_df = sup_rev_reg,
        inference_df  = tst_rev_reg,
        n_components  = K,
        eps=0.001,
        max_iter= 2000,
        tol= 1e-4
    )

    # ── EOT Barycentric Regression ──
    eot_rev_pred_emb, eot_rev_act_emb = reversed_eot_barycentric_regression(
        gene_df       = gene_df_rev,
        image_df      = image_df_rev,
        supervised_df = sup_rev_reg,
        inference_df  = tst_rev_reg,
        max_iter        = 2000,
        ridge_alpha     = 0.01,
        tol             = 1e-09
    )
    # ── GW Metric Alignment Regression ──
    gw_rev_pred_emb, gw_rev_act_emb = reversed_gw_metric_alignment_regression(
        gene_df       = gene_df_rev,
        image_df      = image_df_rev,
        supervised_df = sup_rev_reg,
        inference_df  = tst_rev_reg,
        max_iter        = 2000,
        tol             = 1e-07
    )

    # 12) collect errors
    def eval_(p, a):
        return mean_absolute_error(a, p), mean_squared_error(a, p)

    errors = {}
    mses   = {}

    errors["BKM"],         mses["BKM"]         = eval_(bridged_preds,  bridged_actual)
    errors["KNN"],         mses["KNN"]         = eval_(knn_preds,      y_te)
    errors["GCN"],         mses["GCN"]         = eval_(gc_preds,       gc_actuals)
    errors["FixMatch"],    mses["FixMatch"]    = eval_(fx_preds,       fx_actuals)
    errors["LapRLS"],      mses["LapRLS"]      = eval_(lp_preds,       lp_actuals)
    errors["TSVR"],        mses["TSVR"]        = eval_(ts_preds,       ts_actuals)
    errors["TNNR"],        mses["TNNR"]        = eval_(tn_preds,       tn_actuals)
    errors["UCVME"],       mses["UCVME"]       = eval_(uv_preds,       uv_actuals)
    errors["KMM"],         mses["KMM"]         = eval_(kmm_rev_pred_emb, kmm_rev_act_emb)
    errors["EM"],          mses["EM"]          = eval_(em_rev_pred_emb,  em_rev_act_emb)
    errors["EOT"],         mses["EOT"]         = eval_(eot_rev_pred_emb, eot_rev_act_emb)
    errors["GW"],          mses["GW"]          = eval_(gw_rev_pred_emb,  gw_rev_act_emb)

    #print errors per model
    print("Errors per model:")
    for model, error in errors.items():
        print(f"{model}: MAE={error:.4f}, MSE={mses[model]:.4f}")

    return {
        "clustering": {
            "AMI_X":           ami_text,
            "AMI_Y":          ami_image,
            "Bridging Accuracy":  decision_acc,
        },
        "regression": {
            "BKM":        {"MAE": errors["BKM"],   "MSE": mses["BKM"]},
            "KNN":        {"MAE": errors["KNN"],   "MSE": mses["KNN"]},
            "GCN":        {"MAE": errors["GCN"],   "MSE": mses["GCN"]},
            "FixMatch":   {"MAE": errors["FixMatch"],  "MSE": mses["FixMatch"]},
            "LapRLS":     {"MAE": errors["LapRLS"],   "MSE": mses["LapRLS"]},
            "TSVR":       {"MAE": errors["TSVR"],   "MSE": mses["TSVR"]},
            "TNNR":       {"MAE": errors["TNNR"],   "MSE": mses["TNNR"]},
            "UCVME":      {"MAE": errors["UCVME"],  "MSE": mses["UCVME"]},
            "KMM":        {"MAE": errors["KMM"],   "MSE": mses["KMM"]},
            "EM":         {"MAE": errors["EM"],    "MSE": mses["EM"]},
            "EOT":        {"MAE": errors["EOT"],   "MSE": mses["EOT"]},
            "GW":         {"MAE": errors["GW"],    "MSE": mses["GW"]},
        },
    }



if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser(
        description="Run experiments (forward or reversed)"
    )
    parser.add_argument('--mode',
                    choices=['transductive', 'inductive'],
                    default='transductive',
                help='Data usage mode. In inductive, a held-out test split is carved from inference.')
    parser.add_argument("--reversed", action="store_true",
                        help="Run text→image (reversed) pipeline instead")
    args = parser.parse_args()
    runner = run_reversed_experiment if args.reversed else run_experiment
    pre_key = "304_wiki_reversed" if args.reversed else "303_wiki"
    experiment_key = f"{pre_key}_tran" if args.mode == "transductive" else f"{pre_key}_ind"


    # Experiment grid
    K_values   = [3, 4, 5, 6, 7]
    sup_values = [1, 2, 3, 4]            # sup_per_cluster
    out_only   = 0.2
    cluster_sz = 25
    seeds = list(range(30))

    eligible_pruned = pruned_counts[pruned_counts >= cluster_sz].index

    print(f"Eligible clusters are those: {eligible_pruned.tolist()}")

    models = [
      'BKM', 'KNN',
      'GCN', 'FixMatch',
      'LapRLS', 'TSVR', 'TNNR', 'UCVME',
      'KMM', 'EM', 'EOT', 'GW'
    ]
    nK      = len(K_values)
    nSup    = len(sup_values)
    nModels = len(models)
    nTrials = len(seeds)

    # Make results folder
    os.makedirs("results", exist_ok=True)

    # Preallocate arrays
    ami_x     = np.empty((nK, nSup, nTrials))
    ami_y     = np.empty((nK, nSup, nTrials))
    accuracy  = np.empty((nK, nSup, nTrials))

    mae       = np.empty((nK, nSup, nModels, nTrials))
    mse       = np.empty((nK, nSup, nModels, nTrials))

    # Main loops
    for i, K in enumerate(K_values):
        for j, sup_per in enumerate(sup_values):
            for t, s in enumerate(seeds):
                seed = s + i * nK + j * nK * nSup  # unique seed per (K, sup_per, trial)
                # ── sample K clusters from pruned set ──
                            # initialize a per-trial RNG
                rng = np.random.default_rng(seed)

                # choose clusters
                chosen = rng.choice(eligible_pruned, size=K, replace=False)

                # sample each cluster independently
                samples = []
                for c in chosen:
                    sub = df_pruned[df_pruned['cluster'] == c]
                    subseed = int(rng.integers(0, 2**32))
                    samples.append(sub.sample(cluster_sz, random_state=subseed))
                sample = pd.concat(samples, ignore_index=True)

                # ── 2) run holistic experiment ──
                metrics = runner(
                    sample,
                    supervised_ratio   = sup_per/cluster_sz,
                    output_only_ratio  = out_only,
                    K                   = K,
                    knn_neighbors      = sup_per,
                    seed                = seed
                )

                # store clustering AMI
                ami_x[i, j, t] = metrics['clustering']['AMI_X']
                ami_y[i, j, t] = metrics['clustering']['AMI_Y']
                accuracy[i, j, t] = metrics['clustering']['Bridging Accuracy']

                # store regression & text metrics for each model
                for m_idx, m in enumerate(models):
                    reg = metrics['regression'][m]
                    txt = metrics['text'][m] if 'text' in metrics else {}
                    mae [i, j, m_idx, t] = reg['MAE']
                    mse [i, j, m_idx, t] = reg['MSE']

            print(f"Finished K={K}, sup={sup_per}")


    os.makedirs(f'results/{experiment_key}', exist_ok=True)
    np.save(f"results/{experiment_key}/ami_x.npy", ami_x)
    np.save(f"results/{experiment_key}/ami_y.npy", ami_y)
    np.save(f"results/{experiment_key}/accuracy.npy", accuracy)
    np.save(f"results/{experiment_key}/mae.npy", mae)
    np.save(f"results/{experiment_key}/mse.npy", mse)

    print("All done! Results are in the `results/` folder.")
