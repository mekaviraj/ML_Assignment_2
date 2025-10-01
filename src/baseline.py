import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.datasets import load_iris

def run_baseline():
    X, y = load_iris(return_X_y=True)
    k = 3
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    y_pred = km.fit_predict(X)

    ari = adjusted_rand_score(y, y_pred)
    sil = silhouette_score(X, y_pred)

    print("Baseline KMeans")
    print("ARI:", round(ari,4), "Silhouette:", round(sil,4), "Iterations:", km.n_iter_)


run_baseline()
# baseline.py
