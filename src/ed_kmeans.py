import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.datasets import load_iris
import math

eps = 1e-9

def scale_bpa(X):
    s = MinMaxScaler()
    Xs = s.fit_transform(X)
    sums = Xs.sum(axis=1, keepdims=True)
    M = Xs / (sums+eps)
    return M, Xs, s

def build_D(Xs):
    n = Xs.shape[1]
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            num = np.sum(np.minimum(Xs[:,i], Xs[:,j]))
            den = np.sum(np.maximum(Xs[:,i], Xs[:,j]))+eps
            D[i,j] = num/den
    return D

def edist(m1,m2,D):
    d = m1-m2
    return math.sqrt(0.5 * d.dot(D).dot(d))

class EDKMeans:
    def __init__(self, k=3, max_iter=100, tol=1e-4, seed=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        M, Xs, self.scaler = scale_bpa(X)
        self.D = build_D(Xs)
        n = X.shape[0]
        idx = rng.choice(n, self.k, replace=False)
        centers = X[idx].copy()
        labs = np.zeros(n, dtype=int)

        for it in range(self.max_iter):
            if self.scaler is not None:
                cs = self.scaler.transform(centers)
                Mc = cs / (cs.sum(axis=1, keepdims=True)+eps)
            else:
                Mc = centers

            new_labs = np.zeros(n, dtype=int)
            for i in range(n):
                dists = [edist(M[i], Mc[j], self.D) for j in range(self.k)]
                new_labs[i] = np.argmin(dists)

            new_centers = np.zeros_like(centers)
            for j in range(self.k):
                pts = X[new_labs==j]
                if len(pts)>0:
                    new_centers[j] = pts.mean(axis=0)
                else:
                    new_centers[j] = X[rng.randint(0,n)]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            labs = new_labs
            if shift < self.tol: break

        self.labels_ = labs
        self.cluster_centers_ = centers
        self.n_iter_ = it+1
        return self

    def predict(self,X):
        Xs = self.scaler.transform(X)
        M = Xs / (Xs.sum(axis=1, keepdims=True)+eps)
        cs = self.scaler.transform(self.cluster_centers_)
        Mc = cs / (cs.sum(axis=1, keepdims=True)+eps)
        out = []
        for i in range(X.shape[0]):
            dists = [edist(M[i], Mc[j], self.D) for j in range(self.k)]
            out.append(np.argmin(dists))
        return np.array(out)

def run_edkmeans():
    X,y = load_iris(return_X_y=True)
    ed = EDKMeans(k=3, seed=0)
    ed.fit(X)
    ari = adjusted_rand_score(y, ed.labels_)
    sil = silhouette_score(X, ed.labels_)
    print("ED-KMeans")
    print("ARI:", round(ari,4), "Silhouette:", round(sil,4), "Iterations:", ed.n_iter_)

run_edkmeans()
# ed_kmeans.py
