import itertools
import inspect
from dataclasses import dataclass, replace
from enum import Enum, member
from typing import Callable
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from ClusterValid import ClusterIndexCalculator
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import eigh
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import LabelEncoder

ClusterValid_indexes = [
    "ball_hall",  # max diff
    "davies_bouldin",  # min
    "silhouette",  # max
    "c_index",  # max
    "dunn",  # max
    "calinski_harabasz",  # first relative maxima
    "friedman1",  # min send
    "mariott",
    "scott2",
    "friedman2",
    "tau",
    "gamma",
    "gplus",
    "tracew",
    "mcclain",
    "ptbiserial",
    "ratkowsky",
    "trcovw",
    "sd_dis",
    "sd_scat",
    "sdbw",
    "ccc",
    "dindex",
    "xie_beni",
    "pbm",
    "sf1",
    "sf2",
]


class BehaviorEnum(Enum):
    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)

    def __repr__(self):
        args = inspect.signature(self.value).format()
        return "<%s.%s%s>" % (self.__class__.__name__, self._name_, args)


class OptimumRule(BehaviorEnum):
    @member
    def MAX(scores):
        return np.argmax(scores)

    @member
    def MIN(scores):
        return np.argmin(scores)

    @member
    def FIRST_MAX(scores):
        eps = 1e-10
        scores = np.asarray(scores)
        scores = np.r_[scores[0] - eps, scores, scores[-1] - eps]
        peaks = find_peaks(scores)[0] - 1
        return peaks[0]

    @member
    def FIRST_MIN(scores):
        eps = 1e-10
        scores = -1 * np.asarray(scores)
        scores = np.r_[scores[0] - eps, scores, scores[-1] - eps]
        peaks = find_peaks(scores)[0] - 1
        return peaks[0]

    @member
    def MAX_RATIO(scores):
        eps = 1e-12
        noise = np.random.uniform(eps, eps*10, size=len(scores))
        scores = np.asarray(scores) + noise
        ratios = (scores[1:-1] - scores[:-2]) / (scores[2:] - scores[1:-1])
        return np.argmax(ratios) + 1

    @member
    def MAX_LEFT_DIFF(scores):
        return np.diff(scores, prepend=np.inf).argmax()

    @member
    def MAX_RIGHT_DIFF(scores):
        return np.diff(scores, append=-np.inf).argmax()

    @member
    def MIN_LEFT_DIFF(scores):
        return np.diff(scores, prepend=-np.inf).argmin()

    @member
    def MIN_RIGHT_DIFF(scores):
        return np.diff(scores, append=np.inf).argmin()

type Scorer = Callable[[npt.ArrayLike], int]
type OptimumFinder = OptimumRule | Callable[[pd.DataFrame, npt.ArrayLike], float]

@dataclass
class InternalIndex:
    scorer: Scorer
    optimum: OptimumFinder

    def __init__(
        self, 
        scorer: Scorer | str, 
        optimum: OptimumFinder | str
    ):
        if isinstance(scorer, str):
            scorer = partial(get_index, index=scorer)
        self.scorer = scorer
        if isinstance(optimum, str):
            optimum = OptimumRule[optimum.upper()]
        self.optimum = optimum

    def __call__(self, X, labels):
        return self.scorer(X, labels)

    def finite_difference_mod(self, direction):
        left_trans = {
            OptimumRule.MAX: OptimumRule.MAX_LEFT_DIFF,
            OptimumRule.MIN: OptimumRule.MIN_LEFT_DIFF,
            OptimumRule.FIRST_MAX: OptimumRule.MAX_LEFT_DIFF,
            OptimumRule.FIRST_MIN: OptimumRule.MIN_LEFT_DIFF,
        }
        right_trans = {
            OptimumRule.MAX: OptimumRule.MIN_RIGHT_DIFF,
            OptimumRule.MIN: OptimumRule.MAX_RIGHT_DIFF,
            OptimumRule.FIRST_MAX: OptimumRule.MIN_RIGHT_DIFF,
            OptimumRule.FIRST_MIN: OptimumRule.MAX_RIGHT_DIFF,
        }
        if direction == "left":
            diff_optimum = left_trans.get(self.optimum)
        elif direction == "right":
            diff_optimum = right_trans.get(self.optimum)
        else:
            raise ValueError(f"invalid option {direction!r}")

        if diff_optimum is None:
            raise ValueError(
                f"only the following rules can be modified: {list(left_trans)}"
            )

        return replace(self, optimum=diff_optimum)

    @property
    def ldiff(self):
        return self.finite_difference_mod("left")
    
    @property
    def rdiff(self):
        return self.finite_difference_mod("right")


def fix_labels(labels):
    labels = np.asarray(labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_classes = len(le.classes_)
    return labels, n_classes


def plot_clustered_corrmat(
    cmat, labels=None, viz=True, ticklabels=False, figsize=(8, 6)
):
    cmat = np.asarray(cmat)
    if labels is None:
        labels = np.arange(cmat.shape[0])
    else:
        labels, _ = fix_labels(labels)
    order = labels.argsort()
    grouped_cmat = pd.DataFrame(cmat[order][:, order])
    if viz:
        class_viz = labels[order].copy()
        class_viz[class_viz % 2 == 1] = 1
        class_viz[class_viz % 2 == 0] = -1
        grouped_cmat.insert(0, "#2", class_viz)
        grouped_cmat.insert(0, "#1", class_viz)
        grouped_cmat = grouped_cmat.T
        class_viz = np.r_[[-1, -1], class_viz]
        grouped_cmat.insert(0, "#2", class_viz)
        grouped_cmat.insert(0, "#1", class_viz)
    plt.figure(figsize=figsize)
    sns.heatmap(
        grouped_cmat, xticklabels=ticklabels, yticklabels=ticklabels, vmin=-1, vmax=1
    )
    plt.tick_params(labelsize=16)
    plt.grid(False)


class AgglomerativeTree:
    def __init__(self, metric="euclidean", linkage="average"):
        self.metric = metric
        self.linkage = linkage

    def fit(self, X):
        # condensed distance matrix, then linkage tree — built once
        self.Z_ = linkage(pdist(X, metric=self.metric), method=self.linkage)
        self.n_samples_ = X.shape[0]
        return self

    def predict(self, k):
        # slice the tree at k clusters; labels come back 1..k, shift to 0..k-1
        return fcluster(self.Z_, t=k, criterion="maxclust") - 1

    def predict_all(self, k_range=None):
        if k_range is None:
            k_range = range(2, self.n_samples_)
        return {k: self.predict(k) for k in k_range}


def agglom_ward_predict(k, corr):
    return AgglomerativeClustering(k, metric="euclidean", linkage="ward").fit_predict(
        corr
    )


def agglom_complete_predict(k, corr):
    return AgglomerativeClustering(
        k, metric="euclidean", linkage="complete"
    ).fit_predict(corr)


def agglom_average_predict(k, corr):
    return AgglomerativeClustering(
        k, metric="euclidean", linkage="average"
    ).fit_predict(corr)


def agglom_single_predict(k, corr):
    return AgglomerativeClustering(k, metric="euclidean", linkage="single").fit_predict(
        corr
    )


def k_means_s42_predict(k, corr):
    return KMeans(k, random_state=42).fit_predict(corr)


def make_k_means_predictor(seed):
    def k_means_predict(k, corr):
        return KMeans(k, random_state=seed).fit_predict(corr)

    return k_means_predict


def pca_predict(X_std):
    pca = PCA().fit(X_std)
    pca_correlations = pca.components_ * np.sqrt(pca.explained_variance_)[:, np.newaxis]
    return np.abs(pca_correlations).argmax(axis=0)


def fa_predict(X_std):
    fa = FactorAnalysis().fit(X_std)
    return np.abs(fa.components_).argmax(axis=0)


def favm_predict(X_std):
    favm = FactorAnalysis(rotation="varimax").fit(X_std)
    return np.abs(favm.components_).argmax(axis=0)


def get_index(X, labels, index):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint32).reshape((-1, 1))
    cic = ClusterIndexCalculator([index])
    return getattr(cic.compute(X, labels), index)[0]


def get_indexes(X, labels, indexes):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint32).reshape((-1, 1))
    cic = ClusterIndexCalculator(list(indexes))
    cic_res = cic.compute(X, labels)
    return {idx: getattr(cic_res, idx)[0] for idx in indexes}


def diagi(n):
    return (np.arange(n), np.arange(n))


@dataclass(repr=False)
class GenResult:
    data: pd.DataFrame
    labels: list[int]

    def __iter__(self):
        return iter((self.data, self.labels))


def gen_test_dataset(sizes, mean_rhos, n_samples=100, seed=None):
    rng = np.random.default_rng(seed)
    n_features = sum(sizes)
    base_corr = rng.uniform(0, 0.15, size=(n_features, n_features))
    skip = 0
    labels = []
    idx = 0
    for m, rho in zip(sizes, mean_rhos, strict=True):
        # loc = -0.76942857 + 0.3577381*rho + 1.00833333*rho*rho
        # scales for better mean intercluster corr. Does not allow to get values higher than 0.85
        loc = (
            -0.88281092
            + 1.22996759 * rho
            - 0.77824526 * rho * rho
            + 0.75222568 * rho * rho * rho
        )
        clust = stats.beta.rvs(20, 3, loc=loc, size=(m, m), random_state=rng).clip(0, 1)
        base_corr[skip : (skip + m), skip : (skip + m)] = clust
        skip += m
        labels += m * [idx]
        idx += 1

    for i in range(n_features - 1):
        for j in range(i + 1, n_features):
            base_corr[i, j] = base_corr[j, i]

    base_corr[diagi(n_features)] = 1.0
    evals, evecs = eigh(base_corr)
    evals[evals < 0] = 1e-6
    base_corr = evecs @ np.diag(evals) @ evecs.T
    sqrt_diag = np.sqrt(np.diag(base_corr))
    base_corr = base_corr / np.outer(sqrt_diag, sqrt_diag)

    data = pd.DataFrame(rng.multivariate_normal([0] * n_features, base_corr, n_samples))
    return GenResult(data, labels)


def simple_fclusters_stats(X, labels, abs=False):
    X = pd.DataFrame(X)
    corr = X.corr()
    if abs:
        corr = corr.abs()
    labels, n_clusters = fix_labels(labels)
    all_cluster_stats = []
    for k in range(n_clusters):
        cluster_size = (labels == k).sum()
        upper_tri = np.triu_indices(cluster_size, k=1)
        cluster_corr = corr.iloc[labels == k, labels == k]
        cluster_corr_flat = cluster_corr.to_numpy()[upper_tri]
        pd.Series(cluster_corr_flat).plot.hist()
        plt.show()
        cluster_stats = pd.Series(cluster_corr_flat).describe().to_dict()
        cluster_stats = {"cluster_size": cluster_size} | cluster_stats
        all_cluster_stats.append(cluster_stats)
    upper_tri = np.triu_indices(corr.shape[0], k=1)
    upper_tri_mask = np.zeros(corr.shape, dtype=bool)
    upper_tri_mask[upper_tri] = True
    outside = corr.to_numpy()[(labels != labels.reshape((-1, 1))) & upper_tri_mask]
    outside_stats = pd.Series(outside).describe().to_dict()
    res = pd.DataFrame(
        all_cluster_stats + [outside_stats], index=list(range(n_clusters)) + ["outside"]
    )
    return res.convert_dtypes()


def run_len(lst):
    return len(list(itertools.takewhile(lambda x: x == lst[0], lst)))


def name_dataset(clusters, correlations, version):
    Fsec = []
    Rsec = []
    correlations = (np.array(correlations) * 100).astype(int)
    for arr, sec in [(clusters, Fsec), (correlations, Rsec)]:
        while len(arr):
            rlen = run_len(arr)
            if rlen > 1:
                sec.append(f"({rlen}){arr[0]}")
            else:
                sec.append(f"{arr[0]}")
            arr = arr[rlen:]
    Fsec_joined = ",".join(Fsec)
    Rsec_joined = ",".join(Rsec)
    return f"F[{Fsec_joined}] R[{Rsec_joined}] v{version}"


def gen_test_collection(
    k_opts=[8], f_opts=[5], r_opts=[0.8], k1_opts=[0], n_versions=1, size=500, seed=None
):
    res = {}
    for k in k_opts:
        for f in f_opts:
            fs = f(k) if callable(f) else k * [f]
            for r in r_opts:
                rs = r(k) if callable(r) else k * [r]
                for k1 in k1_opts:
                    for version in range(n_versions):
                        name = name_dataset(fs + k1 * [1], rs + k1 * [0], version)
                        res[name] = gen_test_dataset(
                            fs + k1 * [1], rs + k1 * [0], size, seed=seed
                        )
    return res
