import numpy as np
import pandas as pd
from dataclasses import dataclass, field

def bootstrap_mean_std(samples: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple:
    """Bootstrap the mean across runs (first axis) and the standard error of that mean.

    Given `samples` of shape (n_runs, *rest) -- e.g. one scalar per run (n_runs,) or one curve per
    run (n_runs, n_grid) -- draw `n_boot` resamples of the n_runs runs WITH REPLACEMENT, average
    each resample over its runs, and return (mean, std) where:
      - mean = mean over the n_boot resample-means (≈ the plain across-run mean),
      - std  = std  over the n_boot resample-means (the bootstrap standard error of the mean).
    Outputs have shape (*rest,) (0-d for scalar-per-run input). With a single run the std is 0.

    Implemented with a per-resample draw-count weight matrix (n_boot x n_runs) @ samples rather than
    materializing the full (n_boot, n_runs, *rest) tensor, so memory stays O(n_boot * (n_runs + R)).
    """
    samples = np.asarray(samples, dtype=float)
    n_runs = samples.shape[0]
    if n_runs == 0:
        raise ValueError("need at least one run to bootstrap")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_runs, size=(n_boot, n_runs))          # resampled run indices
    W = np.zeros((n_boot, n_runs))
    for b in range(n_boot):
        W[b] = np.bincount(idx[b], minlength=n_runs)              # how often each run was drawn
    W /= n_runs
    flat = samples.reshape(n_runs, -1)                            # (n_runs, R)
    boot_means = W @ flat                                         # (n_boot, R)
    mean = boot_means.mean(axis=0).reshape(samples.shape[1:])
    std = boot_means.std(axis=0).reshape(samples.shape[1:])
    return mean, std


def sample_info(samples_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract the sample information from a dictionary with form {bitstring: count}."""
    values = np.array([np.array([int(i) for i in bitstring]) for bitstring in samples_dict.keys()])
    probabilities = np.array(list(samples_dict.values())) / sum(samples_dict.values())
    return values, probabilities

def array_to_str(binary_array: np.ndarray) -> list[str]:
    """ Convert binary array of form [[1,1,1],[1,0,1],...] to bitstring array
    of form ["111","101",...] needed for counting """
    str_list = []
    for i in range(binary_array.shape[0]):
        str_list.append("".join(str(int(j)) for j in binary_array[i]))
    return str_list

QUANTIZERS = ("minmax", "arcsinh")
_MAD_TO_SIGMA = 1.4826  # 1/Phi^-1(0.75): makes MAD a consistent estimator of sigma under normality


@dataclass
class FeatureQuantizer:
    """Fixed-point quantizer mapping real features to `bits_per_feature` bits each.

    All parameters are FITTED (see `fit`) and then frozen, so encoding val/test data with a
    train-fitted quantizer carries no look-ahead bias.

    Both kinds share the same two-step structure -- a monotone warp g, then uniform quantization
    of the warped value onto 2**b levels:

        w  = g(x)                                       # per-feature warp
        k  = trunc((w - w_lo) / (w_hi - w_lo) * K),  K = 2**b - 1,  clipped to [0, K]
        x^ = g^-1(w_lo + k * (w_hi - w_lo) / K)         # reconstruction (bin lower edge)

    kind="minmax": g = identity, [w_lo, w_hi] = [min, max] of the fit data. Uniform bins in yield
    space. Simple, but the bin width is set by the most extreme day in the fit slice, so for
    heavy-tailed yield diffs most of the code space goes unused.

    kind="arcsinh": g(x) = arcsinh(x / s) = log(x/s + sqrt((x/s)^2 + 1)), with per-feature scale
    s = 1.4826 * MAD(fit data) (a robust sigma). This is the Johnson S_U translation (Johnson 1949)
    / the IHS transform of Burbidge, Magee & Robb (1988): odd, monotone, and defined for negative
    and zero values, unlike log. It is linear for |x| << s and logarithmic for |x| >> s, so

        dw/dx = 1 / (s * sqrt(1 + (x/s)^2))   =>   bin width in x grows as sqrt(1 + (x/s)^2),

    i.e. near-uniform resolution through the bulk (+-1 robust sigma) and geometrically widening
    bins in the tails. Outer bounds are the `tail_quantile` quantiles of the WARPED fit data, so
    tails are compressed rather than clipped away.
    """
    kind: str
    bits_per_feature: int
    w_min: np.ndarray                                  # per-feature lower bound, in WARPED space
    w_max: np.ndarray                                  # per-feature upper bound, in WARPED space
    scale: np.ndarray = None                           # arcsinh only: per-feature robust sigma s
    tail_quantile: float = field(default=0.999)        # arcsinh only: bound quantile

    # ---------------------------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, data: np.ndarray, bits_per_feature: int, kind: str = "minmax",
            tail_quantile: float = 0.999) -> "FeatureQuantizer":
        """Fit per-feature quantization parameters on `data` (shape (n_samples, n_features)).

        Pass ONLY the training slice here -- everything downstream reuses the frozen parameters.
        """
        if kind not in QUANTIZERS:
            raise ValueError(f"Unknown quantizer {kind!r}; expected one of {QUANTIZERS}")
        data = np.asarray(data, dtype=float)
        if kind == "minmax":
            return cls(kind, bits_per_feature, np.min(data, axis=0), np.max(data, axis=0))

        # robust per-feature scale; fall back to std (then 1.0) for degenerate/constant features
        mad = np.median(np.abs(data - np.median(data, axis=0)), axis=0) * _MAD_TO_SIGMA
        scale = np.where(mad > 0, mad, np.where(data.std(axis=0) > 0, data.std(axis=0), 1.0))
        w = np.arcsinh(data / scale)
        w_min = np.quantile(w, 1.0 - tail_quantile, axis=0)
        w_max = np.quantile(w, tail_quantile, axis=0)
        return cls(kind, bits_per_feature, w_min, w_max, scale, tail_quantile)

    # ------------------------------------------------------------------------------- warp / bins
    @property
    def max_int(self) -> int:
        return 2 ** self.bits_per_feature - 1

    def warp(self, x, feature: int = None):
        """Monotone forward warp g. `feature` selects one column's parameters (else all)."""
        if self.kind == "minmax":
            return np.asarray(x, dtype=float)
        s = self.scale if feature is None else self.scale[feature]
        return np.arcsinh(np.asarray(x, dtype=float) / s)

    def unwarp(self, w, feature: int = None):
        """Exact analytic inverse g^-1."""
        if self.kind == "minmax":
            return np.asarray(w, dtype=float)
        s = self.scale if feature is None else self.scale[feature]
        return np.sinh(np.asarray(w, dtype=float)) * s

    def _bounds(self, feature: int = None):
        if feature is None:
            return self.w_min, self.w_max
        return self.w_min[feature], self.w_max[feature]

    def levels(self, data: np.ndarray, clip: bool = True) -> np.ndarray:
        """Integer codes in [0, 2**b - 1], shape (n_samples, n_features)."""
        data = np.atleast_2d(np.asarray(data, dtype=float))
        lo, hi = self._bounds()
        # truncation (not rounding) toward zero, and multiply-before-divide -- both match the
        # original real_to_binary semantics bit-for-bit (the op order changes which side of a bin
        # edge borderline values land on)
        k = (self.max_int * (self.warp(data) - lo) / (hi - lo)).astype(int)
        if clip:
            return np.clip(k, 0, self.max_int)
        if (k < 0).any() or (k > self.max_int).any():
            raise ValueError("quantized level out of range with clip=False; pass clip=True to "
                             "map out-of-bounds values onto the extreme bins")
        return k

    def decode_levels(self, k, feature: int) -> np.ndarray:
        """Integer codes -> real values for one feature (inverse of `levels`, up to bin width)."""
        lo, hi = self._bounds(feature)
        return self.unwarp(lo + np.asarray(k, dtype=float) * (hi - lo) / self.max_int, feature)

    # ------------------------------------------------------------------------- encode / decode
    def encode(self, data: np.ndarray, clip: bool = True) -> np.ndarray:
        """Real data (n_samples, n_features) -> binary (n_samples, b * n_features), MSB first."""
        k = self.levels(data, clip=clip)
        b = self.bits_per_feature
        out = np.zeros((k.shape[0], b * k.shape[1]), dtype=int)
        for n in range(k.shape[1]):
            for m in range(b):
                out[:, n * b + m] = (k[:, n] >> (b - 1 - m)) & 1
        return out

    def decode(self, binary: np.ndarray) -> np.ndarray:
        """Binary (n_samples, b * n_features) -> real values (n_samples, n_features)."""
        binary = np.atleast_2d(np.asarray(binary, dtype=int))
        b = self.bits_per_feature
        n_features = binary.shape[1] // b
        weights = 2 ** np.arange(b - 1, -1, -1)
        out = np.zeros((binary.shape[0], n_features), dtype=float)
        for n in range(n_features):
            k = binary[:, n * b:(n + 1) * b] @ weights
            out[:, n] = self.decode_levels(k, n)
        return out

    @property
    def conv_min_max(self) -> list:
        """[w_min, w_max] -- the fitted bounds. NOTE: for kind="arcsinh" these live in WARPED
        space, so they are not directly usable for affine inversion; use `decode`/`decode_levels`
        (or `data_min_max`) instead."""
        return [self.w_min, self.w_max]

    @property
    def data_min_max(self) -> list:
        """The fitted bounds mapped back to data (yield) space -- for reporting/plot ranges."""
        return [self.unwarp(self.w_min), self.unwarp(self.w_max)]


def real_to_binary(data: np.ndarray, bits_per_feature: int, x_min=None, x_max=None,
                   clip: bool = True, kind: str = "minmax", quantizer: FeatureQuantizer = None):
    '''Conversion of real-valued data set into binary features. Every real valued number is
    converted into a n-bit binary number.

    Thin wrapper over FeatureQuantizer, kept for backwards compatibility.

    Parameters
    -----------
    data: DataFrame
        The real-valued data set of shape (n_samples, n_features).
    bits_per_feature: int
        Number of bits used to encode each feature.
    x_min, x_max: array-like or None
        Per-feature min/max used for the discretization. When None (default) they are fitted
        from `data` (original behaviour). Pass train-fitted bounds to binarize validation/test
        data without leaking their range into the encoding. Ignored when `quantizer` is given.
    clip: bool
        When True, clip the discretized integer into [0, 2**bits_per_feature - 1] so values that
        fall outside the provided bounds map to the extreme bin instead of overflowing the bit
        width (essential when reusing train bounds on unseen val/test data).
    kind: str
        Quantizer family to fit when neither `quantizer` nor explicit bounds are given.
    quantizer: FeatureQuantizer or None
        A pre-fitted quantizer. Takes precedence over x_min/x_max/kind -- the only way to reuse
        a non-"minmax" fit (arcsinh needs a per-feature scale, which bounds alone cannot carry).

    Returns
    --------
    data_binary: DataFrame
        The binary data set fo shape (n_samples, bits_per_feature * n_features).
    [x_min, x_max]: list
        The fitted bounds, in the quantizer's warped space (identical to data space for
        kind="minmax"). Also available as `quantizer.conv_min_max`.'''
    if quantizer is None:
        if x_min is None or x_max is None:
            quantizer = FeatureQuantizer.fit(data, bits_per_feature, kind)
        elif kind == "minmax":
            quantizer = FeatureQuantizer(kind, bits_per_feature,
                                         np.asarray(x_min, dtype=float),
                                         np.asarray(x_max, dtype=float))
        else:
            # arcsinh's fit is (scale, warped bounds); bounds alone cannot reconstruct it
            raise ValueError(f"kind={kind!r} cannot be rebuilt from x_min/x_max alone -- pass the "
                             "fitted quantizer via quantizer=...")
    return quantizer.encode(data, clip=clip), quantizer.conv_min_max

def binary_to_real(X_binary, X_min, X_max, bits_per_feature, quantizer: FeatureQuantizer = None):
    """
    Converts a set of binary features back into real-valued data.

    Thin wrapper over FeatureQuantizer.decode, kept for backwards compatibility.

    Parameters
    -----------
    X_binary: DataFrame
        The binary data set of shape (n_samples, bits_per_feature * n_features).
    X_min: float or array-like
        The minimum value for each feature (output of from_real_to_binary).
    X_max: float or array-like
        The maximum value for each feature (output of from_real_to_binary).
    quantizer: FeatureQuantizer or None
        A pre-fitted quantizer. Takes precedence over X_min/X_max -- required to invert a
        non-"minmax" encoding, whose bounds live in warped space.

    Returns
    --------
    X_real: DataFrame
        The real-valued data set of shape (n_samples, n_features).
    """
    if quantizer is None:
        X_min = np.atleast_1d(np.asarray(X_min, dtype=float))
        X_max = np.atleast_1d(np.asarray(X_max, dtype=float))
        quantizer = FeatureQuantizer("minmax", bits_per_feature, X_min, X_max)
    return quantizer.decode(X_binary)

def get_features_for_quasi_dist(samples_dict, bits_per_feature, num_features):
    sample_gen_arr, sample_gen_probs = sample_info(samples_dict)
    res_dicts = []
    for i in range(num_features):
        f_arr = array_to_str(sample_gen_arr[:,i*bits_per_feature:(i+1)*bits_per_feature]).tolist()
        f_dict = dict()
        for j in range(len(f_arr)):
            if f_arr[j] in f_dict:
                f_dict[f_arr[j]] += sample_gen_probs[j]
            else:
                f_dict[f_arr[j]] = sample_gen_probs[j]
        assert round(sum([v for v in f_dict.values()]), 12) == 1.0
        res_dicts.append(f_dict)

    return res_dicts

### Variation of Information Metric ###
import numpy as np,scipy.stats as ss
from scipy.spatial import distance
from sklearn.metrics import mutual_info_score

def numBins(nObs,corr=None):
    # Optimal number of bins for discretization
    if corr is None: # univariate case
        z=(8+324*nObs+12*(36*nObs+729*nObs**2)**.5)**(1/3.)
        b=round(z/6.+2./(3*z)+1./3)
    else: # bivariate case
        b=round(2**-.5*(1+(1+24*nObs/(1.-corr**2))**.5)**.5)
    return int(b)

def varInfo(x,y,norm=False):
    # variation of information
    bXY=numBins(x.shape[0],corr=np.corrcoef(x,y)[0,1])
    
    cXY=np.histogram2d(x,y,bXY)[0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    hX=ss.entropy(np.histogram(x,bXY)[0]) # marginal
    hY=ss.entropy(np.histogram(y,bXY)[0]) # marginal
    vXY=hX+hY-2*iXY # variation of information
    if norm:
        hXY=hX+hY-iXY # joint
        vXY/=hXY # normalized variation of information
    return vXY

def varInfoMat(X, norm=False):
    '''Compute VarInfo on whole matrix X'''
    l = X.shape[1]
    metric = np.full([l,l], np.nan)
    for i in range(l):
        for j in range(l):
            if not i == j:
                metric[i,j] = varInfo(X.iloc[:,i].values, X.iloc[:,j].values, norm=norm)
            else:
                metric[i,j] = 0
    return pd.DataFrame(metric, index=X.columns, columns=X.columns)

def mutual_info_matrix(X: np.ndarray) -> np.ndarray:
    '''Pairwise mutual information between the columns (features/qubits) of a binary matrix X.

    Returns a symmetric (n_features x n_features) matrix whose (i, j) entry is I(X_i; X_j) in nats;
    a higher value means the two feature-bits are more statistically dependent. This is the
    edge-affinity consumed by extension.chow_liu_topology to build the Chow-Liu dependency tree.
    The columns are already binary (one bit per qubit), so mutual_info_score is applied directly to
    the label vectors -- no binning -- mirroring how the Hamming metric operates on the raw bits.'''
    n = X.shape[1]
    mim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mim[i, j] = mim[j, i] = mutual_info_score(X[:, i], X[:, j])
    return mim

def feature_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    '''Pairwise feature-distance matrix consumed by the metric_based extension and its threshold
    curve (extension.connection_threshold_curve / knee_threshold). Always returns an ndarray,
    regardless of metric, so downstream code doesn't need to special-case a pandas DataFrame vs a
    numpy array depending on which branch built it.

    metric: "hamming" -> scipy Hamming distance over the raw bit-columns; "varinfo" -> normalized
    variation of information (varInfoMat). Both are 0 for identical columns and increase with
    dependence, i.e. LOWER distance means MORE dependent features.'''
    if metric == "hamming":
        X = np.asarray(X)
        return distance.cdist(X.T, X.T, "hamming")
    elif metric == "varinfo":
        return varInfoMat(pd.DataFrame(X), norm=True).values
    else:
        raise ValueError(f"Invalid extension metric: {metric}")


### FOR PLOTTING RESULTS ###
def get_features_for_quasi_dist(samples_dict, bits_per_feature, num_features):
    sample_gen_arr, sample_gen_probs = sample_info(samples_dict)
    res_dicts = []
    for i in range(num_features):
        f_arr = array_to_str(sample_gen_arr[:,i*bits_per_feature:(i+1)*bits_per_feature])
        f_dict = dict()
        for j in range(len(f_arr)):
            if f_arr[j] in f_dict:
                f_dict[f_arr[j]] += sample_gen_probs[j]
            else:
                f_dict[f_arr[j]] = sample_gen_probs[j]
        assert round(sum([v for v in f_dict.values()]), 12) == 1.0
        res_dicts.append(f_dict)

    return res_dicts


def plot_mmd_two_sets(data: dict, colors: dict, mode: str = "medperc", iter: int = 1000, window: int = 50, filename: str = "test", save: bool = False):
    """
    Plot MMD for two sets of data
    
    Args:
    - data: dictionary with keys 'train' and 'test' and values as list of paths to runs
    - colors: dictionary with keys as data keys and values as color
    - mode: "meanstd" or "medperc" for plotting
    - iter: number of iterations to plot
    - window: window for moving average
    - filename: filename for saving
    - save: save plot or not
    
    Returns:
    - fig: figure object
    """
    import matplotlib.pyplot as plt

    losses = {'train': {}, 'test': {}}
    plot_data = {'train': {}, 'test': {}}

    for l in data.keys():
        losses['train'][l] = []
        losses['test'][l] = []
        plot_data['train'][l] = []
        plot_data['test'][l] = []

    # read in all losses and sort in dictionary
    for k, runs in data.items():
        for run in runs:
            df = pd.read_parquet(f'{run}/qcbm/losses.parquet')
            df = df.iloc[:iter]
            losses['train'][k].append(df['mmd_train'].values)
            losses['test'][k].append(df['mmd_test'].values)
        
    # calculate mean, min and max for each run
    for k, runs in data.items():
        for t in ['train', 'test']:
            losses[t][k] = np.array(losses[t][k])
            if mode == "meanstd":
                std = np.std(losses[t][k], axis=0)
                mean = np.mean(losses[t][k], axis=0)
                plot_data[t][k] = {
                    'line': mean,
                    'upper': mean + std,
                    'lower': mean - std
                }
            elif mode == "medperc":
                plot_data[t][k] = {
                    'line': np.median(losses[t][k], axis=0),
                    'upper': np.percentile(losses[t][k], 90, axis=0),
                    'lower': np.percentile(losses[t][k], 10, axis=0)
                }

    # compute moving average on all data
    for k in data.keys():
        for t in ['train', 'test']:
            plot_data[t][k]['line'] = pd.Series(plot_data[t][k]['line']).rolling(window).mean()
            plot_data[t][k]['lower'] = pd.Series(plot_data[t][k]['lower']).rolling(window).mean()
            plot_data[t][k]['upper'] = pd.Series(plot_data[t][k]['upper']).rolling(window).mean()

    # plot train and test seperately
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    axs = ax.flatten()
    subtitles = ['a) Train', 'b) Test']
    for i, t in enumerate(['train', 'test']):
        for k in data.keys():
            if colors is None:
                axs[i].plot(plot_data[t][k]['line'], label=k)
                axs[i].fill_between(range(len(plot_data[t][k]['line'])), plot_data[t][k]['lower'], plot_data[t][k]['upper'], alpha=0.3, lw=0.0)
            else:
                col = colors[k]
                axs[i].plot(plot_data[t][k]['line'], label=k, ls='-', color=col)
                axs[i].fill_between(range(len(plot_data[t][k]['line'])), plot_data[t][k]['lower'], plot_data[t][k]['upper'], alpha=0.3, color=col, lw=0.0)
        
        axs[i].set_ylabel('MMD')
        axs[i].set_xlabel('Iteration')
        axs[i].set_title(subtitles[i])

    # set same legend for both axs outside of plot
    axs[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    # save plot
    plt.tight_layout()
    if save:
        plt.savefig(f'plots/{filename}.pdf', bbox_inches='tight', transparent=True)
        plt.savefig(f'plots/{filename}.png', bbox_inches='tight', transparent=False, dpi=300)

    return fig


def plot_mmd_one_set(data: dict, colors: dict, evalset: str = "train", mode: str = "medperc", iter: int = 1000, window: int = 50, filename: str = "testfile", save: bool = False):
    """
    Plot MMD for all runs in one figure
    
    Args:
    - data: dictionary with data to plot
    - colors: dictionary with colors for each run
    - evalset: train or test
    - mode: either 'meanstd' or 'medperc'
    - iter: number of iterations to plot
    - window: window for moving average
    - filename: filename to save plot
    - save: save plot or not
    
    Returns:
    - fig: figure object
    """
    import matplotlib.pyplot as plt

    losses = {evalset: {}}
    plot_data = {evalset: {}}

    for l in data.keys():
        losses[evalset][l] = []
        plot_data[evalset][l] = []

    # read in all losses and sort in dictionary
    for k, runs in data.items():
        for run in runs:
            df = pd.read_parquet(f'{run}/qcbm/losses.parquet')
            df = df.iloc[:iter]
            losses[evalset][k].append(df[f'mmd_{evalset}'].values)
        
    # calculate mean, min and max for each run
    for k, runs in data.items():
        losses[evalset][k] = np.array(losses[evalset][k])
        if mode == "meanstd":
            std = np.std(losses[evalset][k], axis=0)
            mean = np.mean(losses[evalset][k], axis=0)
            plot_data[evalset][k] = {
                'line': mean,
                'upper': mean + std,
                'lower': mean - std
            }
        elif mode == "medperc":
            plot_data[evalset][k] = {
                'line': np.median(losses[evalset][k], axis=0),
                'upper': np.percentile(losses[evalset][k], 90, axis=0),
                'lower': np.percentile(losses[evalset][k], 10, axis=0)
            }

    # compute moving average on all data
    for k in data.keys():
        plot_data[evalset][k]['line'] = pd.Series(plot_data[evalset][k]['line']).rolling(window).mean()
        plot_data[evalset][k]['lower'] = pd.Series(plot_data[evalset][k]['lower']).rolling(window).mean()
        plot_data[evalset][k]['upper'] = pd.Series(plot_data[evalset][k]['upper']).rolling(window).mean()

    # plot train only
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    # axs = ax.flatten()
    for k in data.keys():
        if colors is None:
            ax.plot(plot_data[evalset][k]['line'], label=k)
            ax.fill_between(range(len(plot_data[evalset][k]['line'])), plot_data[evalset][k]['lower'], plot_data[evalset][k]['upper'], alpha=0.3, lw=0.0)
        else:
            col = colors[k]
            ax.plot(plot_data[evalset][k]['line'], label=k, ls='-', color=col)
            ax.fill_between(range(len(plot_data[evalset][k]['line'])), plot_data[evalset][k]['lower'], plot_data[evalset][k]['upper'], alpha=0.3, color=col, lw=0.0)
        
        ax.set_ylabel('MMD')
        ax.set_xlabel('Iteration')

    # set legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    # save plot
    plt.tight_layout()
    if save:
        plt.savefig(f'plots/{filename}.pdf', bbox_inches='tight', transparent=True)
        plt.savefig(f'plots/{filename}.png', bbox_inches='tight', transparent=False, dpi=300)
    
    return fig
