import pandas as pd
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Union
from hydra.utils import to_absolute_path

# Own modules
from src.utils import real_to_binary, array_to_str

# Global variables
jgb_data_path = to_absolute_path("data/jgbcme_all.csv")  # publicly accessible on website of ministry of finance Japan, see readme
init_qubit_order_bas = {
    "3x3" : [0, 1, 2, 5, 4, 3, 6 ,7, 8]
}

@dataclass
class BAS:
    binary: np.ndarray
    width: int
    height: int

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        num_samples = self.num_lines_stripes(width, height)

        # generate bars and stripes samples
        dataset = np.zeros((num_samples, width * height), dtype=int)
        for i in range(num_samples):
            sample = self.lines_stripes(width, height, i)
            dataset[i,:] = sample
        self.binary = dataset

    def num_lines_stripes(self, a, b):
        return pow(2,a)+pow(2,b)-2

    def lines_stripes(self, a, b, c):
        arr = np.zeros(a*b)
        c = c+1
        if (c<pow(2,a)): #lines
            lines = np.zeros (a, dtype = bool)
            for i in range (a):
                if (c%2 == 1):
                    lines[i] = True
                c = int(c/2)
            index = 0
            for element in lines:
                if element:
                    for i in range(index, index+b):
                        arr[i] = 1
                index = index+b
        else: #stripes
            c = c-pow(2,a)
            stripes = np.zeros (b, dtype = bool)
            for i in range (b):
                if (c%2 == 1):
                    stripes[i] = True
                c = int(c/2)
            index = 0
            for element in stripes:
                if element:
                    for i in range(index, a*b, b):
                        arr[i] = 1
                index = index+1
        return arr.astype(int)


    def print_lines_stripes(self, a, b, arr):
        index = 0
        for i in range(a):
            word = ''
            for j in range (b):
                if(arr[index] == 1):
                    word = word + 'X'
                else:
                    word = word + '0'
                index = index + 1
            print(word)


@dataclass
class JGB:
    raw: pd.DataFrame
    decimal: pd.DataFrame
    binary: np.ndarray
    conv_min_max: list[float]

    def __init__(self, N_qubits: int, N_features: int):
        df = pd.read_csv(jgb_data_path, skiprows=1, index_col=0, parse_dates=True)
        df = df.apply(pd.to_numeric, errors='coerce')  # convert str to float
        df = df.loc['2000-01-01':]  # low interest rate regime

        # filter number of features
        if N_features == 4:
            df = df[['2Y','5Y','10Y','20Y']]
        elif N_features == 3:
            df = df[['5Y','10Y','20Y']]
        else:
            raise ValueError("Number of features not supported.")

        df = df.dropna()
        self.raw = df.copy()
        df = df.diff().dropna()  # absolute day-over-day differences
        self.decimal = df.copy()
        self.N_features = N_features
        self.bits_per_feature = N_qubits // N_features
        # NOTE: this full-series binarization fits its min/max on ALL rows and is therefore NOT
        # leakage-safe. It is kept only for exploratory use and dataset figures. Training/eval
        # must use DataLoader.train_val_test_split, which re-binarizes JGB with train-only bounds.
        self.binary, self.conv_min_max = real_to_binary(self.decimal.values, self.bits_per_feature)


@dataclass
class DataLoader:
    dataset: Union[BAS, JGB]
    binary: np.ndarray
    count: dict[str, int]

    def __init__(self, dataset: Union[BAS, JGB]):
        self.dataset = dataset
        self.binary = dataset.binary
        self.count = Counter(array_to_str(dataset.binary))
        # per-feature binarization bounds fitted on the train split (JGB); set by train_val_test_split
        self.conv_min_max = None
    
    def reorder_features(self, X: np.ndarray) -> np.ndarray:
        """ Reorder the features based on the dataset.
        This is necessary to define the linear topology for MPS pretraining. 
        The order is defined in the global variable init_qubit_order_bas.
        Args:
            X (np.ndarray): Binary dataset.
        Returns:
            np.ndarray: Reordered binary dataset.
        """
        if isinstance(self.dataset, BAS):
            if  self.dataset.width == 3 and self.dataset.height == 3:
                return np.array([X[i][init_qubit_order_bas["3x3"]] for i in range(X.shape[0])])
            else:
                return X
        else:
            return X
        
    def train_val_test_split(self, train_size: float, val_size: float, reorder: bool = True,
                             seed: int = None, bas_split_mode: str = "full_support"):
        """Leakage-safe 3-way split into train/validation/test sets.

        Returns X_train, X_val, X_test, count_train, count_val, count_test.
        The test fraction is implied as 1 - train_size - val_size.

        - JGB (time series): contiguous chronological blocks (train = earliest, val = next,
          test = most recent). Binarization bounds are fitted on the TRAIN slice only and reused
          (with clipping) for val/test, so no future information leaks into the encoding. The
          fitted bounds are stored on self.conv_min_max for later inversion (binary_to_real).
        - BAS (finite enumerable support):
            * "full_support" (default): train/val/test are all the complete enumerated pattern set
              (the standard QCBM/BAS evaluation, where generalization = mode coverage).
            * "holdout": a seeded disjoint partition of the enumerated patterns.
        """
        assert 0 < train_size < 1, "train_size must be in (0, 1)"
        assert 0 <= val_size < 1 and train_size + val_size < 1, "invalid val_size / test fraction"

        if isinstance(self.dataset, JGB):
            return self._split_jgb(train_size, val_size, reorder)
        return self._split_bas(train_size, val_size, reorder, seed, bas_split_mode)

    def _counts(self, *arrays):
        return tuple(Counter(array_to_str(a)) for a in arrays)

    def _split_jgb(self, train_size, val_size, reorder):
        decimal = self.dataset.decimal.values
        bpf = self.dataset.bits_per_feature
        N = len(decimal)
        N_train = int(N * train_size)
        N_val = int(N * val_size)
        train_dec = decimal[:N_train]
        val_dec = decimal[N_train:N_train + N_val]
        test_dec = decimal[N_train + N_val:]

        # fit discretization bounds on the train slice only, reuse (clipped) for val/test
        _, [x_min, x_max] = real_to_binary(train_dec, bpf)
        self.conv_min_max = [x_min, x_max]
        X_train, _ = real_to_binary(train_dec, bpf, x_min, x_max, clip=True)
        X_val, _ = real_to_binary(val_dec, bpf, x_min, x_max, clip=True)
        X_test, _ = real_to_binary(test_dec, bpf, x_min, x_max, clip=True)

        if reorder:  # no-op for JGB, kept for interface symmetry
            X_train, X_val, X_test = (self.reorder_features(a) for a in (X_train, X_val, X_test))
        return (X_train, X_val, X_test, *self._counts(X_train, X_val, X_test))

    def _split_bas(self, train_size, val_size, reorder, seed, bas_split_mode):
        X = self.reorder_features(self.binary) if reorder else self.binary
        # BAS binarization has no fitted bounds; keep min/max at bit extremes for inversion symmetry
        self.conv_min_max = None

        if bas_split_mode == "full_support":
            # every split is the complete enumerated support (standard BAS eval)
            return (X, X, X, *self._counts(X, X, X))
        elif bas_split_mode == "holdout":
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(X))
            N = len(X)
            N_train = int(N * train_size)
            N_val = int(N * val_size)
            idx_train = perm[:N_train]
            idx_val = perm[N_train:N_train + N_val]
            idx_test = perm[N_train + N_val:]
            X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
            return (X_train, X_val, X_test, *self._counts(X_train, X_val, X_test))
        else:
            raise ValueError(f"Invalid bas_split_mode: {bas_split_mode}")

    def train_test_split(self, train_size: float, reorder: bool = True):
        """Backward-compatible 2-way split (train / test). Prefer train_val_test_split.

        Kept so legacy callers (e.g. the v1 plotting notebook) keep working; internally delegates
        to the 3-way split with val_size=0 and returns the old 4-tuple.
        """
        X_train, _, X_test, c_train, _, c_test = self.train_val_test_split(
            train_size, val_size=0.0, reorder=reorder)
        return X_train, X_test, c_train, c_test