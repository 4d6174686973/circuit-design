import pandas as pd
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Union

# Own modules
from src.utils import real_to_binary, array_to_str

# Global variables
jgb_data_path = "/workspaces/circuit-design/data/jgbcme_all.csv"  # publicly accessible on website of ministry of finance Japan, see readme
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
        self.binary, self.conv_min_max = real_to_binary(self.decimal.values, N_qubits // N_features)


@dataclass
class DataLoader:
    dataset: Union[BAS, JGB]
    binary: np.ndarray
    count: dict[str, int]

    def __init__(self, dataset: Union[BAS, JGB]):
        self.dataset = dataset
        self.binary = dataset.binary
        self.count = Counter(array_to_str(dataset.binary))
    
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
        
    def train_test_split(self, train_size: float, reorder: bool = True):
        """ Split the dataset into train and test set."""
        if reorder:
            X = self.reorder_features(self.binary)
        else:
            X = self.binary
        N = len(X)
        N_train = int(N * train_size)
        X_train, X_test = X[:N_train, :], X[N_train:, :]
        X_train_count = Counter(array_to_str(X_train))
        X_test_count = Counter(array_to_str(X_test))
        return X_train, X_test, X_train_count, X_test_count