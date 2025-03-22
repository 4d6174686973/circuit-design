import numpy as np
import pandas as pd

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

def real_to_binary(data: np.ndarray, bits_per_feature: int):
    '''Conversion of real-valued data set into binary features. Every real valued number is 
    converted into a n-bit binary number.
    
    Parameters
    -----------
    data: DataFrame
        The real-valued data set of shape (n_samples, n_features).
    
    Returns
    --------
    data_binary: DataFrame
        The binary data set fo shape (n_samples, bits_per_feature * n_features).'''
    data_binary = np.array([[0] * (bits_per_feature * data.shape[1]) for _ in range(data.shape[0])])
    x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
    for n in range(data.shape[1]):
        for l in range(data.shape[0]):
            x_integer = int((2**bits_per_feature - 1) * (data[l, n] - x_min[n]) / (x_max[n] - x_min[n]))
            binary_string = bin(x_integer)[2:].zfill(bits_per_feature)
            for bit_index, bit in enumerate(binary_string):
                data_binary[l, n * bits_per_feature + bit_index] = int(bit)
    return data_binary, [x_min, x_max]

def binary_to_real(X_binary, X_min, X_max, bits_per_feature):
    """
    Converts a set of binary features back into real-valued data.

    Parameters
    -----------
    X_binary: DataFrame
        The binary data set of shape (n_samples, bits_per_feature * n_features).
    X_min: float or array-like
        The minimum value for each feature (output of from_real_to_binary).
    X_max: float or array-like
        The maximum value for each feature (output of from_real_to_binary).
    
    Returns
    --------
    X_real: DataFrame
        The real-valued data set of shape (n_samples, n_features).
    """
    N_samples = len(X_binary)
    if isinstance(X_min, float):
        N_variables = 1
    else:
        N_variables = len(X_min)
    X_real = [[0] * N_variables for _ in range(N_samples)]

    for n in range(N_variables):
        for l in range(N_samples):
            X_integer = sum(X_binary[l][n * bits_per_feature + m] * (2 ** (bits_per_feature - 1 - m)) for m in range(bits_per_feature))
            X_real[l][n] = X_min[n] + (X_integer * (X_max[n] - X_min[n]) / ((2 ** bits_per_feature) -1))
    
    X_real = np.array(X_real)

    return X_real

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


### FOR PLOTTING RESULTS ###
import matplotlib.pyplot as plt 


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
