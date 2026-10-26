import torch as T
import torch
import torch.nn.functional as F

import numpy as np
import tqdm
import random
import sys, os
from matplotlib import pyplot as plt
from sympy.combinatorics.graycode import GrayCode
import time
import ipdb
from typing import Tuple
from torch.autograd import Variable, Function



def get_true_samples(db: object, size: int, bm: object, int_salce: float, discrete_dim: int, seed: int = None) -> torch.Tensor:
    '''
    Gets true samples from the dataset.
    Args:
        db: the dataset object
        size: number of samples to get
        bm: binary mapping
        int_scale: integer scale for the dataset
        discrete_dim: dimension of the discrete space
        seed: optional seed for reproducibility
    Returns:
        samples: a tensor of true samples in binary representation
    >>> Example:
        >>> get_true_samples(db, 100, bm, int_scale, discrete_dim)
        [tensor([[0., 1., 0.,  ..., 1., 0., 1.],
                 [1., 0., 1.,  ..., 0., 1., 0.],
                 ...,
                 [0., 1., 1.,  ..., 0., 1., 0.],
                 [1., 0., 0.,  ..., 1., 0., 1.]])]  
    '''
    if seed is None:
        samples = float2bin(db.gen_batch(size), bm, int_salce, discrete_dim)
    else:
        samples = float2bin(db.gen_batch_with_seed(size, seed), bm, int_salce, discrete_dim)
    return torch.from_numpy(samples).float()

def get_ebm_samples(score_func: callable, size: int, inv_bm: object, int_scale: float, discrete_dim: int, device: torch.device, gibbs_sampler: object = None, gibbs_steps: int = 20) -> Tuple[torch.Tensor, np.ndarray]:
    '''
    Gets samples from the EBM using Gibbs sampling.
    Args:
        score_func: the scoring function of the EBM
        size: number of samples to get
        inv_bm: inverse binary mapping
        int_scale: integer scale for the dataset
        discrete_dim: dimension of the discrete space
        device: the device to run the sampling on
        gibbs_sampler: an optional Gibbs sampler object
        gibbs_steps: number of Gibbs sampling steps to perform
    Returns:
        ebm_samples: a tensor of samples from the EBM in binary representation
        ebm_samp_float: a numpy array of samples from the EBM in float representation
    >>> Example:
        >>> get_ebm_samples(score_func, 100, inv_bm, int_scale, discrete_dim, device)
        [tensor([[0., 1., 0.,  ..., 1., 0., 1.],
                 [1., 0., 1.,  ..., 0., 1., 0.],
                 ...,
                 [0., 1., 1.,  ..., 0., 1., 0.],
                 [1., 0., 0.,  ..., 1., 0., 1.]])], 
         array([[ 0.5, -0.5],
                [-0.5,  0.5],
                ...,
                [ 0.5,  0.5],
                [-0.5, -0.5]])]
    '''
    unif_dist = torch.distributions.Bernoulli(probs=0.5)
    ebm_samples = unif_dist.sample((size, discrete_dim)).to(device)
    ebm_samp_float = []
    for ind in range(gibbs_steps * discrete_dim):  # takes about 1s
        ebm_samples = gibbs_sampler.step(ebm_samples, score_func)
    ebm_samp_float.append(bin2float(ebm_samples.data.cpu().numpy().astype(int), inv_bm, int_scale, discrete_dim))
    ebm_samp_float = np.concatenate(ebm_samp_float, axis=0)
    return ebm_samples, ebm_samp_float

def estimate_ll(score_func: callable, samples: torch.Tensor, n_partition: int = None, rand_samples: torch.Tensor = None) -> float:
    '''
    Estimates the log-likelihood of the given samples using importance sampling.
    Args:
        score_func: the scoring function of the EBM
        samples: a tensor of samples to estimate the log-likelihood for
        n_partition: number of partitions for importance sampling (optional)
        rand_samples: a tensor of random samples for importance sampling (optional)
        Returns:
        ll: the estimated log-likelihood of the samples
        >>> Example:
            >>> estimate_ll(score_func, samples, n_partition=1000)
            -2.345678
    '''
    with torch.no_grad():
        if rand_samples is None:
            rand_samples = torch.randint(2, (n_partition, samples.shape[1])).float().to(samples.device)
        n_partition = rand_samples.shape[0]
        f_z_list = []
        for i in range(0, n_partition, samples.shape[0]):  # 从0数到n_partition，每一份是samples.shape[0]大小
            f_z = score_func(rand_samples[i:i+samples.shape[0]]).view(-1, 1)
            f_z_list.append(f_z)
        f_z = torch.cat(f_z_list, dim=0)
        f_z = f_z - samples.shape[1] * np.log(0.5) - np.log(n_partition)  # log(1/2)是unif的概率，importance sample的时候在分母

        # log_part = logsumexp(f_z)
        log_part = f_z.logsumexp(0)
        f_sample = score_func(samples)
        ll = f_sample - log_part

    return torch.mean(ll).item()


def exp_hamming_sim(x: torch.Tensor, y: torch.Tensor, bd: float) -> torch.Tensor:
    '''
    Computes the exponential of the negative Hamming distance between two sets of binary vectors.
    Args:
        x: a tensor of shape (N, D) representing the first set of binary vectors
        y: a tensor of shape (M, D) representing the second set of binary vectors
        bd: a float representing the bandwidth for the exponential function
    Returns:
        sim: a tensor of shape (N, M) representing the exponential of the negative Hamming distance between each pair of vectors from x and y
    >>> Example:
        >>> exp_hamming_sim(torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0, 1], [1, 1]]), bd=0.1)
        tensor([[1.0000, 0.3679],
                [0.3679, 0.1353]])'''
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    d = T.sum(T.abs(x - y), dim=-1)
    return T.exp(-bd * d)

class MMD():
    def __init__(self,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 bandwidth: float = 1.0):
        self.x = x
        self.y = y
        self.bandwidth = bandwidth

    def exp_hamming_mmd(self, ham_sim_x: torch.Tensor, ham_sim_y: torch.Tensor, ham_sim_xy: torch.Tensor) -> torch.Tensor:
        '''
        Computes the Maximum Mean Discrepancy (MMD) between two sets of binary vectors using the exponential of the negative Hamming distance as the kernel.
        Args:
            ham_sim_x: a tensor of shape (N, N) representing the exponential of the negative Hamming distances within the first set of binary vectors
            ham_sim_y: a tensor of shape (M, M) representing the exponential of the negative Hamming distances within the second set of binary vectors
            ham_sim_xy: a tensor of shape (N, M) representing the exponential of the negative Hamming distances between vectors from the two sets
        Returns:
            mmd: a float representing the MMD between the two sets of binary vectors
        >>> Example:
            >>> mmd = MMD(torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0, 1], [1, 1]]), bandwidth=0.1)
            >>> mmd.exp_hamming_mmd(mmd.hamming_sim(), mmd.hamming_sim(), mmd.hamming_sim())
            0.1353'''
        x = self.x.float()
        y = self.y.float()

        with T.no_grad():
            kxx = ham_sim_x
            idx = T.arange(0, x.shape[0], out=T.LongTensor())
            kxx[idx, idx] = 0.0
            kxx = T.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

            kyy = ham_sim_y
            idx = T.arange(0, y.shape[0], out=T.LongTensor())
            kyy[idx, idx] = 0.0
            kyy = T.sum(kyy) / y.shape[0] / (y.shape[0] - 1)

            kxy = ham_sim_xy
            kxy = T.sum(kxy) / x.shape[0] / y.shape[0]

            mmd = kxx + kyy - 2 * kxy
        return mmd


    def hamming_sim(self) -> torch.Tensor:
        '''
        Computes the Hamming similarity between two sets of binary vectors.
        Args:
            x: a tensor of shape (N, D) representing the first set of binary vectors
            y: a tensor of shape (M, D) representing the second set of binary vectors
        Returns:
            sim: a tensor of shape (N, M) representing the Hamming similarity between each pair of vectors from x and y
        >>> Example:
            >>> hamming_sim(torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0, 1], [1, 1]]))
            tensor([[2, 1],
                    [1, 1]])
        '''
        x = self.x.unsqueeze(1)
        y = self.y.unsqueeze(0)
        d = torch.sum(torch.abs(x - y), dim=-1)
        ham_sim = x.shape[-1] - d
        return ham_sim

    def hamming_mmd(self, ham_sim_x: torch.Tensor, ham_sim_y: torch.Tensor, ham_sim_xy: torch.Tensor) -> torch.Tensor:
        '''
        Computes the Maximum Mean Discrepancy (MMD) between two sets of binary vectors using the Hamming similarity as the kernel.
        Args:
            ham_sim_x: a tensor of shape (N, N) representing the Hamming similarity between each pair of vectors from x
            ham_sim_y: a tensor of shape (M, M) representing the Hamming similarity between each pair of vectors from y
            ham_sim_xy: a tensor of shape (N, M) representing the Hamming similarity between vectors from x and y
        Returns:
            mmd: a float representing the MMD between the two sets of binary vectors
        >>> Example:
            >>> mmd = MMD(torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0, 1], [1, 1]]), bandwidth=0.1)
            >>> mmd.hamming_mmd(mmd.hamming_sim(), mmd.hamming_sim(), mmd.hamming_sim())
            0.5
        '''
        x = self.x.float()
        y = self.y.float()
        with torch.no_grad():
            kxx = ham_sim_x
            idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
            kxx[idx, idx] = 0.0
            kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

            kyy = ham_sim_y
            idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
            kyy[idx, idx] = 0.0
            kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
            kxy = torch.sum(ham_sim_xy) / x.shape[0] / y.shape[0]
            mmd = kxx + kyy - 2 * kxy
        return mmd


    def linear_mmd(self) -> torch.Tensor:
        '''
        Computes the Maximum Mean Discrepancy (MMD) between two sets of vectors using the linear kernel.
        Args:
            x: a tensor of shape (N, D) representing the first set of vectors
            y: a tensor of shape (M, D) representing the second set of vectors
        Returns:
            mmd: a float representing the MMD between the two sets of vectors
        >>> Example:
            >>> linear_mmd(torch.tensor([[0.5, 1.0], [1.0, 0.5]]), torch.tensor([[0.5, 1.0], [1.0, 1.0]]))
            0.125
        '''
        x = self.x.float()
        y = self.y.float()
        with torch.no_grad():
            kxx = torch.mm(x, x.transpose(0, 1))
            idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
            kxx = kxx * (1 - torch.eye(x.shape[0]).to(x.device))
            kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

            kyy = torch.mm(y, y.transpose(0, 1))
            idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
            kyy[idx, idx] = 0.0
            kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
            kxy = torch.sum(torch.mm(y, x.transpose(0, 1))) / x.shape[0] / y.shape[0]
            mmd = kxx + kyy - 2 * kxy
        return mmd


    def get_gamma(self, X: torch.Tensor) -> float:
        '''
        Computes the gamma parameter for the RBF kernel based on the median distance between samples.
        Args:
            X: a tensor of shape (N, D) representing the set of samples
        Returns:
            gamma: a float representing the gamma parameter for the RBF kernel
        >>> Example:
            >>> get_gamma(torch.tensor([[0.5, 1.0], [1.0, 0.5]]))
            0.5
        '''
        with torch.no_grad():
            x_norm = torch.sum(X ** 2, dim=1, keepdim=True)
            x_t = torch.transpose(X, 0, 1)
            x_norm_t = x_norm.view(1, -1)
            t = x_norm + x_norm_t - 2.0 * torch.matmul(X, x_t)
            dist2 = F.relu(Variable(t)).detach().data

            d = dist2.cpu().numpy()
            d = d[np.isfinite(d)]
            d = d[d > 0]
            median_dist2 = float(np.median(d))
            gamma = 0.5 / median_dist2 / self.bandwidth
            return gamma

    def pairwise_distances(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        '''
        Computes the pairwise squared Euclidean distances between two sets of vectors.
        Args:
            x: a tensor of shape (N, D) representing the first set of vectors
            y: a tensor of shape (M, D) representing the second set of vectors (optional, defaults to None)
        Returns:
            dist: a tensor of shape (N, M) representing the pairwise squared Euclidean distances between each pair of vectors from x and y
        >>> Example:
            >>> pairwise_distances(torch.tensor([[0.5, 1.0], [1.0, 0.5]]), torch.tensor([[0.5, 1.0], [1.0, 1.0]]))
            tensor([[0.0000, 0.2500],
                    [0.2500, 0.2500]])
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        pairwise_distances = torch.clamp(dist, 0.0, np.inf)
        return pairwise_distances

    def get_kernel_mat(x: torch.Tensor, 
                       landmarks: torch.Tensor, 
                       gamma: float, 
                       pairwise_distances: torch.Tensor) -> torch.Tensor:
        '''
        Computes the RBF kernel matrix between two sets of vectors.
        Args:
            x: a tensor of shape (N, D) representing the first set of vectors
            landmarks: a tensor of shape (M, D) representing the second set of vectors
            gamma: a float representing the gamma parameter for the RBF kernel
        Returns:
            k: a tensor of shape (N, M) representing the RBF kernel matrix between each pair of vectors from x and landmarks
        >>> Example:
            >>> get_kernel_mat(torch.tensor([[0.5, 1.0], [1.0, 0.5]]), torch.tensor([[0.5, 1.0], [1.0, 1.0]]), gamma=1.0, pairwise_distances=torch.tensor([[0.0, 0.25], [0.25, 0.25]])  )
            tensor([[1.0000, 0.3679],
                    [0.3679, 0.3679]])
        '''

        d = pairwise_distances
        k = torch.exp(d * -gamma)
        k = k.view(x.shape[0], -1)
        return k

    def MMD(self, 
            x: torch.Tensor, 
            y: torch.Tensor,
            gamma: float,
            kxx: torch.Tensor,
            kyy: torch.Tensor,
            kxy: torch.Tensor,
            pairwise_distances: torch.Tensor,
            ) -> torch.Tensor:
        '''
        Computes the Maximum Mean Discrepancy (MMD) between two sets of vectors using the RBF kernel.
        Args:
            x: a tensor of shape (N, D) representing the first set of vectors
            y: a tensor of shape (M, D) representing the second set of vectors
            gamma: a float representing the gamma parameter for the RBF kernel
            kxx: a tensor of shape (N, N) representing the RBF kernel matrix for x
            kyy: a tensor of shape (M, M) representing the RBF kernel matrix for y
            kxy: a tensor of shape (M, N) representing the RBF kernel matrix between y and x
            pairwise_distances: a tensor representing the pairwise distances between vectors
        Returns:
            mmd: a float representing the MMD between the two sets of vectors
        >>> Example:
            >>> MMD(torch.tensor([[0.5, 1.0], [1.0, 0.5]]), torch.tensor([[0.5, 1.0], [1.0, 1.0]]), 1.0, torch.tensor([[1.0, 0.3679], [0.3679, 0.3679]]), torch.tensor([[1.0, 0.3679], [0.3679, 0.3679]]), torch.tensor([[1.0, 0.3679], [0.3679, 0.3679]]), torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
            0.1353
        '''
        y = y.detach()
        idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
        kxx = kxx * (1 - torch.eye(x.shape[0]).to(x.device))
        kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

        idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
        kyy[idx, idx] = 0.0
        kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
        kxy = torch.sum(kxy) / x.shape[0] / y.shape[0]
        mmd = kxx + kyy - 2 * kxy
        return mmd



def get_binmap(discrete_dim: int, binmode: str = 'normal') -> Tuple[dict, dict]:
    '''
    Generates a binary mapping and its inverse for a given discrete dimension and binary mode.
    Args:
        discrete_dim: an integer representing the dimension of the discrete space
        binmode: a string representing the binary mode ('normal', 'rand', or 'gray')
    Returns:
        bm: a dictionary mapping binary strings to their corresponding values
        inv_bm: a dictionary mapping values to their corresponding binary strings
    >>> Example:
        >>> get_binmap(4, 'gray')
        ({'00': '00', '01': '01', '10': '11', '11': '10'}, {'00': '00', '01': '01', '11': '10', '10': '11'})
    '''
    b = discrete_dim // 2 - 1
    all_bins = []
    for i in range(1 << b):
        bx = np.binary_repr(i, width=discrete_dim // 2 - 1)
        all_bins.append('0' + bx)
        all_bins.append('1' + bx)
    vals = all_bins[:]
    if binmode == 'rand':
        print('remapping binary repr with random permute')
        random.shuffle(vals)
    elif binmode == 'gray':
        print('remapping binary repr with gray code')
        a = GrayCode(b)
        vals = []
        for x in a.generate_gray():
            vals.append('0' + x)
            vals.append('1' + x)
    else:
        assert binmode == 'normal'
    bm = {}
    inv_bm = {}
    for i, key in enumerate(all_bins):
        bm[key] = vals[i]
        inv_bm[vals[i]] = key
    return bm, inv_bm


def compress(x: float, discrete_dim: int) -> str:
    '''
    Compresses a float value into a binary string representation based on the discrete dimension.
    Args:
        x: a float value to be compressed
        discrete_dim: an integer representing the dimension of the discrete space
    Returns:
        bx: a binary string representation of the float value
    >>> Example:
        >>> compress(0.5, 4)
        '00'
    '''
    x = int(abs(x))
    bx = np.binary_repr(int(abs(x)), width=discrete_dim // 2 - 1)
    bx = '0' + bx if x >= 0 else '1' + bx
    return bx


def recover(bx: str) -> int:
    '''
    Recovers a float value from its binary string representation.
    Args:
        bx: a binary string representation of a float value
    Returns:
        x: the recovered float value
    >>> Example:
        >>> recover('00')
        0
    '''
    x = int(bx[1:], 2)
    return x if bx[0] == '0' else -x


def float2bin(samples: np.ndarray, bm: dict, int_scale: float, discrete_dim: int) -> np.ndarray:
    '''
    Converts a numpy array of float samples into their binary string representations based on the provided binary mapping and integer scale.
    Args:
        samples: a numpy array of float samples
        bm: a dictionary mapping binary strings to their compressed representations
        int_scale: a float value representing the integer scale
        discrete_dim: an integer representing the dimension of the discrete space
    Returns:
        bin_samples: a numpy array of binary string representations
    >>> Example:
        >>> float2bin(np.array([[0.5, -0.5], [-0.5, 0.5]]), bm, int_scale=1.0, discrete_dim=4)
        array([[0, 1, 0, 1],
               [1, 0, 1, 0]])
    '''
    bin_list = []
    for i in range(samples.shape[0]):
        x, y = samples[i] * int_scale
        bx, by = compress(x, discrete_dim), compress(y, discrete_dim)
        bx, by = bm[bx], bm[by]
        bin_list.append(np.array(list(bx + by), dtype=int))
    return np.array(bin_list)


def bin2float(samples: np.ndarray, inv_bm: dict, int_scale: float, discrete_dim: int):
    '''
    Converts a numpy array of binary string representations into their corresponding float values based on the provided inverse binary mapping and integer scale.
    Args:
        samples: a numpy array of binary string representations
        inv_bm: a dictionary mapping compressed binary strings to their original representations
        int_scale: a float value representing the integer scale
        discrete_dim: an integer representing the dimension of the discrete space
    Returns:
        float_samples: a numpy array of float values
    >>> Example:
        >>> bin2float(np.array([[0, 1, 0, 1],
        ...                     [1, 0, 1, 0]]), inv_bm, int_scale=1.0, discrete_dim=4)
        array([[ 0.5, -0.5],
               [-0.5,  0.5]])
    '''
    floats = []
    for i in range(samples.shape[0]):
        s = ''
        for j in range(samples.shape[1]):
            s += str(samples[i, j])
        x, y = s[:discrete_dim // 2], s[discrete_dim // 2:]
        x, y = inv_bm[x], inv_bm[y]
        x, y = recover(x), recover(y)
        x /= int_scale
        y /= int_scale
        floats.append((x, y))
    return np.array(floats)


def plot_heat(score_func: T.nn.Module, bm: dict, size: float, device: T.device, int_scale: float, discrete_dim: int, out_file: str = None):
    '''
    Plots a heatmap of the scores produced by the scoring function over a grid of points in the 2D space.
    Args:
        score_func: a PyTorch module representing the scoring function
        bm: a dictionary mapping binary strings to their compressed representations
        size: a float value representing the size of the grid in the 2D space
        device: a PyTorch device to run the scoring function on
        int_scale: a float value representing the integer scale
        discrete_dim: an integer representing the dimension of the discrete space
        out_file: a string representing the output file path to save the heatmap (optional)
    >>> Example:
        >>> plot_heat(score_func, bm, size=4.0, device=torch.device('cpu'), int_scale=1.0, discrete_dim=4, out_file='heatmap.png')
    '''
    w = 100
    x = np.linspace(-size, size, w)
    y = np.linspace(-size, size, w)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, [-1, 1])
    yy = np.reshape(yy, [-1, 1])
    heat_samples = float2bin(np.concatenate((xx, yy), axis=-1), bm, int_scale, discrete_dim)
    heat_samples = torch.from_numpy(heat_samples).to(device).float()
    heat_score = F.softmax(score_func(heat_samples).view(1, -1), dim=-1)
    a = heat_score.view(w, w).data.cpu().numpy()
    a = np.flip(a, axis=0)
    print("energy max and min:", a.max(), a.min())
    plt.imshow(a)
    plt.axis('equal')
    plt.axis('off')
    # if out_file is None:
    #     out_file = os.path.join(save_dir, 'heat.pdf')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def plot_samples(samples: np.ndarray, out_name: str, lim: float = None, axis: bool = True):
    '''
    Plots a scatter plot of the given samples in 2D space.
    Args:
        samples: a numpy array of shape (N, 2) representing the samples to plot
        out_name: a string representing the output file path to save the scatter plot
        lim: a float value representing the limit for the x and y axes (optional)
        axis: a boolean indicating whether to show the axes (optional, defaults to True)
    >>> Example:
        >>> plot_samples(np.array([[0.5, 1.0], [1.0, 0.5]]), out_name='samples.png', lim=2.0, axis=True)
    '''
    plt.scatter(samples[:, 0], samples[:, 1], marker='.')
    plt.axis('equal')
    if lim is not None:
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
    if not axis:
        plt.axis('off')
    plt.savefig(out_name, bbox_inches='tight')
    plt.close()


############# Model Architecture

class EnergyModel(T.nn.Module):
    '''
    A simple feedforward neural network model for energy-based modeling.
    '''

    def __init__(self, s: int, mid_size: int):
        super(EnergyModel, self).__init__()

        self.m = T.nn.Sequential(T.nn.Linear(s, mid_size),
                                 T.nn.ELU(),
                                 T.nn.Linear(mid_size, mid_size),
                                 T.nn.ELU(),
                                 T.nn.Linear(mid_size, mid_size),
                                 T.nn.ELU(),
                                 T.nn.Linear(mid_size, 1))

    def forward(self, x: T.Tensor) -> T.Tensor:
        '''
        Forward pass of the energy model.
        Args:
            x: a tensor of shape (N, D) representing the input samples
        Returns:
            output: a tensor of shape (N,) representing the energy scores for the input samples
        >>> Example:
            >>> model = EnergyModel(s=4, mid_size=256)
            >>> output = model(torch.tensor([[0.5, 1.0, -0.5, 0.0], [1.0, -1.0, 0.5, 0.5]]))
            >>> output.shape
            torch.Size([2])
        '''
        x = x.view((x.shape[0], -1))
        x = self.m(x)

        return x[:, -1]
