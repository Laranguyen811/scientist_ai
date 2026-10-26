from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import os, sys
import copy
import time
import random
import ipdb
from tqdm import tqdm
import argparse
import network

sys.path.append("./EB_GFN")
from gflownet import get_GFlowNet
import utils_data


def makedirs(path: str) -> None:
    '''
    Makes directory if it does not exist.
    Args:
        path (str): path to directory
    Returns:
        None
    >>> Example:
    >>> makedirs("./test")
    '''
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")

class EBM(nn.Module):
    '''
    Class for Energy-Based Model (EBM) that combines a neural network with an optional base distribution.
    '''
    def __init__(self, net: nn.Module, mean: torch.Tensor=None):
        super().__init__()
        self.net = net
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)
            self.base_dist = torch.distributions.Bernoulli(probs=self.mean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            bd = 0.
        else:
            bd = self.base_dist.log_prob(x).sum(-1)

        logp = self.net(x).squeeze()
        forward_logp = logp + bd
        return forward_logp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "--d", default=0, type=int)
    # data
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--data', type=str, default='dmnist')
    parser.add_argument("--down_sample", "--ds", default=0, type=int, choices=[0, 1])
    parser.add_argument('--ckpt_path', type=str, default=None)
    # models
    parser.add_argument('--model', type=str, default='mlp-256')
    parser.add_argument('--base_dist', "--bd", type=int, default=1, choices=[0, 1])
    parser.add_argument('--gradnorm', "--gn", type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--n_iters', "--ni", type=lambda x: int(float(x)), default=5e4)
    parser.add_argument('--batch_size', "--bs", type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--print_every', "--pe", type=int, default=100)
    parser.add_argument('--viz_every', "--ve", type=int, default=2000)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument("--ebm_every", "--ee", type=int, default=1, help="EBM training frequency")

    # for GFN
    parser.add_argument("--type", type=str)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--hid_layers", "--hl", type=int, default=5)
    parser.add_argument("--leaky", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gfn_bn", "--gbn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--init_zero", "--iz", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gmodel", "--gm", type=str, default="mlp")
    parser.add_argument("--train_steps", "--ts", type=int, default=1)
    parser.add_argument("--l1loss", "--l1l", type=int, default=0, choices=[0, 1], help="use soft l1 loss instead of l2")

    parser.add_argument("--with_mh", "--wm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--rand_k", "--rk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lin_k", "--lk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--warmup_k", "--wk", type=lambda x: int(float(x)), default=0, help="need to use w/ lin_k")
    parser.add_argument("--K", type=int, default=-1, help="for gfn back forth negative sample generation")

    parser.add_argument("--rand_coef", "--rc", type=float, default=0, help="for tb")
    parser.add_argument("--back_ratio", "--br", type=float, default=0.)
    parser.add_argument("--clip", type=float, default=-1., help="for gfn's linf gradient clipping")
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--opt", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--glr", type=float, default=1e-3)
    parser.add_argument("--zlr", type=float, default=1e-1)
    parser.add_argument("--momentum", "--mom", type=float, default=0.0)
    parser.add_argument("--gfn_weight_decay", "--gwd", type=float, default=0.0)
    parser.add_argument('--mc_num', "--mcn", type=int, default=5)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.device)
    device = torch.device("cpu") if args.device < 0 else torch.device("cuda")

    args.device = device
    args.save_dir = os.path.join(args.save_dir, "test")
    makedirs(args.save_dir)

    print("Device:" + str(device))
    print("Args:" + str(args))

    before_load = time.time()
    train_loader, val_loader, test_loader, args = utils_data.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), args.input_size[0],
        args.input_size[1], args.input_size[2]), p, normalize=True, nrow=int(x.size(0) ** .5))
    print(f"It takes {time.time() - before_load:.3f}s to load {args.data} dataset.")

    def preprocess(data: torch.Tensor) -> torch.Tensor:
        '''
        Preprocesses the input data based on the specified arguments.
        Args:
            data (torch.Tensor): Input data tensor.
        Returns:
            torch.Tensor: Preprocessed data tensor.
        >>> Example:
            >>> preprocess(torch.rand(10, 1, 28, 28))
            tensor([[0., 1., 0., ..., 1., 0., 1.],
                    [1., 0., 1., ..., 0., 1., 0.],
                    ...,
                    [0., 1., 0., ..., 1., 0., 1.]])
        '''
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
        
    def create_net(args: argparse.Namespace)-> nn.Module:
        '''
        Creates the neural network based on the provided arguments.
        Args:
            args: Parsed command-line arguments.
        Returns:
            torch.nn.Module: Initialized neural network.
        Raises:
            AssertionError: If any argument is invalid.
            ValueError: If any argument is invalid.
        >>> Example:
            >>> create_net(args)
            MLP_EBM()
        '''
        if args.down_sample:
            assert args.model.startswith("mlp-")

        if args.model.startswith("mlp-"):
            nint = int(args.model.split('-')[1])
            net = network.mlp_ebm(np.prod(args.input_size), nint)
        elif args.model.startswith("cnn-"):
            nint = int(args.model.split('-')[1])
            net = network.MNISTConvNet(nint)
        elif args.model.startswith("resnet-"):
            nint = int(args.model.split('-')[1])
            net = network.ResNetEBM(nint)
        else:
            raise ValueError("invalid model definition")
        return net

    def initialize_batch(args: argparse.Namespace, net: torch.nn.Module) -> tuple[torch.nn.Module, torch.Tensor]:
        '''
        Initializes the EBM model based on the provided arguments.
        Args:
            args: Parsed command-line arguments.
            net: The neural network architecture for the EBM.
        Returns:
            tuple[torch.nn.Module, torch.Tensor]: Initialized EBM and the initial batch.
        >>> Example:
            >>> initialize_batch(args, net)
            (EBM(), tensor([[0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5],
                            ...,
                            [0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5]]))
        '''
        init_batch = []
        for x, _ in train_loader:
            init_batch.append(preprocess(x))
        init_batch = torch.cat(init_batch, 0)
        eps = 1e-2
        init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps

        return net, init_mean
    
    def create_model(args: argparse.Namespace, net: torch.nn.Module, init_mean: torch.Tensor) -> EBM:
        '''
        Creates the EBM model based on the provided arguments.
        Args:
            args: Parsed command-line arguments.
            net: The neural network architecture for the EBM.
            init_mean: The initial mean for the base distribution.
        Returns:
            EBM: Initialized EBM model.
        >>> Example:
            >>> create_model(args, net, init_mean)
            EBM()
        '''
        if args.base_dist:
            model = EBM(net, init_mean)
        else:
            model = EBM(net)
        
        return model
    
    def create_optimiser(model: EBM, args: argparse.Namespace) -> torch.optim.Optimizer:
        '''
        Creates the optimizer for the EBM model based on the provided arguments.
        Args:
            model: The EBM model.
            args: Parsed command-line arguments.
        Returns:
            torch.optim.Optimizer: Initialized optimizer.
        >>> Example:
            >>> create_optimiser(model, args)
            Adam (
                Parameter Group 0
                    amsgrad: False
                    betas: (0.9, 0.999)
                    eps: 1e-08
                    lr: 0.0001
                    weight_decay: 0
            )
        '''
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
        return optimiser
    
    def create_gfn(args: argparse.Namespace, model: EBM) -> get_GFlowNet:
        '''
        Creates the GFlowNet model based on the provided arguments.
        Args:
            args: Parsed command-line arguments.
            model: The EBM model.
        Returns:
            get_GFlowNet: Initialized GFlowNet model.
        >>> Example:
            >>> create_gfn(args, model)
            get_GFlowNet()
        '''
        xdim = np.prod(args.input_size)
        assert args.gmodel == "mlp"
        gfn = get_GFlowNet(args.type, xdim, args, device)
        model.to(device)
        print("model: {:}".format(model))
        return gfn

    def train_loop(args: argparse.Namespace, model: EBM, gfn: get_GFlowNet, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        '''
        Main training loop for the EBM and GFlowNet models.
        Args:
            args: Parsed command-line arguments.
            model: The EBM model.
            gfn: The GFlowNet model.
            optimizer: The optimizer for the EBM model.
        Returns:
            Tuple[float, float]: Final training loss and logZ.
        >>> Example:
            >>> train_loop(args, model, gfn, optimizer)
            (0.1234, -5.6789)
        '''
        itr = 0
        while itr < args.n_iters:
            for x in train_loader:
                st = time.time()
                x = preprocess(x[0].to(device))  #  -> (bs, 784)

                if args.gradnorm > 0:
                    x.requires_grad_()

                update_success_rate = -1.
                assert "tb" in args.type
                train_loss, train_logZ = gfn.train(args.batch_size, scorer=lambda inp: model(inp).detach(),
                    silent=itr % args.print_every != 0, data=x, back_ratio=args.back_ratio)
        return train_loss, train_logZ

    def create_k(args: argparse.Namespace, xdim: int, itr: int) -> int:
        '''
        Creates the value of K for GFlowNet back-forth sampling based on the provided arguments.
        Args:
            args: Parsed command-line arguments.
            xdim: Dimensionality of the input data.
            itr: Current iteration number.
        Returns:
            int: Value of K for back-forth sampling.
        >>> Example:
            >>> create_k(args, xdim, itr)
            5
        '''    
        if args.rand_k or args.lin_k or (args.K > 0):
                if args.rand_k:
                    K = random.randrange(xdim) + 1
                elif args.lin_k:
                    K = min(xdim, int(xdim * float(itr + 1) / args.warmup_k))
                    K = max(K, 1)
                elif args.K > 0:
                    K = args.K
                else:
                    raise ValueError
        return K

    def gfn_eval(args: argparse.Namespace, gfn: get_GFlowNet, test_loader: torch.utils.data.DataLoader, preprocess: callable) -> float:
        '''
        Evaluates the GFlowNet model on the test dataset.
        Args:
            args: Parsed command-line arguments.
            gfn: The GFlowNet model.
            test_loader: DataLoader for the test dataset.
            preprocess: Preprocessing function for the input data.
        Returns:
            float: Test log-likelihood.
        >>> Example:
            >>> gfn_eval(args, gfn, test_loader, preprocess)
            -5.6789
        '''    
        gfn.model.eval()
        gfn_test_ll = gfn.evaluate(test_loader, preprocess, args.mc_num)
        print("GFN Test log-likelihood with {} samples: {}".format(args.mc_num, gfn_test_ll.item()))


    def calculate_delta_logp_traj(args: argparse.Namespace, gfn: get_GFlowNet, x: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Calculates the delta log-probability trajectory for GFlowNet back-forth sampling.
        Args:
            args: Parsed command-line arguments.
            gfn: The GFlowNet model.
            x: Input data tensor.
            K: Number of back-forth steps.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled data and delta log-probability trajectory.
        >>> Example:
            >>> calculate_delta_logp_traj(args, gfn, x, K)
            (tensor([[0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5],
                    ...,
                    [0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5]]),
                tensor([[0.1, 0.2, 0.3, ..., 0.4, 0.5, 0.6],
                        [0.1, 0.2, 0.3, ..., 0.4, 0.5, 0.6],
                        ...,
                        [0.1, 0.2, 0.3, ..., 0.4, 0.5, 0.6]]))
        '''
        x_fake, delta_logp_traj = gfn.backforth_sample(x, K)

        delta_logp_traj = delta_logp_traj.detach()
        return x_fake, delta_logp_traj

    def calculate_update_success_rate(args: argparse.Namespace, 
                                      gfn: get_GFlowNet, 
                                      x: torch.Tensor, 
                                      x_fake: torch.Tensor, 
                                      delta_logp_traj: torch.Tensor) -> float:
        '''
        Calculates the update success rate for GFlowNet back-forth sampling with Metropolis-Hastings step.
        Args:
            args: Parsed command-line arguments.
            gfn: The GFlowNet model.
            x: Input data tensor,
            x: Input data tensor.
            x_fake: Sampled data tensor.
            delta_logp_traj: Delta log-probability trajectory tensor.
        Returns:
            float: Update success rate.
        >>> Example:
            >>> calculate_update_success_rate(args, model, x, x_fake, delta_logp_traj)
            0.75    
        '''
        if args.with_mh:
        # MH step, calculate log p(x') - log p(x)
            lp_update = gfn.model(x_fake).squeeze() - gfn.model(x).squeeze()
            update_dist = torch.distributions.Bernoulli(logits=lp_update + delta_logp_traj)
            updates = update_dist.sample()
            x_fake = x_fake * updates[:, None] + x * (1. - updates[:, None])
            update_success_rate = updates.mean().item()

        else:
            x_fake = gfn.sample(args.batch_size)

        return x_fake, update_success_rate
    def train_model(self,
                    model:EBM,
                    gfn: get_GFlowNet,
                    x: torch.Tensor,
                    x_fake:torch.Tensor, 
                    optimizer: torch.optim.Optimizer,
                    update_success_rate: float,
                    )-> Tuple[float, float]:
        '''
        Trains Energy-based GFlowNet model. 
        Args:
            model (EBM): Energy-based model.
            gfn (get_GFlowNet): GFlowNet instance.
            x (torch.Tensor): Input data tensor.
            x_fake (torch.Tensor): Sampled data tensor.
            optimizer (torch.optim.Optimizer): Optimizer for the EBM model.
            update_success_rate (float): Update success rate from GFlowNet sampling.
        Returns:
            Tuple[float, float]: Loss value and GFlowNet logZ value.
        >>> Example:
            >>> train_model(model, gfn, x, x_fake, optimizer, update_success_rate)
            (0.123, 1.456)
        ''' 
        if itr % args.ebm_every == 0:
            st = time.time() - st

            model.train()
            logp_real = model(x).squeeze()
            if args.gradnorm > 0:
                grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                      create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
                grad_reg = (grad_ld ** 2. / 2.).mean()
            else:
                grad_reg = torch.tensor(0.).to(device)

            logp_fake = model(x_fake).squeeze()
            obj = logp_real.mean() - logp_fake.mean()
            l2_reg = (logp_real ** 2.).mean() + (logp_fake ** 2.).mean()
            loss = -obj + grad_reg * args.gradnorm + args.l2 * l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if itr % args.print_every == 0 or itr == args.n_iters - 1:
            print("({:5d}) | ({:.3f}s/iter) |log p(real)={:.2e}, "
                 "log p(fake)={:.2e}, diff={:.2e}, grad_reg={:.2e}, l2_reg={:.2e} update_rate={:.1f}".format(itr, st,
                 logp_real.mean().item(), logp_fake.mean().item(), obj.item(), grad_reg.item(), l2_reg.item(), update_success_rate))

        if (itr + 1) % args.eval_every == 0:
            model.eval()
            print("GFN TEST")
            gfn.model.eval()
            gfn_test_ll = gfn.evaluate(test_loader, preprocess, args.mc_num)
            print("GFN Test log-likelihood ({}) with {} samples: {}".format(itr, args.mc_num, gfn_test_ll.item()))

            model.cpu()
            d = {}
            d['model'] = model.state_dict()
            d['optimizer'] = optimizer.state_dict()
            gfn_ckpt = {"model": gfn.model.state_dict(), "optimizer": gfn.optimizer.state_dict(),}
            gfn_ckpt["logZ"] = gfn.logZ.detach().cpu()
            torch.save(d, "{}/ckpt.pt".format(args.save_dir))
            torch.save(gfn_ckpt, "{}/gfn_ckpt.pt".format(args.save_dir))

            model.to(device)

        itr += 1
        if itr > args.n_iters:
            print("Training finished!")
            return loss.item(), gfn.logZ.detach().cpu().item()
