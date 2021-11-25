from tqdm import tqdm
import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import make_grid, save_image

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models.base import *
from models.ncsn import *
from models.ncsnv2 import *
from loss import *
from sample import *

def init_process(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=2)
    main(rank, args)

def main(gpu, args):
    if args.dataset == 'mnist':
        data = MNIST
        channels = 1
        model_name = ResNCSN
        sigma_0 = 1.
    else:
        model_name = UnetNCSN
        data = CIFAR10
        channels = 3
        sigma_0 = 50.
        args.n_noise_levels = 200 

    save_path = f"{args.loss}_{args.dataset}_{model_name.__name__}_v2.pth"
    device = torch.device(f"cuda:{gpu}") if gpu >=0 else torch.device('cpu')
        
    transform = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor()
            ])
    dataset = data('../data', train=True, download=True, transform=transform)
    test_dataset = data('../data', train=False, download=True, transform=transform)

    sigmas_np = np.exp(np.linspace(np.log(sigma_0), np.log(1e-2), args.n_noise_levels))
    sigmas = torch.tensor(sigmas_np, dtype = torch.float, device = device)
    model = model_name(channels, sigmas)
    model.to(device)
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle = True, drop_last = True)
        dataloader = DataLoader(dataset, batch_size = args.batch_size, sampler = sampler, num_workers = args.num_workers, pin_memory = True)
        model = DDP(model, device_ids=[gpu])
    else:
        dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, pin_memory = True)
    testloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, pin_memory = True)

    # model = ResNet()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    it = tqdm(range(args.epochs))
    for i in it:
        mean_loss = []
        if args.distributed:
            dataloader.sampler.set_epoch(i)
        for data, _ in dataloader:
            model.train()
            optimizer.zero_grad()
            data = data.to(device)
            # loss = dsm(model, data)
            noise_labels = torch.randint(0, len(sigmas), (data.shape[0],), device=device)
            loss = anneal_dsm(model, data, noise_labels, sigmas)
            loss.backward()
            optimizer.step()
            mean_loss.append(loss.item())
        it.set_description(f"Epoch: {i}, Loss: {np.mean(mean_loss)}")
        if i % 50 == 0 and gpu == 0:
            # imgs = Langevin_dynamics(model, [20, 1, 28, 28], device = device)
            imgs = annealed_Langevin_dynamics(model, [20, channels, 28, 28], sigmas_np, base_step_size=5e-6, n_steps_each=100, device = device)
            img = imgs[-1]
            grid = make_grid(img, 4)
            save_image(grid, f"{model_name.__name__}_base_{i}.png")
        if args.distributed:
            if gpu == 0:
                torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)

    if args.distributed:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--distributed", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--loss", type=str, default='anneal_dsm')
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-noise-levels", type=int, default=10)
    args = parser.parse_args()
    print(args)
    if not args.distributed or args.gpu < 0 or not torch.distributed.is_available():
        args.distributed = False
        main(args.gpu, args)
    else:
        mp.spawn(
            init_process,
            args=(args, ),
            nprocs=2
        )
    