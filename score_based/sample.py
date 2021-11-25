import torch
import numpy as np
from tqdm import tqdm

def Langevin_dynamics(model, data_size, step_size=1e-4, step=200, device = None):
    model.eval()
    img = []
    # model = model.cpu()
    with torch.no_grad():
        # device = torch.device('cuda')
        # step_size = torch.tensor(step_size)#.cuda()
        sample = torch.empty(data_size).normal_().to(device)
        # print(model.device)
        # model = model.cpu()
        img.append(sample)
        # print(model.device, sample.device)
        for i in range(step):
            noise = torch.randn_like(sample)#.cuda()
            sample += step_size * model(sample) + np.sqrt(step_size * 2) * noise
            img.append(torch.clamp(sample, 0.0, 1.0).to('cpu'))

    return img

def annealed_Langevin_dynamics(model, data_size, sigmas, base_step_size = 1e-5, n_steps_each = 100, device = None):
    model.eval()
    img = []
    with torch.no_grad():
        sigma_L = sigmas[-1]
        sample = torch.randn(data_size, device = device)
        for c, sigma in tqdm(enumerate(sigmas), total=len(sigmas)):
            noise_labels  = torch.ones(sample.shape[0], device=device, dtype=torch.long) * c
            step_size = base_step_size * (sigma / sigma_L) ** 2
            for _ in range(n_steps_each):
                img.append(torch.clamp(sample, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(sample) * np.sqrt(step_size * 2)
                
                grad = model(sample, noise_labels).float()
                # print(sample.type(), grad.type(), noise.type())
                assert not torch.isnan(grad).sum()
                sample = sample + step_size * grad + noise
    return img