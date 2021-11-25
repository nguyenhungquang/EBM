#%%
import numpy as np
import torch
from torchsummary import summary
from PIL import Image
import matplotlib.pyplot as plt
from sample import *
from models.base import *
from models.ncsn import *
from models.ncsnv2 import *
device = torch.device("cuda:0")

#%%
model_name = UnetNCSN
sigmas = np.exp(np.linspace(np.log(1.), np.log(1e-2), 10))
model = model_name(1, torch.tensor(sigmas)).to(device)
model.load_state_dict(torch.load(f'anneal_dsm_mnist_{model_name.__name__}_v2.pth', map_location=str(device)))


#%%
img_list = annealed_Langevin_dynamics(model, [20, 1, 28, 28], sigmas, base_step_size=3e-6, n_steps_each = 100, device=device)
img = img_list[-1]
fig = plt.figure(figsize=(8, 8))
for i in range(len(img)):
    im = img[i].permute(1, 2, 0).cpu().numpy()

    fig.add_subplot(4, 5, i + 1)
    plt.imshow(im, cmap='gray')

