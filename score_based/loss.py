import torch
def dsm(model, input, var = 1e-2):
    perturbed = input + torch.randn_like(input) * var
    scores = model(perturbed)
    target = (input - perturbed) / (var ** 2)
    batch_size = scores.shape[0]
    scores = scores.view(batch_size, -1)
    target = target.view(batch_size, -1)
    loss = ((scores - target) ** 2).sum()
    return loss / batch_size

def anneal_dsm(model, input, noise_labels, sigmas, anneal_power = 2.):
    batch_size = input.shape[0]
    noise_var = sigmas[noise_labels].view(batch_size, *([1] * len(input.shape[1:])))
    perturbed = input + torch.randn_like(input) * noise_var
    target = (input - perturbed) / (noise_var ** 2)
    scores = model(perturbed, noise_labels)
    target = target.view(batch_size, -1)
    scores = scores.view(batch_size, -1)
    loss = ((scores - target) ** 2).sum(dim = 1) * noise_var.squeeze() ** anneal_power
    return loss.mean()