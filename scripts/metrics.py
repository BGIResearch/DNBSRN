import os
import time
import torch
import numpy as np
from model_summary import get_model_flops, get_model_activation


def metrics_calculate(opt, net):
    image = np.float32(np.random.randn(1, 1, 500, 500))
    torch.cuda.empty_cache()
    iterations = 300
    net = net.to(opt.device)
    net.eval()
    image = torch.tensor(image).to(opt.device)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = torch.zeros(iterations)
    with torch.no_grad():
        for _ in range(50):
            _ = net(image)
        for iter in range(iterations):
            starter.record()
            _ = net(image)
            ender.record()
            torch.cuda.synchronize()
            times[iter] = starter.elapsed_time(ender)
    mean_time = times.mean().item()
    opt.logger.info(fr'Inference Time: {mean_time:.2f} [ms]')
    max_memory = torch.cuda.max_memory_allocated()/1024**2
    opt.logger.info(fr'Max Memery: {max_memory:.2f} [M]')
    input_dim = (1, 500, 500)
    activations, num_conv = get_model_activation(net, input_dim)
    activations = activations/10**6
    opt.logger.info(fr'Activations: {activations:.2f} [M]')
    opt.logger.info(fr'Conv2d: {num_conv:d}')
    flops = get_model_flops(net, input_dim, False)
    flops = flops/10**9
    opt.logger.info(fr'FLOPs: {flops:.2f} [G]')
    num_parameters = sum(map(lambda x: x.numel(), net.parameters()))
    num_parameters = num_parameters/10**6
    opt.logger.info(fr'Params: {num_parameters:.2f} [M]')
