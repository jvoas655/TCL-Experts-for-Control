import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
from tqdm import tqdm
import matplotlib as mpl

def construct_grid_map(n, map_dims=2, value_dims = 5, init = None, edge_connects = True):
    locs = []
    for dim in range(map_dims):
        locs.append(np.linspace(0, 1, n))
    locs = np.concatenate(list(map(lambda l: l.reshape(1, -1), np.meshgrid(*locs)))).transpose()
    values = np.random.randn(n ** map_dims *  value_dims).reshape(-1, value_dims)
    if (init is not None):
        for sample in range(min(len(values), len(init))):
            values[sample, :] = init[sample, :].detach().cpu().numpy()
    adjs = torch.zeros((len(locs), len(locs)), dtype = torch.bool)
    e = 10e-9
    dist = distance.cdist(locs, locs, 'euclidean')
    for point1 in tqdm(range(len(locs)), total = len(locs)):
        for point2 in range(len(locs)):
            if (edge_connects):
                if (point1 == 0):
                    adjs[len(locs) - 1, point2] = 1
                    adjs[point2, len(locs) - 1] = 1
                if (point2 == 0):
                    adjs[len(locs) - 1, point1] = 1
                    adjs[point1, len(locs) - 1] = 1
                if (point1 == len(locs) - 1):
                    adjs[0, point2] = 1
                    adjs[point2, 0] = 1
                if (point2 == len(locs) - 1):
                    adjs[0, point1] = 1
                    adjs[point1, 0] = 1
            if (dist[point1, point2] > 1 / (n - 1) - e and dist[point1, point2] < 1 / (n - 1) + e):
                adjs[point1, point2] = True
                adjs[point2, point1] = True
        
                
    return torch.tensor(locs), torch.tensor(values), adjs

class SOM(torch.nn.Module):
    def __init__(self, map_nodes_locs, map_nodes_adj, map_node_values, adj_decay, dist_scale_func = None, apply_dist_through_adj = True, max_steps = 3):
        super().__init__()
        self.map_node_values = map_node_values
        self.map_nodes_locs = map_nodes_locs
        self.map_nodes_adj = map_nodes_adj
        self.adj_decay = adj_decay
        self.max_steps = max_steps
        self.dist_scale_func = dist_scale_func
        self.apply_dist_through_adj = apply_dist_through_adj
    def to(self, *args, **kwargs):
        super().to(**kwargs)
        if ("device" in kwargs):
            self.map_node_values = self.map_node_values.to(device = kwargs["device"])
            self.map_nodes_locs = self.map_nodes_locs.to(device = kwargs["device"])
            self.map_nodes_adj = self.map_nodes_adj.to(device = kwargs["device"])
        return self
    def forward(self, samples, n = 1):
        dist = torch.norm(samples - self.map_node_values, dim=1, p=None)
        knn = dist.topk(n, largest=False)
        return knn.indices, knn.values
    def backward(self, ind, value, lr):
        adjusted_nodes = torch.zeros_like(self.map_nodes_adj[0, :])
        if (self.dist_scale_func is not None and not self.apply_dist_through_adj):
            lr_adjusted_value_offset = lr * (value - self.map_node_values)
            dists = torch.sum(torch.pow(self.map_nodes_locs - self.map_nodes_locs[ind], 2), dim = 1)
            self.map_node_values +=  lr_adjusted_value_offset * self.dist_scale_func(dists)[:, None]
        else:
            orig_ind = ind
        step = 0
        while ((step < self.max_steps or self.max_steps == -1) and torch.any(adjusted_nodes == False)):
            lr_adjusted_value_offset = lr * (value - self.map_node_values[ind])
            if (self.dist_scale_func is not None and self.apply_dist_through_adj):
                dists = torch.sum(torch.pow(self.map_nodes_locs[ind] - self.map_nodes_locs[orig_ind], 2), dim = 1)
                self.map_node_values[ind] += (self.adj_decay ** step) * lr_adjusted_value_offset * self.dist_scale_func(dists)[:, None]
            else:
                self.map_node_values[ind] += (self.adj_decay ** step) * lr_adjusted_value_offset
            adjusted_nodes[ind] = True
            ind = torch.where(adjusted_nodes == False, torch.any(self.map_nodes_adj[ind, ...], dim = 0), False)
            #print(adjusted_nodes)
            step += 1
    def plot(self):
        cmap = mpl.colormaps["inferno"]
        locs = self.map_nodes_locs.clone().detach().cpu().numpy()
        vals = self.map_node_values.clone().detach().cpu().numpy()
        for v in range(vals.shape[1]):
            plt.scatter(locs[:, 0], locs[:, 1], c = vals[:, v], cmap = cmap)
            plt.show()



if __name__ == "__main__":
    np.random.seed(42)
    device = "cpu"
    if (torch.cuda.is_available()):
        device = f"cuda:0"
    samples = torch.randn((10000, 6)).to(device = device)
    locs, values, adjs = construct_grid_map(40, 2, 6, samples, False)
    dist_scale_func = lambda d: torch.pow(torch.where(torch.clamp(1 - d, min=0) == 1, 1, torch.clamp(1 - d, min=0) / 1), 2)
    som = SOM(locs, adjs, values, adj_decay = 0.2, dist_scale_func = dist_scale_func, max_steps=10).to(device = device)

    print(samples)
    print(som.map_node_values)
    t = time.time()
    print("Start")
    for i in range(1000000):
        sample = random.randint(0, samples.shape[0]-1)
        ind, dist = som(samples[sample, :])
        som.backward(ind, samples[sample, :], max(10e-6, 10e-2 * (100000 / (100000 + i))))
        if (i % 100 == 0):
            print(i, time.time() - t)
            t = time.time()
        if (i % 30000 == 0):
            dists = []
            for sample in range(samples.shape[0]):
                _, dist = som(samples[sample, :])
                dists.append(dist)
            
            print(i, torch.mean(torch.abs(torch.cat(dists))))
            som.plot()
            som.max_steps = max(0, som.max_steps-1)

        