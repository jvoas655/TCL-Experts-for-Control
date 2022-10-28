import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
from tqdm import tqdm
import matplotlib as mpl
from sklearn.decomposition import PCA

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
                if (locs[point1][0] == 0 and locs[point2][0] == len(locs)-1 and locs[point1][1] == locs[point2][1]):
                    adjs[point1, point2] = True
                    adjs[point2, point1] = True
                if (locs[point2][0] == 0 and locs[point1][0] == len(locs)-1 and locs[point1][1] == locs[point2][1]):
                    adjs[point1, point2] = True
                    adjs[point2, point1] = True

                if (locs[point1][1] == 0 and locs[point2][1] == len(locs)-1 and locs[point1][0] == locs[point2][0]):
                    adjs[point1, point2] = True
                    adjs[point2, point1] = True
                if (locs[point2][1] == 0 and locs[point1][1] == len(locs)-1 and locs[point1][0] == locs[point2][0]):
                    adjs[point1, point2] = True
                    adjs[point2, point1] = True
            if (dist[point1, point2] > 1 / (n - 1) - e and dist[point1, point2] < 1 / (n - 1) + e):
                adjs[point1, point2] = True
                adjs[point2, point1] = True
        
                
    return torch.tensor(locs), torch.tensor(values), adjs

class SOM(torch.nn.Module):
    def __init__(self, map_nodes_locs, map_nodes_adj, map_node_values, adj_decay, dist_scale_func = None, apply_dist_through_adj = False, max_steps = 3):
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
        # Create boolean arrway of which size matching a single row of the adjacency list (size = (N,)) init to False
        # This array will be used to track which nodes have already had their values updated in this backwards step
        adjusted_nodes = torch.zeros_like(self.map_nodes_adj[0, :])
        # If we do not wish to apply distance value updates through adjacency AND we are provided a distance function
        if (self.dist_scale_func is not None and not self.apply_dist_through_adj):
            # Find the value offset for all node values (1, H) - (N, H) and adjust by learning rate
            lr_adjusted_value_offset = lr * (value - self.map_node_values)
            # Find the euclidean distance between all N node locations and the sample value ((N, D) - (1, D))
            dists = torch.sqrt(torch.sum(torch.pow(self.map_nodes_locs - self.map_nodes_locs[ind], 2), dim = 1))
            # Pass distances into the scale function to get distance scaling value, and multiply by the learning rate adjusted value offset
            # This means if LR = 1 and dist_scale_func(dist) = 1 for some node, its value would be equal to the sample value
            self.map_node_values += lr_adjusted_value_offset * self.dist_scale_func(dists)[:, None]
        else:
            # Save original index for use later in calculated distance scaling thorugh adjacency steps
            orig_ind = ind
        # TODO: Add small offset to stop zero blocking
        initial_value_offset_mag = torch.sqrt(torch.sum(torch.pow(value - self.map_node_values[ind], 2), dim = 1)).item() + 10e-6
        # This loop will cycle through sets of nodes in the map. For each step it will update a set of nodes.
        # On the next loop it will target adjacent AND non-updated nodes. This continues until all nodes are updated (or max steps)
        step = 0
        while ((step < self.max_steps or self.max_steps == -1) and torch.any(adjusted_nodes == False)):
            # Find the value offset for all node values (1, H) - (N, H) and adjust by learning rate
            value_offset = value - self.map_node_values[ind]
            value_offset_mag = torch.sqrt(torch.sum(torch.pow(value_offset, 2), dim = 1))
            value_offset_mag = torch.where(value_offset_mag == 0, 1, value_offset_mag)
            val_offset_mag_adjustments = (initial_value_offset_mag / value_offset_mag)
            lr_adjusted_value_offset = lr * value_offset * val_offset_mag_adjustments[:, None]
            # If a distance scaling function is given
            if (self.dist_scale_func is not None and self.apply_dist_through_adj):
                # Find the euclidean distance between all current step node locations and the sample value ((k, D) - (1, D))
                dists = torch.sqrt(torch.sum(torch.pow(self.map_nodes_locs[ind] - self.map_nodes_locs[orig_ind], 2), dim = 1))
                # Pass distances into the scale function to get distance scaling value, and multiply by the learning rate adjusted value offset
                # Also multiply by the adjacency decay value (< 1) to the power of the current step. 
                # This means if LR = 1 and dist_scale_func(dist) = 1 and either adj_decay = 1 or step = 0 for some node, its value would be equal to the sample value
                self.map_node_values[ind] += (self.adj_decay ** step) * lr_adjusted_value_offset * self.dist_scale_func(dists)[:, None]
            else:
                # Similar to above update but no consideration for distance scaling
                self.map_node_values[ind] += (self.adj_decay ** step) * lr_adjusted_value_offset
            # Track that we have updated the value of the nodes in ind for this step
            adjusted_nodes[ind] = True
            # Get a new boolean set for all nodes adjacent to our current ind step. Mask out these nodes if they have already been updated
            ind = torch.where(adjusted_nodes == False, torch.any(self.map_nodes_adj[ind, ...], dim = 0), False)
            #print(adjusted_nodes)
            step += 1
    def plot(self):
        cmap = mpl.colormaps["inferno"]
        locs = self.map_nodes_locs.clone().detach().cpu().numpy()
        vals = self.map_node_values.clone().detach().cpu().numpy()
        pca = PCA(1)
        tvals = pca.fit_transform(vals)
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #for i in range(len(locs)):
        #    ax.scatter(*locs[i, :], c=tvals[i, 0], marker = "o")
        plt.scatter(locs[:, 0], locs[:, 1], c = tvals, cmap = cmap)
        plt.show()



if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    device = "cpu"
    #if (torch.cuda.is_available()):
    #    device = f"cuda:0"
    print(device)
    samples = torch.randn((800, 3)).to(device = device)
    locs, values, adjs = construct_grid_map(40, 2, 3, samples, False)
    dist_scale_func = lambda d: torch.pow(torch.where(torch.clamp(1 - d, min=0) == 1, 1, torch.clamp(1 - d, min=0) / 1), 2)
    som = SOM(locs, adjs, values, adj_decay = 0.8, dist_scale_func = dist_scale_func, apply_dist_through_adj = True, max_steps=1).to(device = device)

    #print(samples)
    #print(som.map_node_values)
    t = time.time()
    print("Start")
    for i in range(1000000):
        sample = random.randint(0, samples.shape[0]-1)
        inds, dist = som(samples[sample, :], 2)
        inds = inds[..., None]
        for ind in range(len(inds)):
            som.backward(inds[ind, ...], samples[sample, :], max(10e-6, 10e-2 * (100000 / (100000 + i))) / (ind + 1) ** 2)
            break
        if (i % 100 == 0):
            print(i, time.time() - t)
            t = time.time()
        if (i % 20000 == 0):
            dists = []
            for sample in range(samples.shape[0]):
                _, dist = som(samples[sample, :])
                dists.append(dist)
            
            print(i, torch.mean(torch.abs(torch.cat(dists))))
            som.plot()
            if (som.max_steps == 1):
                som.max_steps = 40
            else:
                som.max_steps = 1

        