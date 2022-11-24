import random
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
from tqdm import tqdm
import matplotlib as mpl
from sklearn.decomposition import PCA
import anti_lib_progs.geodesic as geod
import h5py

def construct_grid_map(n, map_dims=2, value_dims = 5, init = None, edge_connects = True):
    locs = []
    for dim in range(map_dims):
        locs.append(np.linspace(0, 1, n))
    locs = np.concatenate(list(map(lambda l: l.reshape(1, -1), np.meshgrid(*locs)))).transpose() * 2 - 1
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
                if (locs[point1][0] == -1 and locs[point2][0] == len(locs)-1 and locs[point1][1] == locs[point2][1]):
                    adjs[point1, point2] = True
                    adjs[point2, point1] = True
                if (locs[point2][0] == -1 and locs[point1][0] == len(locs)-1 and locs[point1][1] == locs[point2][1]):
                    adjs[point1, point2] = True
                    adjs[point2, point1] = True

                if (locs[point1][1] == -1 and locs[point2][1] == len(locs)-1 and locs[point1][0] == locs[point2][0]):
                    adjs[point1, point2] = True
                    adjs[point2, point1] = True
                if (locs[point2][1] == -1 and locs[point1][1] == len(locs)-1 and locs[point1][0] == locs[point2][0]):
                    adjs[point1, point2] = True
                    adjs[point2, point1] = True
            if (dist[point1, point2] > 2 / (n - 1) - e and dist[point1, point2] < 2 / (n - 1) + e):
                adjs[point1, point2] = True
                adjs[point2, point1] = True
    locs, values = torch.tensor(locs), torch.tensor(values)
    mask = torch.ones_like(adjs[:, 0])
    for i in range(n ** map_dims):
        if (torch.all(torch.abs(locs[i, :]) < 1)):
            mask[i] = False
    locs = locs[mask, :]
    values = values[mask, :]
    adjs = adjs[mask, :][:, mask]
    return locs, values, adjs

def construct_polyhedra_map(m, n, value_dims, samples):
    verts = []
    edges = {}
    faces = []
    geod.get_poly("i", verts, edges, faces)
    freq = m ** 2 + n * m + n ** 2
    grid = geod.make_grid(freq, m, n)
    post_edges = {}
    points = verts
    for face in faces:
        face_edges = face
        new_edges = geod.grid_to_points(
            grid, freq, False, [verts[face[i]] for i in range(3)], face_edges
        )
        points[len(points):len(points)] = new_edges
    points = [p.unit() for p in points]
    
    locs = torch.zeros((len(points), 3))
    for p_ind, point in enumerate(points):
        locs[p_ind, 0] = point[0]
        locs[p_ind, 1] = point[1]
        locs[p_ind, 2] = point[2]
    values = torch.zeros((len(points), value_dims))
    for s in range(min(len(samples), len(values))):
        values[s, :] = samples[s, :]
    print(len(locs))
    return locs, values, None





class SOM(torch.nn.Module):
    def __init__(self, map_nodes_locs, map_node_values, base_lr, base_sigma, lr_steps, sigma_steps, metric):
        super().__init__()
        self.map_node_values = map_node_values
        self.map_nodes_locs = map_nodes_locs
        self.base_lr = base_lr
        self.base_sigma = base_sigma
        self.lr_steps = lr_steps
        self.sigma_steps = sigma_steps
        self.metric = metric
        self.mapped_inds = None
    def to(self, *args, **kwargs):
        super().to(**kwargs)
        if ("device" in kwargs):
            self.map_node_values = self.map_node_values.to(device = kwargs["device"])
            self.map_nodes_locs = self.map_nodes_locs.to(device = kwargs["device"])
        return self
    def forward(self, samples, n = 1, anti=False):
        dist = torch.norm(samples - self.map_node_values, dim=1, p=None)
        knn = dist.topk(n, largest=anti)
        return knn.indices, knn.values
    def batch_forward(self, samples, n=1, anti=False):
        dists = torch.cdist(samples, self.map_node_values).squeeze(dim=1)
        knn = dists.topk(dim = 1, k = n, largest = anti)
        return knn.indices, knn.values
    def reward(self, ind, value, t):
        timestep_sigma = self.base_sigma * math.e ** (-1 * t / self.sigma_steps)
        neighborhood_const = -2 * timestep_sigma ** 2
        timestep_lr = self.base_lr * math.e ** (-1 * t / self.lr_steps)
        if (self.metric == "dist"):
            all_metrics = torch.sqrt(torch.sum(torch.pow(self.map_nodes_locs - self.map_nodes_locs[ind], 2), dim = 1))
        elif (self.metric == "ang"):
            norm_locs = torch.nn.functional.normalize(self.map_nodes_locs, p=2.0, dim=1)
            cos_ang = norm_locs * norm_locs[ind][None, :]
            cos_ang = torch.sum(cos_ang.squeeze(dim=0), dim=1)
            eps = 10e-12
            cos_ang = torch.clamp(cos_ang, min=-1 + eps, max = 1 - eps)
            all_metrics = torch.acos(cos_ang)
        else:
            raise Exception
        scaling_adjustment = timestep_lr * torch.exp(torch.pow(all_metrics, 2) / neighborhood_const)
        all_value_offsets = value - self.map_node_values
        self.map_node_values = torch.add(self.map_node_values, all_value_offsets * scaling_adjustment[:, None])
    def encode(self, x, n=3, modulate_dist = True): # [Dog, Mammal] = (10, 1024), n = 3, = (752) < [0, 1] 
        z = torch.zeros(len(self.map_nodes_locs)).to(device = self.map_node_values.get_device())
        enc_dists = []
        enc_inds = []
        for sample in range(len(x)):
            inds, dists = self(x[sample, :], n=n)
            for sub_sample in range(len(inds)):

                enc_dists.append(dists[sub_sample].reshape(1))
                enc_inds.append(inds[sub_sample])
        enc_dists = torch.concat(enc_dists)
        max_dist = max(1, torch.max(enc_dists).item())
        enc_dists = max_dist - enc_dists
        for i, ind in enumerate(enc_inds):
            if (modulate_dist):
                z[ind] += enc_dists[i]
            else:
                z[ind] = 1
        if (modulate_dist):
            norm_scale = torch.sqrt(torch.sum(torch.pow(z, 2))).item()
            if (norm_scale != 0):
                z = z / norm_scale
        return z
    def inv_encode(self, z, method="som"):
        inv_z = torch.zeros_like(z)
        active_nodes = torch.where(z > 0, True, False)
        z_scales = z[active_nodes]
        z_locs = self.map_nodes_locs[active_nodes, :]
        z_values = self.map_node_values[active_nodes, :]
        if (method == "som"):
            dists = torch.cdist(z_locs[None, ...], self.map_nodes_locs[None, ...]).squeeze(dim=0)
            inv_inds = torch.argmax(dists, dim=1)
        elif (method == "vals"):
            dists = torch.cdist(z_values[None, ...], self.map_node_values[None, ...]).squeeze(dim=0)
            inv_inds = torch.argmax(dists, dim=1)
        inv_z[inv_inds] += z_scales
        return inv_z
            
        
    @staticmethod
    def load_samples(path, key, name_key):
        samples = []
        names = []
        with h5py.File(path, "r") as data_file:
            data = data_file[key]
            name_data = data_file[name_key]
            for sub_key in tqdm(data.keys(), desc="Loading Samples"):
                samples.append(data[sub_key][()])
                names += name_data[sub_key][()].tolist()
        samples = torch.tensor(np.concatenate(samples))
        return samples, names
        
    def save(self, path, t):
        map_locs = self.map_nodes_locs.clone().detach().cpu().numpy()
        map_values = self.map_node_values.clone().detach().cpu().numpy()
        np.savez(
            path, 
            map_locs = map_locs, 
            map_values = map_values, 
            base_lr = self.base_lr, 
            base_sigma = self.base_sigma,
            lr_steps = self.lr_steps, 
            sigma_steps = self.sigma_steps, 
            metric = self.metric,
            t = t
        )
    @staticmethod
    def load(path):
        data = np.load(path)
        map_nodes_locs = torch.tensor(data["map_locs"])
        map_node_values = torch.tensor(data["map_values"])
        som = SOM(map_nodes_locs, map_node_values, data["base_lr"], data["base_sigma"], data["lr_steps"], data["sigma_steps"], data["metric"])
        return som, data["t"]
        

    def plot(self):
        axs = []
        cmap = mpl.colormaps["inferno"]
        locs = self.map_nodes_locs.clone().detach().cpu().numpy()
        vals = self.map_node_values.clone().detach().cpu().numpy()
        fig = plt.figure(figsize=(10, 10))
        pca = PCA(4)
        tvals = pca.fit_transform(vals)
        print(np.cumsum(pca.explained_variance_ratio_))
        for i in range(0, len(pca.explained_variance_ratio_), 1):
            tvals_ind = 255 * (tvals[:, i] - tvals[:, i].min()) / (tvals[:, i].max() - tvals[:, i].min())

            ax = fig.add_subplot(3, 2, i+1, projection='3d')
            ax.grid(False)
            ax.axis('off')
            
            ax.scatter3D(locs[:, 0], locs[:, 1], locs[:, 2], c=tvals_ind, cmap = cmap)
            axs.append(ax)
        dists = np.zeros(math.ceil(len(self.map_nodes_locs) ** 0.5) ** 2)
        angs = np.zeros(math.ceil(len(self.map_nodes_locs) ** 0.5) ** 2)
        for ind in range(len(self.map_nodes_locs)):
            anti_ind, _ = self(self.map_node_values[ind, :], int(0.05 * len(self.map_nodes_locs)), anti=True)
            anti_locs = self.map_nodes_locs[anti_ind, :]
            loc = self.map_nodes_locs[ind, :]
            loc = loc / torch.norm(loc, p=2, dim=-1, keepdim=True)
            angs[ind] = torch.mean(torch.acos(torch.clamp(torch.sum(loc[None, ...] *  anti_locs, dim=-1), min=-1 + 10e-12, max = 1 - 10e-12))).item()
            loc_dist = np.mean(np.sqrt(np.sum((self.map_nodes_locs[ind, :] - anti_locs).clone().detach().cpu().numpy() ** 2, axis=1)))
            dists[ind] = loc_dist
        dists = dists / 2
        print(2 * np.mean(dists), 2 * np.std(dists))
        dists = dists.reshape(int(len(dists) ** 0.5), -1)

        angs = angs / math.pi
        print(180 * np.mean(angs), 180 * np.std(angs))
        angs = angs.reshape(int(len(angs) ** 0.5), -1)

        ax = fig.add_subplot(3, 3, 8)
        ax.grid(False)
        ax.axis('off')
        ax.imshow(dists, cmap = cmap)

        ax = fig.add_subplot(3, 3, 9)
        ax.grid(False)
        ax.axis('off')
        ax.imshow(angs, cmap = cmap)
        
        def on_move(event):
            ifound = -1
            ax = None
            for i in range(len(axs)):
                if (event.inaxes == axs[i]):
                    ifound = i
                    ax = axs[i]
                    break
            if (ifound >= 0):
                for i in range(len(axs)):
                    if (i != ifound):
                        if ax.button_pressed in ax._rotate_btn:
                            axs[i].view_init(elev=ax.elev, azim=ax.azim)
                        elif ax.button_pressed in ax._zoom_btn:
                            axs[i].set_xlim3d(ax.get_xlim3d())
                            axs[i].set_ylim3d(ax.get_ylim3d())
                            axs[i].set_zlim3d(ax.get_zlim3d())
            else:
                return
            fig.canvas.draw_idle()
        c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
        plt.tight_layout()
        plt.show()
    def find_inverse_node(self, inds):
        locs = self.map_nodes_locs[inds, :]
        anti_locs = -1 * locs
        dists = torch.cdist(anti_locs[None, ...], self.map_nodes_locs[None, ...]).squeeze(dim=0)
        knn = torch.topk(dists, k=1, dim=1, largest=False, sorted=True)
        return knn.indices
    def process_cluster_stats(self, matched_inds):
        counts = {}
        for ind in range(len(matched_inds)):
            matched_ind = matched_inds[ind].item()
            if (matched_ind in counts):
                counts[matched_ind] += 1
            else:
                counts[matched_ind] = 1
        counts_list = list(counts.values())
        return len(list(set(counts.keys()))) / self.map_nodes_locs.shape[0], np.max(counts_list), np.mean(counts_list), np.median(counts_list), np.std(counts_list), np.min(counts_list)
    def sample_anti_encoding(self, encoding, sample_mean, sample_std, n=1, cluster_map = None):
        angular_samples = torch.normal(sample_mean, sample_std, size = (encoding.shape[0], n)).to(device = encoding.get_device())
        matched_inds, _ = self.batch_forward(encoding)
        matched_locs = self.map_nodes_locs[matched_inds, :]
        anti_locs = (1.0 - matched_locs).squeeze(dim=1)
        anti_locs = torch.nn.functional.normalize(anti_locs, p=2.0, dim=1)
        if (cluster_map is None):
            norm_locs = torch.nn.functional.normalize(self.map_nodes_locs, p=2.0, dim=1)
        else:
            first_map = False
            if (self.mapped_inds is None):
                first_map = True
                self.mapped_inds = torch.sort(torch.tensor(list(cluster_map.keys()))).values
            norm_locs = torch.nn.functional.normalize(self.map_nodes_locs[self.mapped_inds], p=2.0, dim=1)
        cos_ang = anti_locs.matmul(norm_locs.transpose(0, 1))
        eps = 10e-12
        cos_ang = torch.clamp(cos_ang, min=-1 + eps, max = 1 - eps)
        angles = torch.acos(cos_ang)
        angle_sample_offsets = torch.abs(angles.unsqueeze(2) - angular_samples.unsqueeze(1))
        knn = torch.topk(angle_sample_offsets, k=1, dim=1, largest=False, sorted=True)
        
        anti_inds = knn.indices
        if (cluster_map is not None):
            if (first_map):
                self.mapped_encodings = torch.zeros((len(cluster_map.keys()), self.map_node_values.shape[1]))
                for ind in range(self.mapped_inds.shape[0]):
                    self.mapped_encodings[ind, :] = cluster_map[self.mapped_inds[ind].item()]
            anti_encodings = self.mapped_encodings[anti_inds, :].squeeze(dim=1)
        else:
            anti_encodings = self.map_node_values[anti_inds, :].squeeze(dim=1)
        return anti_encodings
    def form_cluster_map(self, encodings):
        matched_inds, _ = self.batch_forward(encodings)
        cluster_map = {}
        for ind in range(matched_inds.shape[0]):
            matched_ind = matched_inds[ind].item()
            if (matched_ind not in cluster_map):
                cluster_map[matched_ind] = []
            cluster_map[matched_ind].append(encodings[ind, :][None, ...])
        for ind in cluster_map:
            cluster_map[ind] = torch.mean(torch.cat(cluster_map[ind]), dim=0)
        return cluster_map





if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    device = "cpu"
    #if (torch.cuda.is_available()):
        #device = f"cuda:0"
    print(device)
    samples, names = SOM.load_samples("..\\..\\data\\category_text_pairs_xl.hdf5", "train/raw_cat_embeddings_con_t5_large", "train/categories")
    
    samples = samples.to(device=device)
    print(torch.mean(torch.sqrt(torch.sum(torch.pow(samples, 2), dim=1))))
    exit()
    names = list(map(lambda t: t.decode("utf-8"), names))
    #locs, values, adjs = construct_grid_map(10, 3, 1, samples, False)
    load = None
    load =  "..\\..\\logs\\som\\raw_cat_embeddings_con_t5_large\\som_checkpoint_40.npz"
    if (load):
        som, _ = SOM.load(load)
        som = som.to(device = device)
    else:
        locs, values, _ = construct_polyhedra_map(4, 4, samples.shape[1], samples)
        som = SOM(locs, values, 1e-4, math.pi, 1e6, 1e5, "dist").to(device = device)
    som.plot()
    exit()
    z = som.encode(samples[0:3, :])
    print(z)
    inv_z = som.inv_encode(z)
    #som.plot()
    inds, dists = som.batch_forward(samples)
    print(inds.shape, dists.shape)
    print(torch.mean(dists))
    inds, dists = som.batch_forward(samples, n=3)
    print(inds.shape, dists.shape)
    print(torch.mean(dists))
    inds, dists = som.batch_forward(samples, anti=True)
    print(inds.shape, dists.shape)
    print(torch.mean(dists))
    exit()
    t = time.time() # [(0, 1)] * N
    print("Start")
    for i in range(0, 50000000):
        sample = random.randint(0, samples.shape[0]-1)
        inds, dist = som(samples[sample, :])
        som.reward(inds, samples[sample, :], i)
        if (i % 20000 == 0):
            print(i)
            som.compute_cluster_stats()
        if (i % 200000 == 0):
            som.save("checkpoint_" + str(i) + ".npz")
        