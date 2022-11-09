import torch



class Autoencoder(torch.nn.Module):
    def __init__(self, input_size, expansion_size = 1024, reduction_steps = 1, activation_func = "ReLU", batch_size = 512):
        self.activations = {
            "ReLU" : torch.nn.ReLU,
            "LeakyReLU" : torch.nn.LeakyReLU,
        }
        super().__init__()
        self.example_input_array = torch.rand(batch_size, input_size)
        self.expansion_size = expansion_size
        self.activation_func = activation_func
        self.layers_per_stage = 2
        self.encoder_layers = torch.nn.ModuleList([
            torch.nn.Linear(input_size, expansion_size),
            self.activations[self.activation_func](),
        ])
        self.decoder_layers = torch.nn.ModuleList([
            torch.nn.Linear(expansion_size, input_size)
        ])
        self.depth = reduction_steps
        for d in range(reduction_steps):
            self.encoder_layers.append(torch.nn.Linear(expansion_size // (2 ** d), expansion_size // (2 ** (d + 1))))
            self.encoder_layers.append(self.activations[self.activation_func]())
            self.decoder_layers = torch.nn.ModuleList([self.activations[self.activation_func]()]) + self.decoder_layers
            self.decoder_layers = torch.nn.ModuleList([torch.nn.Linear(expansion_size // (2 ** (d + 1)), expansion_size // (2 ** d))]) + self.decoder_layers

    def increment_depth(self):
        self.encoder_layers.append(torch.nn.Linear(self.expansion_size // (2 ** self.depth), self.expansion_size // (2 ** (self.depth + 1))))
        self.encoder_layers.append(self.activations[self.activation_func]())
        self.decoder_layers = torch.nn.ModuleList([self.activations[self.activation_func]()]) + self.decoder_layers
        self.decoder_layers = torch.nn.ModuleList([torch.nn.Linear(self.expansion_size // (2 ** (self.depth + 1)), self.expansion_size // (2 ** self.depth))]) + self.decoder_layers
        
        self.depth += 1
        return self

    def encode(self, x, use_depth = -1):
        if (use_depth == -1):
            use_depth = self.depth
        z = [x]
        for layer in self.encoder_layers[:use_depth * self.layers_per_stage + self.layers_per_stage]:
            z.append(layer(z[-1]))
        return z[-1], z[:-1]

    def decode(self, z, use_depth = -1):
        if (use_depth == -1):
            use_depth = self.depth
        y = [z]
        for layer in self.decoder_layers[len(self.decoder_layers) - use_depth * self.layers_per_stage - 1:]:
            y.append(layer(y[-1]))
        return y[-1], y[:-1]

    def forward(self, x, use_depth = -1):
        if (use_depth == -1):
            use_depth = self.depth
        z_fin, z_acts = self.encode(x, use_depth)
        y_fin, y_acts = self.decode(z_fin, use_depth)
        return y_fin, z_acts, y_acts