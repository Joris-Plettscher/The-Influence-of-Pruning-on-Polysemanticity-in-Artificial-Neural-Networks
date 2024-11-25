import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FeatureModel(nn.Module):
    def __init__(self, num_features, output_dim, input_dim=0, min_neurons=5, max_neurons=100, linear=True, layer_dims=None):
        super(FeatureModel, self).__init__()
        self.num_features = num_features
        self.input_dim = input_dim
        self.output_dim = output_dim
        if(input_dim==0):
            self.input_dim = np.random.randint(min_neurons, max_neurons + 1)

        # Generate random layer dimensions such that the total number of neurons (excluding the input and output layer) is num_features
        if layer_dims is None:
            self.layer_dims = self._generate_random_layers(min_neurons, max_neurons)
        else:
            self.layer_dims = layer_dims
            self.num_features = sum(self.layer_dims) - self.layer_dims[0] - self.layer_dims[-1]

        # Create the Sequential model
        layers = []
        for i in range(len(self.layer_dims) - 1):
            layers.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if not linear and i > 0 and i < len(self.layer_dims) - 2:  # Linear activation in the first and last layer
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def _generate_random_layers(self, min_neurons, max_neurons):
        layer_dims = [self.input_dim]
        remaining_features = self.num_features
        lower_bound = (min_neurons+max_neurons) // 2
        upper_bound = max_neurons
        while remaining_features > 0:
            try:
                neurons = np.random.randint(lower_bound, min(upper_bound, remaining_features) + 1)
            except:
                for i in range(1, len(layer_dims)):
                    layer_dim = layer_dims[i]
                    if layer_dim < max_neurons:
                        added_neurons = min(max_neurons - layer_dim, remaining_features)
                        layer_dims[i] = layer_dim + added_neurons
                        remaining_features -= added_neurons
                        if remaining_features <= 0:
                            break
                if remaining_features > 0:
                    layer_dims.append(remaining_features)
                break
            layer_dims.append(neurons)
            remaining_features -= neurons
            if(remaining_features<=upper_bound):
                upper_bound = lower_bound
                lower_bound = (min_neurons + upper_bound) // 2

        layer_dims.append(self.output_dim)  # Add output layer dimension
        return layer_dims

    def forward(self, x):
        activations = []
        activations.append(x) # View input as activation
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations.append(x)
        return activations

    def get_output(self, activations):
        return activations[-1]

    def get_dims(self):
        return self.layer_dims

    def get_feature_activations(self, activations):
        selected_activations = []
        for layer_activations in activations[1:-1]:
            if layer_activations.dim() == 1:
                selected_activations.append(layer_activations)
            else:
                selected_activations.append(layer_activations[0])
        return torch.cat(selected_activations)

    def get_seed(self, target_features, steps=1000, lr=0.01):
        input_dim = self.layer_dims[0]
        input_vector = torch.randn(input_dim, requires_grad=True)  # Start with a random input vector

        optimizer = optim.Adam([input_vector], lr=lr)
        criterion = nn.MSELoss()
        approximate = False

        for _ in range(steps):  # Number of optimization steps
            optimizer.zero_grad()
            activations = self.forward(input_vector.unsqueeze(0))
            selected_activations = self.get_feature_activations(activations)
            loss = criterion(selected_activations, torch.tensor(target_features, dtype=torch.float32))
            loss.backward()
            optimizer.step()

        # Check if the result is approximate
        final_activations = self.get_feature_activations(self.forward(input_vector.unsqueeze(0)))
        if not torch.allclose(final_activations, torch.tensor(target_features, dtype=torch.float32), atol=1e-2):
            approximate = True

        return input_vector.detach().numpy(), approximate


class ToyDataGenerator:
    def __init__(self, features, sparsity, input_dim, output_dim, importance=None, importance_bound=0.1, min_layers=2, min_neurons=5, max_neurons=256, linear=True, layer_dims=None):
        self.features = features
        self.sparsity = sparsity
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.importance_bound = importance_bound
        self.linear = linear

        if importance is None:
            self.importance = lambda: 2 * torch.rand(1).item() - 1
        else:
            self.importance = importance

        self.model = FeatureModel(features, output_dim, input_dim=features, min_neurons=min_neurons, max_neurons=max_neurons, linear=linear, layer_dims=layer_dims)
        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self):
        with torch.no_grad():
            for layer in self.model.model:
                if isinstance(layer, nn.Linear):
                    for i in range(layer.out_features):
                        zero_indices = []
                        non_zero_count = 0
                        for j in range(layer.in_features):
                            rand_val = self.importance()
                            if abs(rand_val) < self.importance_bound:
                                layer.weight[i, j] = 0.0
                                zero_indices.append(j)
                            else:
                                layer.weight[i, j] = rand_val
                                non_zero_count += 1
                        #Ensure at least 2 non-zero weights for each neuron
                        if non_zero_count < 2 and len(zero_indices) > 2 - non_zero_count:
                            additional_indices = np.random.choice(zero_indices, 2 - non_zero_count, replace=False)
                            for j in additional_indices:
                                layer.weight[i, j] = self.importance()

                    expected_input = torch.full((layer.in_features,), 0.5, dtype=layer.weight.dtype)
                    expected_output = torch.matmul(layer.weight, expected_input)
                    layer.bias = nn.Parameter(0.5-expected_output)

    def get_feature_model(self):
        return self.model

    def get_compressor_model(self):
        return self.autoencoder

    def generate_output(self, x):
        with torch.no_grad():
            return self.model.get_output((self.model(x)))

    def gen_toy_data(self, num_samples):
        features = np.zeros((num_samples, self.features))
        for i in range(self.features):
            features[:, i] = np.where(np.random.rand(num_samples) < self.sparsity[i], 0, np.random.rand(num_samples))

        f_tensor = torch.tensor(features, dtype=torch.float32)

        seed_tensor = []
        for feature_list in f_tensor:
            seed, _ = self.model.get_seed(feature_list.numpy())
            seed_tensor.append(seed)
        seed_tensor = torch.tensor(seed_tensor, dtype=torch.float32)
        Y = self.generate_output(seed_tensor)

        autoencoder = self.gen_autoencoder(f_tensor)

        with torch.no_grad():
            X, _ = autoencoder(f_tensor)

        return X.numpy(), Y.numpy()

    def gen_autoencoder(self, sample_features, num_epochs = 100, batch_size = 32):
        class Autoencoder(nn.Module):
            def __init__(self, feature_dim, encoded_dim, linear=True):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    *([nn.ReLU()] if not linear else []),
                    nn.Linear(feature_dim // 2, encoded_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoded_dim, feature_dim // 2),
                    *([nn.ReLU()] if not linear else []),
                    nn.Linear(feature_dim // 2, feature_dim)
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded

        autoencoder = Autoencoder(self.features, self.input_dim, self.linear)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            for i in range(0, len(sample_features), batch_size):
                batch = sample_features[i:i+batch_size]
                optimizer.zero_grad()
                _, decoded = autoencoder(batch)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
        self.autoencoder = autoencoder
        return autoencoder