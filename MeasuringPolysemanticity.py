import torch
import numpy as np
from captum.attr import IntegratedGradients
from scipy.stats import entropy

def get_neuron_gradients(features, encoding_model, specimen_model, layer_types=[torch.nn.Linear]):
    """
    Calculate gradients of neurons with respect to input features for linear models.

    This function computes the gradients of each neuron's activation with respect to each input feature for the given encoding and specimen model. 
    It uses a linear approach suitable for models without non-linearities between layers.

    Args:
        features (int): Number of input features.
        encoding_model (torch.nn.Module): The encoding model.
        specimen_model (torch.nn.Module): The observed model.
        layer_types (list): Types of layers to consider for activation extraction. Default is [torch.nn.Linear].

    Returns:
        numpy.ndarray: A 2D array of shape (total_neurons, features) containing the gradients.

    Note:
        This function assumes a linear relationship between layers and may not be suitable for models with non-linear activations between layers.
    """

    def get_all_activations(model, input_tensor):
        activations = []
        def hook(module, input, output):
            activations.append(output)
        hooks = []
        layers = [layer for layer in model.modules() if isinstance(layer, tuple(layer_types))]
        # Register hooks only for hidden layers (excluding input and output layer)
        for layer in layers[:-1]:
            hooks.append(layer.register_forward_hook(hook))
        model(input_tensor)
        for h in hooks:
            h.remove()
        return torch.cat([a.view(a.size(0), -1) for a in activations], dim=1)

    # Initialize tensor to store gradients
    neuron_gradients = []

    # Calculate gradient for each feature
    for feature in range(features):
        # Create a feature vector with requires_grad=True
        feature_vector = torch.zeros(features)
        feature_vector[feature] = 1.0
        feature_vector = feature_vector.unsqueeze(0).requires_grad_(True)

        # Feed the feature vector through both models
        intermediate_output = encoding_model.encoder(feature_vector.unsqueeze(0))
        specimen_activations = get_all_activations(specimen_model, intermediate_output)

        # Calculate gradients for each neuron
        feature_gradients = []
        for i in range(specimen_activations.size(1)):  # Iterate over each neuron
            grad_output = torch.zeros_like(specimen_activations)
            grad_output[0, i] = 1  # Set to 1 for the current neuron
            grad = torch.autograd.grad(specimen_activations, feature_vector, grad_outputs=grad_output, retain_graph=True)[0]
            feature_gradients.append(grad[0, feature].item()) # Only keep the gradient for the current feature

        neuron_gradients.append(feature_gradients)

    # Convert to numpy array for easier manipulation
    neuron_gradients = np.array(neuron_gradients)  # Shape: (features, total_neurons)

    # Transpose to get (total_neurons, features) shape
    neuron_gradients = neuron_gradients.T
    return neuron_gradients

def get_nlin_neuron_gradients(features, encoding_model, specimen_model, layer_types=[torch.nn.Linear]):
    """
    Calculate gradients of neurons with respect to input features for non-linear models.

    This function computes the gradients of each neuron's activation with respect to each input feature for the given encoding and specimen model. 
    It uses Integrated Gradients, which is suitable for models with non-linearities between layers.

    Args:
        features (int): Number of input features.
        encoding_model (torch.nn.Module): The encoding model.
        specimen_model (torch.nn.Module): The observed model.
        layer_types (list): Types of layers to consider for activation extraction. Default is [torch.nn.Linear].

    Returns:
        numpy.ndarray: A 2D array of shape (total_neurons, features) containing the gradients.

    Note:
        This function uses Integrated Gradients and is suitable for models with non-linear activations between layers.
    """
    def get_all_activations(model, input_tensor):
        activations = []
        def hook(module, input, output):
            activations.append(output)
        hooks = []
        layers = [layer for layer in model.modules() if isinstance(layer, tuple(layer_types))]
        # Register hooks only for hidden layers (excluding input and output layer)
        for layer in layers[:-1]:
            hooks.append(layer.register_forward_hook(hook))
        model(input_tensor)
        for h in hooks:
            h.remove()
        return torch.cat([a.view(a.size(0), -1) for a in activations], dim=1)

    class CombinedModel(torch.nn.Module):
        def __init__(self, encoding_model, specimen_model):
            super().__init__()
            self.encoding_model = encoding_model
            self.specimen_model = specimen_model

        def forward(self, x):
            intermediate_output = encoding_model.encoder(x)
            specimen_activations = get_all_activations(specimen_model, intermediate_output)
            return specimen_activations

    combined_model = CombinedModel(encoding_model, specimen_model)

    baseline = torch.zeros(1, features, requires_grad=True)
    baseline_activations = combined_model(baseline)

    gradients = []
    for i in range(baseline_activations.size(1)):
        grad = []
        for f in range(features):
            input_vector = torch.zeros(1, features, requires_grad=True)
            with torch.no_grad():
                input_vector[0, f] = 1

            ig = IntegratedGradients(combined_model)
            attributions = ig.attribute(
                                input_vector,
                                target=i,
                                baselines=baseline,
                                n_steps=50,
                                method='riemann_right')
            grad.append(attributions[0,f].item())

        gradients.append(torch.tensor(grad).squeeze(0))

    gradients = torch.stack(gradients)

    return gradients.detach().numpy()

def measure_relative_polysemanticity(neuron_gradients):
    """
    Calculate polysemanticity for each neuron based on its gradients.

    This function measures the polysemanticity of each neuron by comparing the sum of absolute gradients to the maximum absolute gradient. 
    A higher value indicates higher polysemanticity. This option was used for the experiments of this thesis.

    Args:
        neuron_gradients (numpy.ndarray): A 2D array of shape (total_neurons, features) containing the gradients of each neuron with respect to input features.

    Returns:
        numpy.ndarray: A 1D array containing the polysemanticity measure for each neuron.
    """
    polysemanticity = []
    for gradient in neuron_gradients:
        total_gradient = np.sum(np.abs(gradient))  # Sum of non-negative gradients
        max_gradient = np.max(np.abs(gradient))  # Largest gradient
        if max_gradient > 0:
            polysemanticity_value = total_gradient / max_gradient
        else:
            polysemanticity_value = 0  # Avoid division by zero
        polysemanticity.append(polysemanticity_value)
    return np.array(polysemanticity)

def measure_entropy_polysemanticity(neuron_gradients):
    """
    Calculate entropy-based polysemanticity for each neuron based on its gradients.

    This function measures the polysemanticity of each neuron using the entropy of its normalized absolute gradients. 
    The resulting values are then normalized to a [0, 1] range.

    Args:
        neuron_gradients (numpy.ndarray): A 2D array of shape (total_neurons, features) containing the gradients of each neuron with respect to input features.

    Returns:
        numpy.ndarray: A 1D array containing the normalized entropy-based polysemanticity measure for each neuron.
    """
    polysemanticity = []

    for gradient in neuron_gradients:
        # Normalize the absolute gradients to create a probability distribution
        abs_gradient = np.abs(gradient)
        total_gradient = np.sum(abs_gradient)

        if total_gradient > 0:
            # Normalize to create a probability distribution
            prob_dist = abs_gradient / total_gradient
            # Calculate the entropy of the probability distribution
            entropy_value = entropy(prob_dist)

            polysemanticity.append(entropy_value)
        else:
            polysemanticity.append(0)  # If all gradients are zero, entropy is zero

    if polysemanticity:
        min_entropy = np.min(polysemanticity)
        max_entropy = np.max(polysemanticity)
        normalized_polysemanticity = (polysemanticity - min_entropy) / (max_entropy - min_entropy)
    else:
        normalized_polysemanticity = polysemanticity  # In case of an empty list

    return np.array(normalized_polysemanticity)