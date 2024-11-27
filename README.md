# The-Influence-of-Pruning-on-Polysemanticity-in-Artificial-Neural-Networks
Features are measurable properties of the data that are relevant to solving a certain task. When multiple unrelated features significantly impact the activation of an individual neuron in a neural network, it is called polysemanticity. This thesis aims to investigate the influence of pruning, meaning the removal of parameters from a model, on the degree of polysemanticity in neural networks. To solve this question, a method to generate toy data with known ground truth features is presented and used to create a controlled environment in which polysemanticity can be measured. Starting with a large model that sufficiently represents the ground truth features, different pruning ratios are applied and the pruned model retrained. The polysemanticity in the retrained model is then measured. Assuming that the number of represented features is preserved through pruning, the degree of polysemanticity could be expected to increase since there are fewer neurons to represent the same amount of features. In addition to that, the Dropout technique will be examined in the context of pruning, as it encourages neurons to adapt to the potential absence of other neurons and promotes redundancy in feature representations. The experiments of this thesis could show that the training process includes an adjustment of polysemanticity, often reducing it until it converges to a certain level. The impact of pruning on polysemanticity was not definitive, though a tendency for polysemanticity to decrease after pruning was observed in some cases.

## Repository Structure

### Root Directory
- **`MeasuringPolysemanticity.py`**
  - Methods for analyzing neuron gradients and measuring polysemanticity.
- **`ToyDataGeneration.py`**
  - Code for generating toy data with ground-truth features.
  - Contains:
    - `FeatureModel`: Represents the ground truth with its features.
    - `ToyDataGenerator`: Methods for generating toy datasets.

### `Experiments/`
- Subdirectories for each experiment.
- Each subdirectory contains:
  - Generated data (using `ToyDataGenerator`).
  - A Jupyter Notebook detailing the experiment code and results.

## Dependencies

This project relies on the following Python libraries:

- [Captum](https://captum.ai/) (`captum-0.7.0`): For interpretability methods.
- [PyTorch](https://pytorch.org/): For neural network implementation and analysis.
- [NumPy](https://numpy.org/): For numerical computations.
- [Matplotlib](https://matplotlib.org/): For data visualization.
