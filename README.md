# Simplicial Complex Net (SCoNe)
A convolutional neural network architecture for trajectory prediction on simplicial complexes (i.e. graphs). In addition to node- and edge-level data, uses higher-order structures (triangles) in the graph to train very generalizable trajectory prediction models. Paper can be found on [arXiv](https://arxiv.org/abs/2102.10058).

## Use
1. Clone this repo 
    * Dependencies: Python 3.7; numpy, matplotlib, scipy, networkx, [jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), jaxlib, treelib)
2. Set up a dataset in one of two ways (see [synthetic_data_gen.py](trajectory_analysis/synthetic_data_gen.py) for more info):
    * Generate a synthetic dataset (graph + trajectories) using [synthetic_data_gen.py](trajectory_analysis/synthetic_data_gen.py)
    * Convert your own data to the format SCoNe accepts
3. Train a model (see [trajectory_experiments.py](trajectory_analysis/trajectory_experiments.py) for more info)
    * SCoNe: Run [trajectory_experiments.py](trajectory_analysis/trajectory_experiments.py) (with arguments)
    * Projection: Run [projection_model.py](trajectory_analysis/projection_model.py) (see file for more info)
    * Markov: Run [trajectory_experiments.py](trajectory_analysis/trajectory_experiments.py) with arg -markov 1
    * SNN ([Ebli 2010](https://arxiv.org/pdf/2010.03633.pdf)): Run [trajectory_experiments.py](trajectory_analysis/trajectory_experiments.py) with arg -model 'ebli'
    * SCCONV ([Bunch 2012](https://arxiv.org/pdf/2012.06010.pdf)): Run [trajectory_experiments.py](trajectory_analysis/trajectory_experiments.py) with arg -model 'bunch'
    
