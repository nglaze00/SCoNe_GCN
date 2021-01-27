# Simplicial Complex Net (SCoNe)
A novel convolutional neural network architecture for trajectory prediction on simplicial complexes (i.e. graphs). Uses higher-order graph structure (triangles) in the graph to train very generalizable trajectory predictionmodels.

## Use
1. Clone this repo
2. Set up a dataset in one of two ways (see [synthetic_data_gen.py](trajectory_analysis/synthetic_data_gen.py) for more info):
    * Generate a synthetic dataset (graph + trajectories) using [synthetic_data_gen.py](trajectory_analysis/synthetic_data_gen.py)
    * Convert your own data to the format SCoNe accepts
3. Train a model (see [trajectory_experiments.py](trajectory_analysis/trajectory_experiments.py) for more info)
    * SCoNe: Run [trajectory_experiments.py](trajectory_analysis/trajectory_experiments.py) (with arguments)
    * Projection: Run [projection_model.py](trajectory_analysis/projection_model.py) (see file for more info)
    * Markov: Run [trajectory_experiments.py](trajectory_analysis/trajectory_experiments.py) with arg -markov 1
    
