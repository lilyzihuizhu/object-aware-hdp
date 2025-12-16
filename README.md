# An Object-Aware Hierarchical Dirichlet Process as a Rational Model of Infant Category Learning

Final project for 9.660 Computational Cognitive Science at MIT (Fall 2025).

## Repo structure

```
.
├── data/: .csv files and .jld2 files that store training & test datasets and models' test performance
├── figures/: figures included in the writeup, including model structure, dataset distribution, model performance, etc.
├── scripts/:
│   ├── data_sampling_helpers.jl: helper functions for sampling the toy dataset and generating train-test splits
│   ├── data_structures.jl: user-defined data structures for implementing the models
│   ├── object_aware_HDP.jl: code for implementing and doing collapsed Gibbs inference on the object-aware HDP model
│   ├── plotting.jl: helper functions for plotting (mainly used in `generate_data.ipynb` for visualizing dataset distributions)
│   └── traditional_HDP.jl: code for implementing and doing collapsed Gibbs inference on the object-aware HDP model
├── generate_data.ipynb: notebook that generates the complete toy dataset and train-test splits
├── data_analysis.Rmd: statistical analyses of model performance
└── model_train_test.ipynb: notebook for model training and testing
```

## Writeup

See [Overleaf](https://www.overleaf.com/read/hmjpttyrjwzz#2afbc6).

## Setting up computing environment

The models are implemented in Julia v1.12.1. See the `*.toml` files for more info on package versions, etc. 

To replicate results in the writeup, run the following notebooks in order:
1. `generate_data.ipynb`: generates the dataset and train-test splits (outputs: `data/*.jld2` and also other `*.csv` files for visualizing dataset distributions)
2. `model_train_test.ipynb`: model training (collapsed Gibbs) and testing (output: `data/results_all.csv`)
3. `data_analysis.Rmd`: analyzes the test outcomes



