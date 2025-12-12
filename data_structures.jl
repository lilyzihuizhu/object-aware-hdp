"""
Hyperparameters for the object-aware HDP+,infinity generative model.

m0      : prior mean for feature vectors (d-dimensional)
k_clu   : shrinkage from cluster means mu_k toward m0
k_obj   : shrinkage from object means phi_i toward their cluster mean mu_k
k_per   : shrinkage from percepts y_io toward their object mean phi_i
a0,b0   : Inverse-Gamma hyperparameters for sigma^2 (per category)
sigma2  : prior variance if one wishes to hard code it instead of sampling it from IG(a0, b0)
alpha   : CRP concentration (controls number of clusters)
"""
Base.@kwdef struct ObjectAwareHDPHyperparams
    m0::Vector{Float64}
    k_clu::Float64
    k_obj::Float64
    k_per::Float64
    a0::Union{Float64,Nothing} = nothing
    b0::Union{Float64,Nothing} = nothing
    sigma2::Union{Float64,Nothing} = nothing
    alpha::Float64 = 1.0
end

"Data for one object: its category label, cluster label, and percepts."
Base.@kwdef struct ObjectData
    j::Int                              # category index
    z::Int                              # cluster index
    i::Int                              # object index within a category
    phi::Vector{Float64}                # object mean (latent)
    percepts::Vector{Vector{Float64}}   # y_o \in R^d for o = 1..O_i
end

"All data for one cluster in a category."
Base.@kwdef struct ClusterSample
    j::Int                              # category index
    k::Int                              # cluster index
    mu::Vector{Float64}                 # cluster mean
    objects::Vector{ObjectData}         # all objects assigned to this cluster
end

"All data for a single category."
Base.@kwdef struct CategorySample
    j::Int                              # category index
    sigma2::Float64                     # category variance
    mus::Vector{Vector{Float64}}        # cluster means mu_k
    objects::Vector{ObjectData}         # objects i = 1..I
end

"Full dataset across multiple categories."
Base.@kwdef struct Dataset
    categories::Vector{CategorySample}  # length J
end

"Train/test view for one category, splitting at the object level."
struct CategoryTrainTest
    train_objects_uniform::Vector{ObjectData}
    train_objects_skewed::Vector{ObjectData}
    test_objects::Vector{ObjectData}
end