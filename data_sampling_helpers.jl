using Random
using LinearAlgebra: I
using Distributions

"""
    sample_crp_cluster!(n_k, alpha, rng)

Given current cluster counts n_k (length K), sample a cluster index k for a new object
using a CRP(alpha). If k == K+1, this means "start a new cluster".
"""
function sample_crp_cluster!(; n_k::Vector{Int}, alpha::Float64, rng=Random.default_rng())
    K = length(n_k)
    if K == 0
        push!(n_k, 1)
        return 1
    end 
    total = sum(n_k) + alpha
    weights = K == 0 ? [1.0] : [n / total for n in n_k]  # existing clusters
    push!(weights, alpha / total)                        # new cluster option
    k = rand(rng, Categorical(weights))
    if k == K + 1
        push!(n_k, 0)  # initialize count for new cluster
    end
    n_k[k] += 1
    return k
end

"""
    sample_category(Ii, O, d, alpha, hyper, rng=Random.default_rng())

Sample one category under the object-aware HDP+,infinity generative model.

Arguments
--------
j     : category label
Ii    : number of objects in the category (adding i to avoid confusion with LinearAlgebra.I)
O     : number of percepts per object (fixed for simplicity)
d     : feature dimension
hyper : ObjectAwareHDPHyperparams

Returns
-------
CategorySample with sigma^2, cluster means mu_k, and per-object percept data.
"""
function sample_category(; j::Int, Ii::Int, O::Int, d::Int,
                         hyper::ObjectAwareHDPHyperparams,
                         rng = Random.default_rng())

    # 1. Category-specific variance
    if hyper.sigma2 != nothing  # If category variance is provided, use it
        sigma2 = hyper.sigma2
    else                        # If not, we sample from IG(a0, b0)
        sigma2 = rand(rng, InverseGamma(hyper.a0, hyper.b0))
    end

    # 2. Storage for cluster means and counts
    mus = Vector{Vector{Float64}}()  # mu_k
    n_k = Int[]                      # counts per cluster

    # 3. Sample objects
    objects = Vector{ObjectData}(undef, Ii)

    for i in 1:Ii
        # 3a. CRP: choose cluster for object i
        k = sample_crp_cluster!(n_k=n_k, alpha=hyper.alpha, rng=rng)

        # If this is a brand-new cluster, sample its mean mu_k
        if k > length(mus)
            sigma_clu = (sigma2 / hyper.k_clu) * I(d)
            mu_k = rand(rng, MvNormal(hyper.m0, sigma_clu))
            push!(mus, mu_k)
        end
        mu_k = mus[k]

        # 3b. Sample object-level mean phi_i given mu_k
        sigma_obj = (sigma2 / hyper.k_obj) * I(d)
        phi_i = rand(rng, MvNormal(mu_k, sigma_obj))

        # 3c. Sample percepts for this object
        sigma_per = (sigma2 / hyper.k_per) * I(d)
        ys_i = Vector{Vector{Float64}}(undef, O)
        for o in 1:O
            ys_i[o] = rand(rng, MvNormal(phi_i, sigma_per))
        end

        objects[i] = ObjectData(j, k, i, phi_i, ys_i)
    end

    return CategorySample(j, sigma2, mus, objects)
end

"""
    train_test_split_sample_percepts(cat, mode; frac_train=0.5, rng)

Split objects in `cat` into train/test and further sample percepts according to mode.
(Assume that the input `cat` always contains 20 objects to be evenly split between train & test.)

- uniform mode => each training object gets 5 percepts
- skewed mode  => training percept counts are [18, 8, 6, 5, 4, 3, 2, 2, 1, 1]
- all test objects receive 5 percepts (total 50 percepts)
"""
function train_test_split_sample_percepts(cat::CategorySample;
                                          frac_train::Float64 = 0.5,
                                          rng = Random.default_rng())

    I = length(cat.objects)
    idx = shuffle(rng, collect(1:I))
    n_train = floor(Int, frac_train * I)
    train_idx = idx[1:n_train]
    test_idx  = idx[n_train+1:end]

    train_objs = [cat.objects[i] for i in train_idx]
    test_objs  = [cat.objects[i] for i in test_idx]

    ## -------------------------------------------------------
    ## 1. RESAMPLE TRAIN PERCEPTS (UNIFORM)
    ## -------------------------------------------------------
    train_percept_counts_uniform = fill(5, n_train)  

    new_train_objs_uniform = Vector{ObjectData}(undef, n_train)
    for (t, (obj, n)) in enumerate(zip(train_objs, train_percept_counts_uniform))
        existing = obj.percepts
        new_percepts = shuffle(rng, existing)[1:n]
        new_train_objs_uniform[t] = ObjectData(
            j         = obj.j,
            z         = obj.z,
            i         = obj.i,
            phi       = obj.phi,
            percepts  = new_percepts
        )
    end

    ## -------------------------------------------------------
    ## 2. RESAMPLE TRAIN PERCEPTS (SKEWED)
    ## -------------------------------------------------------
    train_percept_counts_skewed = [18, 8, 6, 5, 4, 3, 2, 2, 1, 1]  

    new_train_objs_skewed = Vector{ObjectData}(undef, n_train)
    for (t, (obj, n)) in enumerate(zip(train_objs, train_percept_counts_skewed))
        existing = obj.percepts
        new_percepts = shuffle(rng, existing)[1:n]
        new_train_objs_skewed[t] = ObjectData(
            j         = obj.j,
            z         = obj.z,
            i         = obj.i,
            phi       = obj.phi,
            percepts  = new_percepts
        )
    end

    ## -------------------------------------------------------
    ## 3. RESAMPLE TEST PERCEPTS (always uniform)
    ## -------------------------------------------------------
    test_percept_counts = fill(5, length(test_objs))

    new_test_objs = Vector{ObjectData}(undef, n_train)
    for (t, (obj, n)) in enumerate(zip(test_objs, test_percept_counts))
        existing = obj.percepts
        new_percepts = shuffle(rng, existing)[1:n]
        new_test_objs[t] = ObjectData(
            j         = obj.j,
            z         = obj.z,
            i         = obj.i,
            phi       = obj.phi,
            percepts  = new_percepts
        )
    end

    return CategoryTrainTest(new_train_objs_uniform, new_train_objs_skewed, new_test_objs)
end
