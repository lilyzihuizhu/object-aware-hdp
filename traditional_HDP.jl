using Distributions
using LinearAlgebra
using Random
using StatsFuns: logsumexp, logaddexp

include("data_structures.jl");

"""
Update stats to add or subtract an observation
"""
function add_point!(cs::TraditionalHDPClusterStats, x::Vector{Float64}; sign::Int = +1)
    @assert sign == 1 || sign == -1 "sign must be +1 or -1"
    cs.n += sign
    @. cs.sum_x += sign * x
    cs.sumsq += sign * sum(abs2, x)
    return cs
end


"""
Compute scatter S = sum ||x||^2 - n ||xbar||^2
"""
function scatter(cs::TraditionalHDPClusterStats)
    n = cs.n
    n == 0 && return 0.0
    xbar = cs.sum_x / n
    return cs.sumsq - n * sum(abs2, xbar)
end

"""
    posterior_hyperparams(cs, hyper)

Given cluster stats and prior hyperparameters, return
(kn, mn, an, bn) for that cluster.
"""
function posterior_hyperparams(cs::TraditionalHDPClusterStats, hyper::TraditionalHDPHyperparams)
    n = cs.n
    d = length(hyper.m0)
    @assert d > 0 "Dimension needs to be at least 1"
    k_clu, k_obj = hyper.k_clu, hyper.k_obj
    m0, a0, b0 = hyper.m0, hyper.a0, hyper.b0

    xbar = cs.sum_x / n
    S = scatter(cs)

    kn = k_clu + k_obj * n
    mn = (k_clu * m0 + k_obj * n * xbar) / kn
    an = a0 + 0.5 * n * d
    bn = b0 + 0.5 * (k_obj * S + k_clu * k_obj * n / kn * sum(abs2, xbar - m0))

    return kn, mn, an, bn
end

"""
    log_predictive_existing(x, cs, hyper)

Compute log p(x | cluster with stats `cs`) using the multivariate t predictive
(eq. 13).
"""
function log_predictive_existing(x::Vector{Float64},
                                 cs::TraditionalHDPClusterStats,
                                 hyper::TraditionalHDPHyperparams)
    @assert cs.n > 0 "Need to have at least one observation in the cluster"
    d = length(x)
    kn, mn, an, bn = posterior_hyperparams(cs, hyper)
    df = 2 * an
    Sigma = (bn / an) * (1 / hyper.k_obj + 1 / kn) * Matrix(1.0I, d, d)
    dist = MvTDist(df, mn, Sigma)
    return logpdf(dist, x)
end

"""
    log_predictive_new(x, hyper)

Compute log p(x | new cluster) using prior-only predictive (eq. 14).
"""
function log_predictive_new(x::Vector{Float64}, hyper::TraditionalHDPHyperparams)
    d = length(x)
    m0, a0, b0 = hyper.m0, hyper.a0, hyper.b0
    df = 2 * a0

    Sigma = (b0 / a0) * (1 / hyper.k_obj + 1 / hyper.k_clu) * Matrix(1.0I, d, d)
    dist = MvTDist(df, m0, Sigma)
    return logpdf(dist, x)
end

"""
    gibbs_sweep!(xs, z, clusters, hyper; rng=Random.default_rng())

One collapsed-Gibbs sweep over all points for a single category.
- xs::Vector{Vector{Float64}}                  : data points
- z::Vector{Int}                               : cluster labels (1..K)
- clusters::Vector{TraditionalHDPClusterStats} : stats for each active cluster
- hyper::TraditionalHDPHyperparams             : traditional HDP hyperparameters
"""
function gibbs_sweep!(xs::Vector{Vector{Float64}},
                      z::Vector{Int},
                      clusters::Vector{TraditionalHDPClusterStats},
                      hyper::TraditionalHDPHyperparams;
                      rng = Random.default_rng())
    N = length(xs)
    d = length(xs[1])
    Kmax = length(clusters) + N       # safe upper bound (rarely needed)
    log_total = log(N - 1 + hyper.alpha)  # constant over i

    # ---- Work buffers (allocated once) ----
    logps = Vector{Float64}(undef, Kmax + 1)
    idxs  = Vector{Int}(undef, Kmax + 1)

    # -------------------------------
    # LOG-SPACE SAMPLER (NO ALLOCS)
    # -------------------------------
    @inline function sample_from_logweights(logw, n, rng)
        # shift by max for stability
        max_logw = maximum(logw[1:n])
        @inbounds logw[1:n] .-= max_logw
        logZ = logsumexp(@view logw[1:n])   # log normalizing constant
        threshold = log(rand(rng)) + logZ   # log(u) + log(Z)
        acc = -Inf                          # log(0)
        @inbounds for k in 1:n
            w = logw[k]
            acc = logaddexp(acc, w)
            if acc >= threshold
                return k
            end
        end
        return n
    end
    
    # -------------------------------
    # MAIN GIBBS LOOP
    # -------------------------------
    for i in 1:N
        x = xs[i]

        # 1. Remove x from its current cluster
        k_old = z[i]
        add_point!(clusters[k_old], x; sign = -1)
        # If empty, just leave it with n == 0 (mark as inactive)

        # 2. Compute unnormalized log-posteriors for *active* clusters + new
        K = length(clusters)
        @inbounds fill!(logps, -Inf) # fill all logps with -Inf (log 0), assumes 0 prob
        n_active = 0
        # Existing clusters
        @inbounds for k in 1:K
            nk = clusters[k].n
            if nk == 0
                continue  # skip empty cluster
            end
            n_active += 1
            idxs[n_active] = k

            log_crp_prior = log(nk) - log_total
            log_like  = log_predictive_existing(x, clusters[k], hyper)
            logps[n_active] = log_crp_prior + log_like
        end
        # New cluster option = index n_active + 1
        new_idx = n_active + 1
        log_crp_prior_new = log(hyper.alpha) - log_total
        log_like_new  = log_predictive_new(x, hyper)
        logps[new_idx] = log_crp_prior_new + log_like_new

        # 3. Normalize & sample new assignment over n_active + 1 entries
        k_choice = sample_from_logweights(logps, new_idx, rng)

        if k_choice == new_idx
            # Create or reuse an empty cluster
            empty_idx = findfirst(c -> c.n == 0, clusters)
            if empty_idx === nothing
                cs_new = TraditionalHDPClusterStats(d) 
                add_point!(cs_new, x; sign = +1)
                push!(clusters, cs_new)
                z[i] = length(clusters)
            else 
                add_point!(clusters[empty_idx], x; sign = +1)
                z[i] = empty_idx
            end
        else
            # Assign to existing active cluster
            k_new = idxs[k_choice]
            add_point!(clusters[k_new], x; sign = +1)
            z[i] = k_new
        end
    end 
end 

"""
    trad_hdp_cluster_update(xs, hyper; iters=1000)

Run collapsed Gibbs for one category.
Returns (z, clusters).
"""
function trad_hdp_cluster_update(xs::Vector{Vector{Float64}},
                                 hyper::TraditionalHDPHyperparams;
                                 iters::Int = 1000,
                                 rng = Random.default_rng())

    N = length(xs)
    d = length(xs[1])

    # initialize: all points in one cluster
    z = fill(1, N) # collect(1:N)
    clusters = [TraditionalHDPClusterStats(d) for _ in 1:maximum(z)]
    for i in 1:N
        add_point!(clusters[z[i]], xs[i])
    end

    # Prune empty clusters (none yet)
    for it in 1:iters
        gibbs_sweep!(xs, z, clusters, hyper; rng=rng)
    end

    return z, clusters
end

# total number of training observations
function total_points(clusters::Vector{TraditionalHDPClusterStats})
    s = 0
    for cs in clusters
        s += cs.n
    end
    return s
end

"""
    log_post_pred_x(x_new, clusters, hyper)

Implements Eq (9): log P(x_new | x) for the traditional HDP+,infinity
given:
- x_new   : new observation (Vector{Float64})
- clusters : Vector{TraditionalHDPClusterStats} for *training* data
- hyper    : TraditionalHDPHyperparams
"""
function log_post_pred_x(x_new::Vector{Float64},
                         clusters::Vector{TraditionalHDPClusterStats},
                         hyper::TraditionalHDPHyperparams)

    K = length(clusters)
    I = total_points(clusters)   # total training points
    alpha = hyper.alpha

    # Sanity: nothing to predict from if I == 0
    if I == 0
        # pure prior predictive
        return log_predictive_new(x_new, hyper)
    end

    log_terms = Vector{Float64}(undef, K + 1)
    @inbounds fill!(log_terms, -Inf) # fill all log_terms with -Inf (log 0), assumes 0 prob

    # existing clusters
    for k in 1:K
        nk = clusters[k].n
        if nk == 0
            continue # Skip empty clusters
        end
        prior_weight = nk / (I + alpha)                      
        log_lik = log_predictive_existing(x_new, clusters[k], hyper) # P(x_new | cluster k)
        log_terms[k] = log(prior_weight) + log_lik
    end

    # new cluster term
    prior_new = alpha / (I + alpha)
    log_lik_new = log_predictive_new(x_new, hyper)       # P(x_new | new cluster)
    log_terms[K+1] = log(prior_new) + log_lik_new

    # log-sum-exp to get log P(x_new | x)
    return logsumexp(log_terms)
end
