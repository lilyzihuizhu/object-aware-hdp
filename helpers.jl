"""
    build_clusters(cat::CategorySample) -> Vector{ClusterSample}

Reconstruct cluster-level data for a sampled category.

Given a `CategorySample`, this function groups all `ObjectData` objects
according to their cluster labels (`obj.z`) and returns a vector of
`ClusterSample` structs. Each `ClusterSample` contains:

  • the category index `j`
  • the cluster index `k`
  • the cluster mean `mu_k`
  • all objects assigned to that cluster

The resulting vector is sorted by the cluster index `k` (ascending).

Parameters
----------
cat :: CategorySample
    Output of `sample_category` containing category-level variance, cluster
    means, and object-level assignments.

Returns
-------
Vector{ClusterSample}
    One entry per cluster present in the category.
"""
function build_clusters(cat::CategorySample)
    # group objects by cluster index z
    groups = Dict{Int, Vector{ObjectData}}()
    for obj in cat.objects
        push!(get!(groups, obj.z, ObjectData[]), obj)
    end

    clusters = ClusterSample[]
    for (k, objs) in sort(collect(groups); by = first)  # ensure ordered by k
        mu_k = cat.mus[k]
        push!(clusters, ClusterSample(j = cat.j, k = k, mu = mu_k, objects = objs))
    end

    return clusters
end
