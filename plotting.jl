using Plots

"""
    plot_category(cat::CategorySample, hyper::ObjectAwareHDPHyperparams;
                  plt = nothing,
                  r_scale::Real = 2.0,
                  show_legend::Bool = true)

Visualize a single `CategorySample` in 2D assuming `d = 2`.

For the given category:
  * Plot the global/category prior mean `m0`.
  * For each cluster, plot its mean `mu_k` and draw a line from `m0` to `mu_k`.
  * For each object:
      - Plot its mean `phi_i`.
      - Draw a circle around `phi_i` with radius proportional to the
        percept-level standard deviation: r = r_scale * sqrt(sigma^2 / k_per).
      - Draw a line from `mu_k` to `phi_i`.
      - Plot each percept `y_io` as a small marker.

Arguments
---------
cat             :: CategorySample
    Category to visualize (must be 2D: length of `mu` and `phi` is 2).
hyper           :: ObjectAwareHDPHyperparams
    Hyperparameters, providing `m0` and `k_per`.
plt             : Plot or `nothing` (default = nothing)
    Existing plot to draw into; if `nothing`, a new plot is created.
r_scale         :: Float64
    Optional scaling factor for the percept circles (default 1.0).
plot_circles     : Bool (default = true)
    Whether to plot the circles that sketch the mean and variance of each object.
show_legend     : Bool (default = true)
    Whether to show legend entries for this category.
cluster_palette :: Union{Symbol,Nothing} = nothing
    Color palette used to generate the cluster colors. If `nothing`
    (default), automatically chosen distinguishable colors are used.
    If a `Symbol` (e.g. `:reds`, `:blues`), clusters are colored using
    shades from that palette.

Returns
-------
plt :: Plots.Plot
    The generated plot object.
"""
function plot_category(cat::CategorySample,
                       hyper::ObjectAwareHDPHyperparams; 
                       plt = nothing,
                       r_scale::Float64 = 2.0,
                       plot_circles::Bool = true, show_legend::Bool = true,
                       cluster_palette::Union{Symbol,Nothing} = nothing)

    # sanity check
    @assert length(hyper.m0) == 2 "plot_category_2d assumes 2D m0"
    @assert all(length(mu) == 2 for mu in cat.mus) "All cluster means must be 2D"

    # base plot
    if plt == nothing 
        plt = plot(
            legend = show_legend,
            aspect_ratio = :equal,
            xlabel = "x1",
            ylabel = "x2",
            title = "Category $(cat.j)"
        )
    end 

    # 1. Plot m0 (prior mean) as a star
    m0x, m0y = hyper.m0[1], hyper.m0[2]
    scatter!(plt, [m0x], [m0y],
             marker = :star5, ms = 10,
             color = :gold,
             label = show_legend ? "m0 (prior mean)" : "")

    # radius for percept circles: std dev at percept level
    r_per = r_scale * sqrt(cat.sigma2 / hyper.k_per)

    # 2. Loop over clusters
    K = length(cat.mus)
    cluster_colors = cluster_palette === nothing ?
        distinguishable_colors(K) :
        palette(cluster_palette, K)   # different shades of same family

    k_min = 1 # make sure that we plot the legend on the nonempty cluster w/ the smallest index 
    for (k, mu_k) in enumerate(cat.mus)
        mux, muy = mu_k[1], mu_k[2]
        color = cluster_colors[k]

        # collect objects belonging to this cluster
        objs_k = [obj for obj in cat.objects if obj.z == k]

        if isempty(objs_k)
            if k == k_min
                k_min += 1
            end
        else
            # connect mu_k to m0
            plot!(plt, [m0x, mux], [m0y, muy],
            lc = :black, alpha = 0.7, label=false)
            # plot mu_k itself
            scatter!(plt, [mux], [muy], c = color,
                    marker = :diamond, ms = 7,
                    label = (k == k_min && show_legend) ? "mu_k (cluster means)" : "")
        end 

        # 3. For each object in cluster k
        for (i, obj) in enumerate(objs_k)
            phix, phiy = obj.phi[1], obj.phi[2]

            # line mu_k -> phi_i
            plot!(plt, [mux, phix], [muy, phiy], ls = :dot, 
                  lc = :gray, alpha = 0.7, label=false)

            # circle around phi_i with radius r_per
            if plot_circles
                theta = range(0, 2pi; length = 100)
                circle_x = phix .+ r_per .* cos.(theta)
                circle_y = phiy .+ r_per .* sin.(theta)
                plot!(plt, circle_x, circle_y,
                    seriestype = :shape,   
                    fillcolor = color,
                    linecolor = nothing, linewidth=0,
                    label = false, alpha = 0.05 
                    )
            end 
            
            # object mean phi_i
            scatter!(plt, [phix], [phiy], c = color,
                        marker = :circle, ms = 4,
                        label = (k==k_min && i == 1 && show_legend) ? "phi_i (object means)" : "")

            # percepts y_io
            ys = obj.percepts
            px = [y[1] for y in ys]
            py = [y[2] for y in ys]
            scatter!(plt, px, py, c = color,
                     marker = :circle, ms = 2,
                     alpha = 0.3, markerstrokewidth=0.5,
                     label = false)
        end
    end

    return plt
end

"""
    plot_categories(cats::Vector{CategorySample},
                    hypers::Vector{ObjectAwareHDPHyperparams};
                    n_std::Real = 2.0)

Overlay multiple categories in a single 2D plot.

Each category is drawn using `plot_category`, but with a distinct base color
assigned automatically. Within a category, all elements (m0, mu_k, phi_i, and
percepts) share that base color.

Arguments
---------
cats        : Vector{CategorySample}
    List of categories to visualize.
hypers      : Vector{ObjectAwareHDPHyperparams}
    Hyperparameters associated with each category (same length as `cats`).
plot_circles     : Bool (default = true)
    Whether to plot the circles that sketch the mean and variance of each object.
title       : String (default = "Multiple Categories")
    Plot title
r_scale     : Real (default = 2.0)
    Number of standard deviations used for object-circle radii.

Returns
-------
plt :: Plots.Plot
    The combined plot.
"""
function plot_categories(cats::Vector{CategorySample},
                         hypers::Vector{ObjectAwareHDPHyperparams};
                         plot_circles::Bool = true,
                         title::String = "Multiple Categories",
                         r_scale::Real = 2.0)

    @assert length(cats) == length(hypers) "cats and hypers must have same length"

    plt = plot(xlabel = "x1", ylabel = "x2",
               aspect_ratio = 1,
               title = title,
               legend = :bottomright,
          )
    
    # One palette per category: shades of red, blue, green, ...
    cat_palettes = [:reds, :blues, :greens, :oranges, :purples]

    for (idx, (cat, hyper)) in enumerate(zip(cats, hypers))
        pal = cat_palettes[mod1(idx, length(cat_palettes))]
        # ---- Add legend entry for category ----
        # pick a representative shade (middle of palette)
        base_color = palette(pal, 5)[3]
        scatter!(plt, [NaN], [NaN],   # invisible point
                 color = base_color,
                 markerstrokewidth = 0,
                 label = "Category $(idx)")
        # ---- Plot the category ----
        plot_category(cat, hyper;
                      r_scale = r_scale,
                      plot_circles = plot_circles,
                      show_legend = idx==1, 
                      plt = plt,
                      cluster_palette = pal)
    end

    return plt
end
