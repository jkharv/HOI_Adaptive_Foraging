using WGLMakie
using DataFrames
using CSV
using Statistics
using CategoricalArrays
using Distributions

df = CSV.read("sim-output/niche-model-2026-01-12/data.csv", DataFrame)

filter!(:richness_pre => x-> x >= 20, df)
filter!(:retcode => x -> x == "Success", df)
transform!(df,
    [:richness_pre, :secondary_extinctions] =>
    ((x, y) ->  y ./ x)
    => :extinction_proportion
)

filter!(:extinction_proportion => (x-> x>0), df)


# Hist plots
fig = Figure(size = (750, 500))
ax  = Axis(fig[1,1], xlabel = "log(Cascade Size)", ylabel = "count")
for (i, g) in (enumerate ∘ groupby)(df, :g)

    density!(ax, 
        log.(g[:, :extinction_proportion]), 
        color = (:slategray, 0.4),
        offset = i,
        # normalization = :pdf
        
    )
end
save("./figures/kde-v-g.png", fig)


param = []
# Plots of exponential distribution fit to data.
fig = Figure(size = (750, 500))
ax  = Axis(fig[1,1], xlabel = "Cascade Size", ylabel = "count")
xlims!(0, 0.5)
for (i, g) in (enumerate ∘ groupby)(df, :g)
    
    x = fit(Exponential, g[:, :extinction_proportion])
    append!(param, params(x))
    band!(ax, 0:0.001:1.0, 
        pdf.(Ref(x), 0:0.001:1.0) .+ 4i,
        4i;
        color = (:slategray, 0.4)
    )
end

# Exp fit on log scale
fig = Figure(size = (750, 500))
ax  = Axis(fig[1,1], xlabel = "Cascade Size", ylabel = "log(count)")
xlims!(0, 0.5)
for (i, g) in (enumerate ∘ groupby)(df, :g)
    
    x = fit(Exponential, g[:, :extinction_proportion])
    append!(param, params(x))
    lines!(ax, 0:0.001:1.0, 
        (log ∘ pdf).(Ref(x), 0:0.001:1.0), 
    )
end
save("./log-exp-fit-v-g.png", fig)


empty!(ax)
g_vals = unique(df[:, :g])
lines!(ax, g_vals, param)