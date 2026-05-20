include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using GLM
using Statistics
using CategoricalArrays
using QuantileRegressions
using LsqFit

# One extinction
df1 = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)
# Two extinctions
df2 = CSV.read("sim-output/two-extinctions-2026-04-30/data.csv", DataFrame)
# Three extinctions
df3 = CSV.read("sim-output/three-extinctions-2026-04-30/data.csv", DataFrame)
# Four extinctions
df4 = CSV.read("sim-output/four-extinctions-2026-04-30/data.csv", DataFrame)

df = vcat(df1, df2, df3, df4)

filter!(:richness_pre => x-> x >= 20, df)

preprocessing!(df)

transform!(df,
    [:trophic_mean, :target_trophic_level] =>
    ((x, y) ->  y - x)
    => :trophic_difference
);

# -------------------------------------------------------- #
# Trophic Difference Analyses:                             #
# Are top-down or bottom-up effects implicated more often? #
# Positive is top-down, negative is bottom-up              #
# -------------------------------------------------------- #

filt = copy(df)
filter!(:trophic_difference => !ismissing, filt)
disallowmissing!(filt, :trophic_difference)

f  = Figure(size = (800, 800))
ax = Axis(f[1,1]; 
    xlabel = "Strength of Adaptive Foraging", 
    ylabel = "Trophic Difference",
    xticks = unique(filt[:, :g]),
    xtickformat = "{:.2f}"
) 

boxplot!(ax, 
    filt[:, :g], 
    filt[:, :trophic_difference];
    width = 0.05,
)

save("figures/trophic_difference_v_g.png", f)

# --------------------------------------------------------- #
# Are top-down and bottom-up effects implicated differently #
# for different cascade sizes? Yes                          #
# --------------------------------------------------------- #

filt = copy(df)
filter!(:trophic_difference => !ismissing, filt)
disallowmissing!(filt, :trophic_difference)

f  = Figure(size = (800, 800))
ax = Axis(f[1,1]; 
    xlabel = "Cascade Size", 
    ylabel = "Trophic Difference",
    xtickformat = "{:.2f}"
) 

scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :trophic_difference];    
)

model(t, p) = p[1] .+ (p[2] .* t)
fitOLS = curve_fit(model, filt[:, :extinction_proportion], filt[:, :trophic_difference], [1.0, 1.0])
wt = 1 ./ fitOLS.resid.^2
fitWLS = curve_fit(model, filt[:, :extinction_proportion], filt[:, :trophic_difference], wt, [0.0, 0.0])

ablines!(ax, fitWLS.param[1], fitWLS.param[2], color=:red, linewidth = 3)

save("figures/trophic_difference_v_cascade_size.png", f)