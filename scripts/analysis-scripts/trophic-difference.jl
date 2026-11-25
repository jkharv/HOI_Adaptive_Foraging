include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using CairoMakie
using DataFrames
using CSV
using JLD2
using GLM
using Statistics
using LsqFit

df1 = CSV.read("sim-output/one-extinction-2026-09-15.csv", DataFrame)
jld1 = jldopen("sim-output/one-extinction-2026-09-15.jld2")
df1.dataset_num .= 1

df2 = CSV.read("sim-output/one-extinction-2026-10-30.csv", DataFrame)
jld2 = jldopen("sim-output/one-extinction-2026-10-30.jld2")
df2.dataset_num .= 2

preprocessing!(df1, jld1)
preprocessing!(df2, jld2)

df = vcat(df1, df2)

transform!(df,
    [:trophic_mean, :target_trophic_level] =>
    ((x, y) ->  y - x)
    => :trophic_difference
);

# --------------------------------------------------------- #
# Trophic Difference Analyses:                              #
# Are top-down or bottom-up effects implicated more or less # 
# often at differing levels of adaptive foraging? No.       #
# --------------------------------------------------------- #

filt = copy(df)
filter!(:trophic_difference => !ismissing, filt)
filter!(:cascade_number => x-> x == 1, filt)
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

save("figures/suplementary/trophic_difference_v_g.png", f)

# --------------------------------------------------------- #
# Are top-down and bottom-up effects implicated differently #
# for different cascade sizes? Yes                          #
# --------------------------------------------------------- #

filt = copy(df)
filter!(:trophic_difference => !ismissing, filt)
filter!(:cascade_number => x->x==1, filt)
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

# This isn't just using the standard GLM package cause this looked like a case 
# where the data was clearly heteroscedastic and weighted least squares would
# need to be used. Turns out to have not really made a noticeable difference,
# but we'll keep it since that's the right thing to do!
model(t, p) = p[1] .+ (p[2] .* t)
fitOLS = curve_fit(model, 
    filt[:, :extinction_proportion], 
    filt[:, :trophic_difference], 
    [1.0, 1.0]
)
wt = 1 ./ fitOLS.resid.^2
fitWLS = curve_fit(model, 
    filt[:, :extinction_proportion], 
    filt[:, :trophic_difference], 
    wt, 
    [0.0, 0.0]
)

ablines!(ax, fitWLS.param[1], fitWLS.param[2], color=:red, linewidth = 3)

save("figures/trophic_difference_v_cascade_size.svg", f)