include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using SpeciesInteractionNetworks
using DataFrames
using CSV
using Statistics
using CategoricalArrays
using Printf
using GLM

df1 = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)
df2 = CSV.read("sim-output/two-extinctions-2026-04-30/data.csv", DataFrame)
df3 = CSV.read("sim-output/three-extinctions-2026-04-30/data.csv", DataFrame)
df4 = CSV.read("sim-output/four-extinctions-2026-04-30/data.csv", DataFrame)

df = vcat(df1, df2, df3, df4)

filter!(:richness_pre => x -> x > 20, df)

preprocessing!(df)

# Move this to preprocessing?
transform!(df,
    [:target_species] =>
    ByRow(length) =>
    [:disturbance_size]
)

filt = filter(:disturbance_size => x -> x > 0, df)
file = "./figures/between-web-variation.svg"

gdf = groupby(filt, [:foodweb_number, :sequence_number, :disturbance_size])

intercepts = []
slopes = []
pvals = []

models = DataFrame(
    intercepts = Vector{Float64}(),
    slopes = Vector{Float64}(),
    pvals = Vector{Float64}(),
) 

for g in gdf

    x = lm(@formula(extinction_proportion ~ g), g)
   
    push!(models, (
        intercepts = coef(x)[1], 
        slopes = coef(x)[2], 
        pvals = ftest(x.model).pval)
    )
end

fig = Figure(size = (1400, 800))

ax = Axis(fig[1,1];
    xlabel = "Cascade Size ~ Adaptation Rate", 
    yticksvisible = false,
    yticklabelsvisible = false,
)
hist!(ax, models.slopes, bins = 30)

# Do another histogram of only the slopes which are significant. Use a BH
# corrected threshold of alpha < 0.05.
sort!(models, [:pvals])
kcrit = 0

for (k, pval) in enumerate(models.pvals)

    if pval > (k / length(models.pvals)) * 0.05

        kcrit = k - 1
        break
    end
end

indxs = collect(1:kcrit)

hist!(ax, models.slopes[indxs], bins = 30, color = :orange)

vlines!(ax, [0], color = :red)

# Put some text on the figure saying what proportion of webs have positive or
# negative responses to adaptive foraging.

n_positive = count(x->x>0, models.slopes[indxs])
n_negative = count(x->x<0, models.slopes[indxs])

p_positive = Printf.format(Printf.Format("%.2f"), 
    (n_positive / length(models.slopes[indxs]) * 100)
)
p_negative = Printf.format(Printf.Format("%.2f"), 
    (n_negative / length(models.slopes[indxs]) * 100)
)

text!(ax, 0.8, 0.8; space = :relative, fontsize = 20,
    text = p_positive * " % positive\n" * p_negative * " % negative"
)

save(file, fig)