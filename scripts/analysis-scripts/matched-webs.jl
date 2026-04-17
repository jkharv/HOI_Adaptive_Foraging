include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using GLM
using Statistics

df = CSV.read("sim-output/niche-model-2026-03-16/data.csv", DataFrame)

preprocessing!(df)

filter!(:richness_pre => x-> x > 20, df)

q95 = quantile(df[:, :extinction_proportion], 0.99)
special_webs = copy(df)
filter!(:extinction_proportion => x-> x>q95, special_webs)
gdf = groupby(special_webs, [:foodweb_number])
gdf = combine(gdf, :sequence_number => Set => :sequence_number)

function is_special_web(gdf, fwnum, sqnum)

    if fwnum ∉ gdf[:, :foodweb_number]

        return false
    end

    row = first(filter(:foodweb_number => x->x==2, gdf))

    if sqnum ∉ row[:sequence_number]

        return false
    end

    return true 
end

is_special_web(fwnum, sqnum) = is_special_web(gdf, fwnum, sqnum)

filter!([:foodweb_number, :sequence_number] => (x,y) -> is_special_web(x,y), df)
filter!(:secondary_extinctions => x-> x>0, df)

fig = Figure(size = (700, 700))
ax  = Axis(fig[1,1], xlabel = "g")

scatter!(ax, df[:, :g], df[:, :extinction_proportion])














pf(x) = quantile(x, 0.95)
gdf = groupby(df, [:foodweb_number, :g])
gdf = combine(gdf, :extinction_proportion => pf => :extinction_proportion)

fig = Figure(size = (800,800))
ax  = Axis(fig[1,1], xlabel = "g", ylabel = "Proportion Extinct")

for fw_num in unique(gdf[:, :foodweb_number])

    f = filter(:foodweb_number => x-> x == fw_num, gdf)
    sort!(f, :g)

    lines!(ax, f[:, :g], f[:, :extinction_proportion])
end

#
# Same but a line for each fw_num x sequence now.
#

pf(x) = quantile(x, 0.5)
gdf = groupby(df, [:foodweb_number, :sequence_number, :g])
gdf = combine(gdf, :extinction_proportion => mean => :extinction_proportion)

fig = Figure(size = (800,800))
ax  = Axis(fig[1,1], xlabel = "g", ylabel = "Proportion Extinct")

for fw_num in unique(gdf[:, :foodweb_number])
    for seq_num in unique(gdf[:, :sequence_number])

        f = filter(
            [:foodweb_number, :sequence_number] => 
            (x,y) -> (x == fw_num) & (y == seq_num), gdf
        )
        sort!(f, :g)

        lines!(ax, f[:, :g], f[:, :extinction_proportion])
    end
end

#
# Same but pairing of target spp now
#

gdf = groupby(df, [:foodweb_number, :sequence_number, :extinction_target])

fig = Figure(size = (800,800))
ax  = Axis(fig[1,1], xlabel = "Generality", ylabel = "Extinctions ~ g Coef")

begin
    ys = Vector{Float64}()
    xs = Vector{Float64}() 

    for x in gdf

        push!(ys, (coef ∘ lm)(@formula(extinction_proportion ~ g), x)[2])
        push!(xs, mean(x[:, :richness_pre]))
    end

    empty!(ax)
    scatter!(ax, xs, ys)
end

x = DataFrame(y = ys, x = xs)
cs = lm(@formula(y ~ x), x)
cs = coef(cs)

ablines!(ax, cs[1], cs[2], color = :red)

#
#
#
