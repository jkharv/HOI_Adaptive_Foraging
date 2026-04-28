# This file contains the preprocessing steps that are common to all the other
# analyses. This pretty much takes the raw simulation output and calculates
# anything that didn't need to be done during the simulation when the
# time-series itself was available.

function preprocessing!(df::DataFrame)

    # Any smaller than this and we can hardly call it a foodweb anymore. Some
    # other analyses further restrict this to get at the interaction of size and
    # adaptive foraging, but smaller than this and it's so noisy there's hardly
    # any value in keeping it.  
    filter!(:richness_pre => x -> x >= 5, df)

    # It's possible that a run ends early because of numerical issues. Since 
    # settling on a reliable setup for the foodweb model itself, I've not seen 
    # this actually happen. Still a good idea to exlude them just in case some 
    # rare failure slips in.
    filter!(:retcode => x -> x == "Success", df)

    # Node names get saved in the CSV as strings. We'll turn them back into
    # Symbols, since that's what everything expects.
    transform!(df,
        :target_species =>
        ByRow(str -> eval(Meta.parse(str))) =>
        :target_species
    )

    # Load the realized webs from disk, as well as the density of species before
    # the extinction.
    transform!(df,
        :realized_web =>
        ByRow(load_realized_web)
        => [:realized_web, :density_pre]
    )
 
    # Trim the network of any extinct species before we go looking at trophic
    # metrics, just to be sure.
    transform!(df,
        [:realized_web, :density_pre] =>
        ByRow((web, u) -> begin

            if ismissing(web) | ismissing(u)
                return missing
            else
                return trim_network(web, nonzerospecies(web, u))   
            end
        end)
        => :realized_trim
    )

    # Create a version of the realized web with probabilistic edge weights. 
    transform!(df,
        [:realized_trim] =>
        ByRow(x -> begin
        
            if ismissing(x)        
                return missing
            else
                return rescale_network(x)
            end
        end)
        => :realized_prob
    )

    # The Vector{Tuple{Float64, Symbol}} that we use to keep track  of the
    # extinction cascade gets saved as a String when we output to CSV. We'll
    # need to undo that now.
    transform!(df, :cascade => ByRow(parse_cascade_string) => :cascade)

    # The extinction times in the cascades column are in simulations timesteps. 
    # It'll be more convenient to standardize these to be between 0 and 1.
    # Where 0 is the instant of the primary extinction and 1 is just before the
    # next primary extinction.
    transform!(df,
        :cascade =>
        ByRow(standardize_extinction_times)
        => :cascade 
    )

    transform!(df,
        :cascade =>
        ByRow(cascade_timespan)
        => :cascade_timespan 
    )

    # Count the number of primary extinctions.
    transform!(df,
        :target_species =>
        ByRow(length)
        => :n_targets
    )

    # Count the number of secondary extinctions in each cascade.
    transform!(df,
        :cascade =>
        ByRow(x -> length(x))
        => :secondary_extinctions
    )

    # Add a column for proportion of community gone extinct after a primary
    # extinction. This is to account for differences in community size at the
    # time of primary extinction. Early on, I did some sensitivity analyses to
    # see if this would be problematic. Using raw number of secondary
    # extinctions produced qualitatively the same results.
    transform!(df,
        [:richness_pre, :secondary_extinctions, :n_targets] =>
        ByRow((x, y, z) ->  y / (x - z))
        => :extinction_proportion
    )

    transform!(df,
        [:realized_trim, :target_species, :cascade] =>
        ByRow(trophic_measures) =>
        [
            :target_trophic_level,
            :trophic_mean,
            :trophic_median,
            :trophic_range
        ]
    )

    return nothing
end

function HigherOrderFoodwebs.trim_network(web::Missing, nz::Any)::Missing

    return missing
end

function trophic_measures(web, target::Vector{Symbol}, cascade)

    if ismissing(web)

        return (missing, missing, missing, missing)
    end

    tls = trophic_levels(web)

    targets_tls = [tls[t] for t in target]
    
    extinctions = Iterators.flatten(last.(cascade))

    extinctions_tls = [tls[x] for x in extinctions]

    if isempty(extinctions_tls)
        
        avg1 = missing
        avg2 = missing
        range = missing
    else

        avg1 = mean(extinctions_tls)
        avg2 = median(extinctions_tls)
        range = maximum(extinctions_tls) - minimum(extinctions_tls)
    end

    return (mean(targets_tls), avg1, avg1, range)
end

"""
TODO: Cascade times already get standardized to be between 0-1_000 (or whatever
the extinction interval) ends up being. And then further changed to be between
0-1 here. I should fold these steps together. That'll get rid of the constant
too.
"""
function standardize_extinction_times(cascade::Vector{Tuple{Float64, Vector{Symbol}}}
    )::Vector{Tuple{Float64, Vector{Symbol}}}

    cascade_new = Vector{Tuple{Float64, Vector{Symbol}}}(undef, length(cascade))

    for (i, (t, sp)) in enumerate(cascade)

        t_new = t / EXTINCTION_INTERVAL
        cascade_new[i] = (t_new, sp)
    end

    return cascade_new
end

function parse_cascade_string(str::String)::Vector{Tuple{Float64, Vector{Symbol}}}

    val = eval(Meta.parse(str))

    if isempty(val)

        # val will be parsed as simply Tuple{Float64, Tuple} if it's empty.
        # No bueno.
        return Vector{Tuple{Float64, Vector{Tuple}}}()
    else
        return val
    end
end

function load_realized_web(path::String)

    net = missing
    upre = missing

    # This isn't strictly necessary since the try-catch already handles the file
    # not existing along with any other possible problems. But when path is NA
    # (most trials) JLD2 produces a warning when trying to open the non-valid
    # path, and it's super annoying.
    if !ispath(path)

        return (missing, missing)
    end

    try
        
        net = jldopen(path)["net"]
        upre = jldopen(path)["u_pre"]
    catch

        return (missing, missing)
    end

    # Only the beggining bit of this vector are species densities. The rest are
    # the foraging parameters.
    upre = upre[1:richness(net)]

    return (net, upre)
end

function nonzerospecies(x::Missing, y::Any)

    return missing
end

function nonzerospecies(
    web::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}},
    u::Vector{Float64}
    )::Vector{T} where T

    indxs = findall(map(!iszero, u))
    return species(web)[indxs]
end