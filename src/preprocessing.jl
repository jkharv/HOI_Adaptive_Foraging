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

    # Load the realized webs from disk, as well as the density of species before
    # the extinction.
    transform!(df,
        :realized_web =>
        ByRow(load_realized_web)
        => [:realized_web, :density_pre]
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
        [:richness_pre, :secondary_extinctions] =>
        ((x, y) ->  y ./ x)
        => :extinction_proportion
    )

    return nothing
end

const EXTINCTION_INTERVAL = 1000.0

"""
TODO: Cascade times already get standardized to be between 0-1_000 (or whateve
the extinction interval) ends up being. And then further changed to be between
0-1 here. I should fold these steps together. That'll get rid of the constant
too.
"""
function standardize_extinction_times(cascade::Vector{Tuple{Float64, Symbol}})::Vector{Tuple{Float64, Symbol}}

    cascade_new = Vector{Tuple{Float64, Symbol}}(undef, length(cascade))

    for (i, (t, sp)) in enumerate(cascade)

        t_new = t / EXTINCTION_INTERVAL
        cascade_new[i] = (t_new, sp)
    end

    return cascade_new
end

function parse_cascade_string(str::String)::Vector{Tuple{Float64, Symbol}}

    val = eval(Meta.parse(str))

    if isempty(val)

        # val will be parsed as simply Tuple{Float64, Tuple} if it's empty.
        # No bueno.
        return Vector{Tuple{Float64, Tuple}}()
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