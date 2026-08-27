# This file contains functions that are common to all of the simulation scripts.

"""
    treatments(; args...) :: NamedTuple

Given keyword arguments with values or collections of values for parameters return
all combinations of those parameter values as a vector of `NamedTuple`s where the names
match those given as a keyword for the argument

== Example

treatments(a = 1, b = [1,2,3], c = 4)
 
3-element Vector{@NamedTuple{a::Int64, b::Int64, c::Int64}}:
 (a = 1, b = 1, c = 4)
 (a = 1, b = 2, c = 4)
 (a = 1, b = 3, c = 4)

"""
function treatments(; args...)

    pnames = Tuple(p[1] for p in args)
    pvals  = [p[2] for p in args]
    
    # Find the combinations
    combos = (collect ∘ reduce)(Iterators.product, pvals)
     
    # Flatten the tuples
    combos = map((collect ∘ Iterators.flatten), combos) 

    # Turn them into named tuples
    combos = map(NamedTuple{pnames}, combos)

    return combos
end

"""
    make_output_file!(stem)

Make an output directory for the simulation under ./sim-output
of the form "stem" * current_date * [a-z].

If it gets to z without finding an available name, just do less
simualtions in a day, cause it'll throw and error.
"""
function make_output_file!(stem)::String

    output_file = "sim-output/$stem-" * string(Dates.today())

    if ispath(output_file * ".jld2")

        output_file *= "-a"
    end
    
    count = 2
    while ispath(output_file * ".jld2")

        if count > 26
            error("try doing less simulations in a day")
        end

        prefix = output_file[begin:end-1]
        output_file = prefix * ('a':'z')[count]
        count += 1
    end

    output_file *= ".jld2"

    jldsave(output_file, compress = ZstdFilter())

    return output_file
end

"""
    index_of_time(sol, t)

Return the index into `sol.t` corresponding
to a time point `t`.
"""
function index_of_time(sol, t)

    return findfirst(x -> x == t, sol.t)
end

"""
    extinction_indices(sol, primary_extinctions)

Returns a `Vector` of the bounding time indices between a primary extinction
and the following primary extinctions. Each entry in the return `Vector`
is formated as:


    (spp::Vector{Symbol}, i1::Int64, i2::Int64)

"""
function extinction_indices(sol, primary_extinctions)

    indices = Vector{Tuple{Vector{Symbol}, Int64, Int64}}()

    for (i, (t, spp)) in enumerate(primary_extinctions)
      
        i1 = index_of_time(sol, t)

        if i + 1 <= length(primary_extinctions)

            t_next, spp_next = primary_extinctions[i + 1]
            i2 = index_of_time(sol, t_next)
        else

            i2 = lastindex(sol.t)
        end

        push!(indices, (spp, i1, i2))
    end

    return indices
end

