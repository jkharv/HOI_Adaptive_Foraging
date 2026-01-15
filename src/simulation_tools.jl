# This file contains functions that are common to all of the simulation scripts.

"""
    make_output_dir!(stem)

Make an output directory for the simulation under ./sim-output
of the form "stem" * current_date * [a-z].

If it gets to z without finding an available name, just do less
simualtions in a day, cause it'll throw and error.
"""
function make_output_dir!(stem)

    output_dir = "sim-output/$stem-" * string(Dates.today())

    if ispath(output_dir)

        output_dir *= "-a"
    end
    
    count = 2
    while ispath(output_dir)

        if count > 26
            error("try doing less simulations in a day")
        end

        prefix = output_dir[begin:end-1]
        output_dir = prefix * ('a':'z')[count]
        count += 1
    end

    mkpath(output_dir)

    return output_dir
end

"""
    save_parameters!(output_dir; kwargs...)

Save the parameters of simulation run to the output directory as
`simulation_parameters.txt`. Parameters are given this function as
a list of arbitrary keyword arguments.
"""
function save_parameters!(output_dir; kwargs...)

    open(output_dir * "/simulation-parameters.txt", "a") do io 
        for args in kwargs

            println(io, args)
        end
    end

    return
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

Returns a `Vector` of the bouding time indices between a primary extinction
and the following primary extinctions. Each entry in the return `Vector`
is formated as:


    (spp::Symbol, i1::Int64, i2::Int64)

"""
function extinction_indices(sol, primary_extinctions)

    indices = Vector{Tuple{Symbol, Int64, Int64}}()

    for (i, (t, sp)) in enumerate(primary_extinctions)
      
        i1 = index_of_time(sol, t)

        if i + 1 <= length(primary_extinctions)

            t_next, sp_next = primary_extinctions[i + 1]
            i2 = index_of_time(sol, t_next)
        else

            i2 = lastindex(sol.t)
        end

        push!(indices, (sp, i1, i2))
    end

    return indices
end

"""
    simulation_batch(fwm::FoodwebModel, prob::ODEProblem, process_solution::Function;
        extinction_order = shuffle(species(fwm)),
        extinction_times = missing,
        g1 = 0.0,
        g2 = 0.5,
        ntrajectories = 5
    )

Performs a batch of simulations. Each simulation in the batch varies only by the
value of the `g` parameter. Simulations within a batch are run in parallel on
the CPU using EnsembleProblem.

process_solution should be a function witht the following signature:

    process_solution(
        sol::ODESolution, 
        g::Float64, 
        primary_extinctions::Vector{Tuple{Float64, Symbol}},
        secondary_extinctions::Vector{Tuple{Float64, Symbol}}
    )

## Arguments

`g1`, `g2` : 
    Starting and end point for the range of g values simulated.

`ntrajectories` : 
    The number of g values between `g1` and `g2` to simulate. Must be greater
    than 2.

`extinction_order` : 
    The order in which species should be made to go extinct (primary extinction
    order). Given as `Vector{Symbol}`, Defaults to `shuffle(species(fwm))`

`extinction_times` : 
    The times at which primary extinctions should occur. Given as
    `Vector{Float64}`.  This is a mandatory argument, and will be used to
    determine the `tspan` needed for the simulation.

"""
function simulation_batch(fwm, prob, process_solution::Function;
    extinction_order = shuffle(species(fwm)),
    extinction_times = missing,
    g1 = 0.0,
    g2 = 0.5,
    ntrajectories = 5
    )

    @assert ntrajectories > 2
    @assert !ismissing(extinction_times)
    @assert g1 < g2
    @assert extinction_order ⊆ species(fwm) 

    n_extinctions = length(extinction_times)
    stepsize = (g2 - g1)/(ntrajectories - 1)
    gs = collect(g1:stepsize:g2)

    primary_extinctions = [Vector{Tuple{Float64, Symbol}}() for i in 1:ntrajectories]
    secondary_extinctions = [Vector{Tuple{Float64, Symbol}}() for i in 1:ntrajectories]

    # This function is responsible for taking and index into the batch, `i`, and
    # returning the `ODEProblem` that is to be run. `repeat` is unused but is
    # required by the `EnsembleProblem` API.
    function prob_func(prob, i, repeat)

        fwm = prob.f.sys

        es = ExtinctionSequenceCallback(fwm, deepcopy(extinction_order), extinction_times;
            extinction_history = primary_extinctions[i]
        );
        et = ExtinctionThresholdCallback(fwm, 1e-15;
            extinction_history = secondary_extinctions[i]
        );
        am = AlphaManifoldCallback(fwm);
        cb = CallbackSet(et, es, am);

        return remake(prob,
            callback = cb,
            p = Dict(:g => gs[i]),
        )
    end

    function out_func(sol, i)

        ret = process_solution(
            sol,
            gs[i],
            primary_extinctions[i],
            secondary_extinctions[i]
        )

        return (ret, false)
    end

    eprob = EnsembleProblem(prob;
        prob_func = prob_func,
        output_func = out_func
    )

    @time sols = solve(eprob,
        Tsit5(),
        EnsembleThreads();
        force_dtmin = true,
        maxiters = 1e7,
        tstops = extinction_times,
        trajectories = ntrajectories,
        tspan = (1, 10_000 * n_extinctions + 1_000)
    );

    return sols
end