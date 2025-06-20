function modified_niche_model(s::Integer=10, c::AbstractFloat=0.2)

    β = Beta(1.0, (1.0/(2c)) - 1.0)
    U = Uniform(0.0, 1.0)

    edges = zeros(Bool, (s, s))

    niches = (sort ∘ rand)(U, s)
    ranges = niches .* rand(β, s)
    centres = (rand ∘ Uniform).(0.5 * ranges, niches)

    for smallest_species ∈ findall(isequal(minimum(niches)), niches)
        ranges[smallest_species] = 0.0
    end

    lower = centres - 0.5 * ranges
    upper = centres + 0.5 * ranges

    for i ∈ eachindex(IndexCartesian(), edges)

        resource, consumer = Tuple(i)

        if lower[consumer] < niches[resource] < upper[consumer] 
        
            edges[consumer, resource] = true
        end
    end
    
    # I shouldn't just assert this, I should actually ensure this.  I should
    # also ensure no trophically identical species, as per Williams & Martinez
    # 2000
    @assert isconnected(edges)

    z = Tuple.(findall(edges))
    z = [[Symbol("sp$(t[1])"), Symbol("sp$(t[2])")] for t ∈ z] 

    intxs = AnnotatedHyperedge.(z, Ref([:subject, :object]))
   
    spp = Symbol.(["sp$i" for i in 1:s])
    spp_pool = Unipartite(spp)

    net = SpeciesInteractionNetwork(spp_pool, intxs)
    traits = DataFrame(
        species = spp, 
        trait_value = niches, 
        niche_lower = lower, 
        niche_upper = upper,
    )

    return (net, edges, traits)
end

function laplacian(adj::Matrix{Bool})

    diag = sum(adj; dims = 1) + (transpose ∘ sum)(adj; dims = 2)
    diag = Diagonal(vec(diag))

    return adj - diag
end

function isconnected(adj::Matrix{Bool})

    return eigen(laplacian(adj)).values[2] ≠ 0.0
end

# Count the number of basal species in the foodweb.
function count_basal(web::SpeciesInteractionNetwork)

    count(iszero, (values ∘ generality)(web))
end

# Niche model with a minimum guaranteed number of basal species
function niche_model_min_basal(s, c, b)

    for i ∈ 1:100

        web = HOI_Adaptive_Foraging.modified_niche_model(s, c)

        if count_basal(first(web)) >= b

            return web
        end
    end

    throw(ErrorException("Couldn't find a web with $b basal species."))
end

