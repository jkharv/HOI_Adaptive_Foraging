# Count the number of basal species in the foodweb.
function count_basal(web::AnnotatedHypergraph)

    count(iszero, (values ∘ generality)(web))
end

# Niche model with a minimum guaranteed number of basal species.
function niche_model_min_basal(s, c, b)

    for i ∈ 1:100

        web = nichemodel(s, c)

        if count_basal(web) >= b

            return web
        end
    end

    throw(ErrorException("Couldn't find a web with $b basal species."))
end