module StateSpaceFilters

using LinearAlgebra

abstract type AbstractStateSpaceModel end


struct PDMatChol{T<:Real,S<:AbstractMatrix} <: AbstractPDMat{T}
    dim::Int                    # matrix dimension
    chol::Cholesky{T,S}         # Cholesky factorization of mat
end

struct StateSpaceModel
    dynamics
    measurement
end

struct StationaryKalmanFilter
    model
    F
    G
    v
    H
end

struct InformationFilter end

struct SquareRootFilter end

struct FilterResults
    states_predicted
    states_filtered
    innovations
end

function run_filter(filter, y, x0)

    N = length(y)
    x_pred = Array{typeof(x0)}(undef, N)
    x_filtered = Array{typeof(x0)}(undef, N)
    e = Array{typeof(y)}(undef, N)

    x_pred[1] = x0
    for n = 1:N
        model_obs = get_observation_model(filter, n)
        x_filt[n], e[n] = measurement_update(model_obs, y[n], x_pred[n])

        if n != N
            model_dynamics = get_dynamics_model(filter, n)
            x_pred[n+1] = time_update(model_dynamics, x_filt[n])
        end
    end
    return  FilterResults(
        states_predicted,
        states_filtered,
        innovations
    )
end

end # module
