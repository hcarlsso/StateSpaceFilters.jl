function time_update(filter::SquareRootFilter, x, k)
    F = jacobian_dynamics_x(filter, x, k)
    v = get_process_noise(filter, k) # Zero mean RV
    G_v = jacobian_dynamics_v(filter, x, k)

    Q_sqrt = chol(v)
    P_sqrt = chol(x)

    x_pred = predict_x(filter, x, k)

    L = lq(hcat(F*P_sqrt, G_v*Q_sqrt)).L
    n_x = size(F, 1)
    P_pred_sqrt = L[1:n_x, 1:n_x]

    return MvNormal(x_pred, P_pred_sqrt)
end
function measurement_update(filter::SquareRootFilter, y, x_pred, k)

    H = get_H(filter, k)
    P_pred_sqrt = chol(x_pred)
    R_sqrt = chol(y)

    y_pred = H*mean(x_pred) # Include other input and shit
    e = mean(y) - y_pred # Innovation

    n_x = size(P_pred_sqrt, 1)
    n_y = size(R_sqrt, 1)
    z = zeros(n_x, n_y)
    M = vcat(
        hcat(R_sqrt, H*P_pred_sqrt),
        hcat(z,  P_pred_sqrt)
    )
    L = lq(M).L

    S_sqrt = L[1:n_y, 1:n_y]
    P_sqrt = L[1+n_y:n_y + n_x, 1+n_y : n_y + n_x]
    KS_sqrt = L[1 + n_y:n_x + ny, 1:n_y]

    K = KS_sqrt/S_sqrt # Should be fast since triangular

    x_filtered = mean(x_pred) + K*e

    return MvNormal(x_filtered, P_sqrt), MvNormal(e, S_sqrt)
end
