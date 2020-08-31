"""
    Predicted state in here
"""
function predict_output(filter, x, k)
    H = get_H(filter, k)
    y_pred = H*x
end
"""
    Aka propagation. Time  update. Take the filtered state and propagate.
"""
function time_update(model, x::MvNormal, k)
    F = jacobian_x(model, x, k)
    v = get_process_noise(model, x, k) # Zero mean RV

    x_pred = F_k*x + v

    return x_pred
end
function measurement_update(model, y, x_pred, k)
    y_pred = predict_output(model, x_pred, k)
    e = y - y_pred # Innovation

    P_pred = cov(x_pred)
    S = cov(e)

    H = jacobian_x(model, k)
    K = P_pred*H'/S

    x = mean(x_pred) + K*mean(e)
    P = (I - K*H)*P_pred
    # Trick  to make the  covariance matrix symmetric
    return MvNormal(x, 0.5*(P + P')), e
end
