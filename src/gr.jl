function get_christoffel(x::TensorHead,TIT::TensorIndexType,metric::T ) where {T <: Tensor}
    """
    returns christoffel symbols as ``Γ_{ijk}``
    """
    @indices TIT i j k l
    g = metric

    simp(x) = contract_metric(x,TIT.metric)
    gg = diff(g(-i,-j),x(k))
    gg = simp(gg)

    h = (gg(-i,-j,-k) - gg(-j,-k,-i) + gg(-k,-i,-j))/2
    h = simp(h)

    return h(-i,-j,-k)
end


function get_riemann(x::TensorHead,h::S,TIT::TensorIndexType,metric::T) where {S <: Tensor, T <: Tensor}
    """
    compute Riemann curvature tensor as ``R_{ijkl}``
    """
    @indices TIT i j k l m n
    g = metric
    simp(x) = contract_metric(x,TIT.metric)
    dh = diff(h(-i,-j,-k),x(l))
    dh = simp(dh)

    hh = (TIT.metric(m,n)*h(-m,-i,-j))*h(-n,-k,-l)
    hh = simp(hh)

    Riemann = dh(-l,-i,-k,-j) - dh(-l,-j,-k,-i) + hh(-j,-k,-i,-l) - hh(-i,-k,-j,-l)
    return Riemann
end

"""
    Compute the Ricci tensor as ``R_{αβ}``
"""
function get_ricci(x::TensorHead, h::S, TIT::TensorIndexType, metric::T) where {S<:Tensor, T<:Tensor}
    @indices TIT i j k l
    g = metric
    Riemann = get_riemann(x, h, TIT, metric)
    Ricci = Riemann(-i, -j, -k, -l) * TIT.metric(i,k)
    return Ricci
end

"""
Compute the Ricci Scalar as ``R ≡ g^{μν}R_{μν}``
"""
function get_ricci_scalar(x::TensorHead, h::S, TIT::TensorIndexType, metric::T) where {S<:Tensor, T<:Tensor}
    @indices TIT i j
    g = metric
    Ricci = get_ricci(x, h, TIT, metric)
    R = Ricci(-i,-j)*TIT.metric(j,i)
    return R
end

"""
Compute the Einstein tensor as ``G^{μν} ≡ R^{μν} - \frac{1}{2}⋅ g^{μν}R
"""
function get_einstein(x::TensorHead, h::S, TIT::TensorIndexType, metric::T) where {S<:Tensor, T<:Tensor}
    @indices TIT i j
    g = metric
    R = get_ricci_scalar(x, h, TIT, metric)
    Ricci = get_ricci(x, h, TIT, metric)
    return Ricci(-i,-j) - 1/2*g(-i,-j)*R
end


