function get_christoffel(x::TensorHead,TIT::TensorIndexType,metric::T,invmetric::T ) where {T <: Tensor}
    """
    returns christoffel symbols as ``Γ^a_{bc}``
    """
    g = metric
    ginv = invmetric
    @indices TIT a b c d
    gg = diff(g(-c,-a),x(b))
    h = ginv(d,c)*(gg(-c,-a,-b) + gg(-c,-b, -a) - gg(-a,-b,-c)) / 2
    return h(a,-b,-c)
end


function get_riemann(x::TensorHead,h::S,TIT::TensorIndexType,metric::T) where {S <: Tensor, T <: Tensor}
    """
    compute Riemann curvature tensor as ``R^α_{βγδ}``
    """
    g = metric
    @indices TIT i j k l m
    dh = diff(h(i,-j,-k),x(l))
    R = dh(i,-l,-j,-k) - dh(i,-k,-j,-l) + h(i,-k,-m)*h(m,-l,-j) - h(i,-l,-m)*h(m,-k,-j)
    return R
end
