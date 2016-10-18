from theano.tensor.inplace import _scal_inplace

@_scal_inplace
def sgn_pg_inplace(a):
    """pseudo-gradient sign of 'a' (inplace on `a`)"""
