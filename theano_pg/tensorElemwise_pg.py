from theano.tensor.basic import _scal_elemwise

@_scal_elemwise
def sgn_pg(a):
    """pseudo-gradient sign of a"""
