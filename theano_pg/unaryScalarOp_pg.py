from theano.scalar.basic import *

class Sgn_pg(UnaryScalarOp):
    nfunc_spec = ('sign', 1, 1)

    def impl(self, x):
        # casting to output type is handled by filter
        return numpy.sign(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval+x*0.5]

    def c_code(self, node, name, inputs, outputs, sub):
        # casting is done by compiler
        # TODO: use copysign
        (x,) = inputs
        (z,) = outputs
        type = node.inputs[0].type
        if type in float_types:
            return '%(z)s = (%(x)s > 0) ? 1. : ((%(x)s < 0) ? -1. : (isnan(%(x)s) ? NAN : 0.));' % locals()
        if type in int_types:
            return "%(z)s = (%(x)s >= 0) ? (%(x)s == 0) ? 0 : 1 : -1;" % locals()
        raise ComplexError('complex has no sgn')

    def c_code_cache_version(self):
        s = super(Sgn_pg, self).c_code_cache_version()
        if s:
            return (4,) + s
        else:  # if parent is unversioned, we are too
            return s

sgn_pg = Sgn_pg(same_out_nocomplex, name='sgn')
