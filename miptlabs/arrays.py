from .pq import *


"""def get_nparray_from_PQs(pqs):
    vals = PQ.get_from_array(lambda elem: elem.val, pqs)
    sigmas = PQ.get_from_array(lambda elem: elem.sigma, pqs)
    x = (vals/pqs[0].dim).astype(float)
    x_s = (sigmas/pqs[0].dim).astype(float)
    return (x, x_s)"""


def repr_ndarray_as(arr, dim):
    arr = pqarray([val.repr_as(dim) for val in arr])
    return arr


class pqarray(np.ndarray):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return

    @staticmethod
    def get_from_array(lambd, arr):
        return pqarray([lambd(elem) for elem in arr])

    @property
    def val(self):
        return self.get_from_array(lambda elem:elem.val, self)

    @property
    def sigma(self):
        return self.get_from_array(lambda elem: elem.sigma, self)

    @property
    def epsilon(self):
        return self.get_from_array(lambda elem: elem.epsilon, self)

    @property
    def dim(self):
        return self.get_from_array(lambda elem: elem.dim, self)

    @property
    def val_float(self):
        return (self.val/self[0].dim).astype(float)

    @property
    def sigma_float(self):
        return (self.sigma/self[0].dim).astype(float)
