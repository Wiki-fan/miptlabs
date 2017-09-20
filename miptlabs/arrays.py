from .pq import *

def get_nparray_from_PQs(pqs):
    vals = PQ.get_from_array(lambda elem: elem.val, pqs)
    sigmas = PQ.get_from_array(lambda elem: elem.sigma, pqs)
    x = (vals/pqs[0].dim).astype(float)
    x_s = (sigmas/pqs[0].dim).astype(float)
    return (x, x_s)


def repr_ndarray_as(arr, dim):
    arr = np.array([val.repr_as(dim) for val in arr])
    return arr
