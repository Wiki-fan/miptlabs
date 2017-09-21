from .pq import *


class pqarray(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    @staticmethod
    def get_from_array(lambd, arr):
        return pqarray([lambd(elem) for elem in arr])

    @property
    def val(self):
        return self.get_from_array(lambda elem: elem.val, self)

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

    def repr_as(self, dim):
        return pqarray([val.repr_as(dim) for val in self])
