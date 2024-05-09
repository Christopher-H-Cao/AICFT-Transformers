
from abc import ABC, abstractmethod
import numpy as np
import math
from logging import getLogger

logger = getLogger()

class Generator(ABC):
    def __init__(self, params):
        super().__init__()

    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass

# empty for now
class Sequence(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.operation = params.operation


    def generate(self, rng):
        return None
        degree = rng.randint(self.min_degree, self.max_degree + 1)
        roots = rng.rand(degree) * self.max_root
        roots = roots.astype(complex)
        for i in range(degree//2):
            cplex = (rng.rand() < self.prob_complex)
            if cplex:
                roots[2 * i] = complex(roots[2 * i], roots[2 * i + 1])
                roots[2 * i + 1] = np.conj(roots[2 * i])
        poly = np.real(np.poly(roots))
        roots = np.sort_complex(roots)
        return poly, roots

    def evaluate(self, src, tgt, hyp, prec=0.01):
        return -1., -1., -1., -1.
        hyp = np.sort_complex(hyp)
        tgt = np.sort_complex(tgt)
        m = np.abs(hyp - tgt) 
        s = np.abs(tgt)
        if np.max(s) == 0.0:
            if np.max(m) == 0.0:
                return 0.0, 0.0, 0.0, 1.0
            else:
                return -1.0, -1.0, -1.0, 0.0
        errors = m / (s + 1.0e-12)
        e = np.sum(errors < prec) / m.size
        return np.max(errors), np.mean(errors), math.sqrt(np.sum(errors * errors)), e        

