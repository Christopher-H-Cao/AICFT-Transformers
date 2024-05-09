from abc import ABC, abstractmethod
import numpy as np
from sympy import factorint,prime

class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self):
        pass
    @abstractmethod
    def encode(self, val):
        pass
    @abstractmethod
    def decode(self, val):
        pass

class Binary(Encoder):
    """
    O or 1
    """
    def __init__(self):
        super().__init__()
        self.symbols = ['0', '1']

    def encode(self, value):
        return [str(value)]

    def decode(self, lst):
        if len(lst) != 1 or (lst[0] != '0' and lst[0] != '1'):
            return None
        return str(lst[0])

class Integer(Encoder):
    """
    Single integers, in base params.base
    """
    def __init__(self, params, base):
        super().__init__()
        self.base = base
        self.modulus = params.modulus
        self.sign_last = params.sign_last
        self.symbols = [str(i) for i in range(self.base)]

    def encode(self, value):
        if self.modulus > 1:
            value = value % self.modulus
        if value != 0:
            prefix = []
            w = abs(value)
            while w > 0:
                prefix.append(str(w % self.base))
                w = w // self.base
            prefix = prefix[::-1]
        else:
            prefix =['0']
        if self.sign_last:
            prefix = prefix + (['+'] if value >= 0 else ['-'])
        else:
            prefix = (['+'] if value >= 0 else ['-']) + prefix
        return prefix

    def decode(self, lst):
        if len(lst) < 1:
            return None
        if self.sign_last:
            if lst[-1] != '+' and lst[-1] != '-':
                return None
            if len(lst) == 1:
                return 0
            res = 0
            for x in lst[:-1]:
                if not (x.isdigit()):
                    return None
                res = res * self.base + int(x)
            return -res if lst[-1] == '-' else res
            
        if len(lst) < 1 or (lst[0] != '+' and lst[0] != '-'):
            return None
        if len(lst) == 1:
            return 0
        res = 0
        for x in lst[1:]:
            if not (x.isdigit()):
                return None
            res = res * self.base + int(x)
        return -res if lst[0] == '-' else res

class IntegerVector(Encoder):
    """
    Single integers, in base params.base
    """
    def __init__(self, params, base):
        super().__init__()
        self.base = base
        self.modulus = params.modulus
        self.max_vect_len = params.max_vect_len
        self.symbols = [str(i) for i in range(self.base)] + ['N'+str(i) for i in range(1, self.max_vect_len+1)]

    def write_int(self, value):
        if self.modulus > 1:
            value = value % self.modulus
        if value != 0:
            prefix = []
            w = abs(value)
            while w > 0:
                prefix.append(str(w % self.base))
                w = w // self.base
            prefix = prefix[::-1]
        else:
            prefix =['0']
        prefix = (['+'] if value >= 0 else ['-']) + prefix
        return prefix

    def parse_int(self, lst):
        if len(lst) <= 1 or (lst[0] != '+' and lst[0] != '-'):
            return None, 0
        res = 0
        pos = 1
        for x in lst[1:]:
            if not (x.isdigit()):
                return None, pos
            res = res * self.base + int(x)
            pos += 1
        if lst[0] == '-': res = -res
        return res, pos

    def encode(self, vector):
        lst = []
        l = len(vector)
        lst.append("N" + str(l))
        for val in vector:
            lst.extend(self.write_int(val))
        return lst

    def decode(self, lst):
        if len(lst) < 1 or lst[0][0] != "N":
            return None
        nr_lines = int(lst[0][1:])
        h = lst[1:]
        m = []
        for _ in range(nr_lines):
            val, pos = self.parse_int(h)
            if val is None:
                return None
            h = h[pos:]
            m.append(val)
        return m

class IntegerPrimeEnc(Encoder):
    """
    Single integers, in base params.base
    """
    def __init__(self, params, biggest_prime_index, biggest_power,base):
        super().__init__()
        self.biggest_power = biggest_power
        self.biggest_prime_index = biggest_prime_index
        self.modulus = params.modulus
        self.sign_last = params.sign_last
        self.base = base
        self.symbols = ['0', '1']+ [str(i) for i in range(self.base)] + [f"E{i}" for i in range(biggest_power)]
        #for now, only support leading primes < base
        if prime(self.biggest_prime_index) > self.base: raise ValueError

    def encode(self, value):
        prefix=[]

        if abs(value) == 1:
            factors = {1: 1}
        else:
            factors = factorint(abs(value))
        remainder = value
        for fac in factors.keys():
            if fac < prime(self.biggest_prime_index) and fac > 0:
                remainder = int(remainder / pow(fac,factors[fac]))
                prefix += ([f"{fac}"] + [f"E{factors[fac]}"])
        if remainder != 0:
            suffix = []
            w = abs(remainder)
            if w != 1:
                while w > 0:
                    suffix.append(str(w % self.base))
                    w = w // self.base
                prefix += suffix[::-1]
        else:
            prefix = ['0']
        prefix = (['+'] if value >= 0 else ['-']) + prefix
        return prefix

    def decode(self, lst):
        if len(lst) < 1:
            return None
        if self.sign_last:
            if lst[-1] != '+' and lst[-1] != '-':
                return None
            res = 1
            primelst=[]
            remainderlst=[]
            numlst=lst[:-1]
            for x in range(len(numlst)):
                if ('E' in numlst[x]) or (x < len(numlst)-1 and 'E' in numlst[x+1]):
                    primelst.append(numlst[x])
                else: remainderlst.append(numlst[x])

            if len(remainderlst) == 0:
                remainder = 1
            else:
                remainder = 0

            for x in range(len(primelst)):
                if x % 2 == 0:
                    if not (numlst[x].isdigit()) and numlst[x+1].replace('E','').isdigit():
                        return None
                    res *= (int(primelst[x])**int(primelst[x+1].replace('E','')))
            for x in remainderlst:
                if not (x.isdigit()):
                    return None
                remainder = remainder * self.base + int(x)
            res = res * remainder
            return -res if lst[-1] == '-' else res

        if len(lst) < 1 or (lst[0] != '+' and lst[0] != '-'):
            return None
        res = 1
        primelst = []
        remainderlst = []
        numlst = lst[1:]
        for x in range(len(numlst)):
            if ('E' in numlst[x]) or (x < len(numlst)-1 and 'E' in numlst[x + 1]):
                primelst.append(numlst[x])
            else:
                remainderlst.append(numlst[x])

        if len(remainderlst) == 0:
            remainder = 1
        else: remainder = 0

        for x in range(len(primelst)):
            if x % 2 == 0:
                if not (primelst[x].isdigit() and primelst[x + 1].replace('E', '').isdigit()):
                    return None
                res *= (int(primelst[x]) ** int(primelst[x + 1].replace('E', '')))
        for x in remainderlst:
            if not (x.isdigit()):
                return None
            remainder = remainder * self.base + int(x)
        res = res * remainder
        return -res if lst[0] == '-' else res

class IntegerVectorPrimeEnc(Encoder):
    """
    Single integers, in base params.base
    """

    def __init__(self, params, biggest_prime_index, biggest_power, base):
        super().__init__()
        self.modulus = params.modulus
        self.base = base
        self.biggest_prime_index = biggest_prime_index
        self.biggest_power = biggest_power
        self.max_vect_len = params.max_vect_len
        self.symbols = (['0', '1']+ [str(i) for i in range(self.base)] + [f"E{i}" for i in range(biggest_power)]
                        + ['N' + str(i) for i in range(1, self.max_vect_len + 1)])

    def write_int(self, value):
        prefix=[]
        if abs(value) == 1:
            factors = {1: 1}
        else:
            factors = factorint(abs(value))
        remainder = value
        for fac in factors.keys():
            if fac < prime(self.biggest_prime_index) and fac > 0:
                remainder = int(remainder / pow(fac,factors[fac]))
                prefix += ([f"{fac}"] + [f"E{factors[fac]}"])
        if remainder != 0:
            suffix = []
            w = abs(remainder)
            if w != 1:
                while w > 0:
                    suffix.append(str(w % self.base))
                    w = w // self.base
                prefix += suffix[::-1]
        else:
            prefix = ['0']
        prefix = (['+'] if value >= 0 else ['-']) + prefix
        return prefix

    def parse_int(self, lst):
        if len(lst) < 1 or (lst[0] != '+' and lst[0] != '-'):
            return None,0
        res = 1
        pos = 1
        primelst = []
        remainderlst = []
        numlst = lst[1:]
        for x in range(len(numlst)):
            if ('E' in numlst[x]) or (x < len(numlst)-1 and 'E' in numlst[x + 1]):
                primelst.append(numlst[x])
            else:
                remainderlst.append(numlst[x])
            pos += 1

        if len(remainderlst) == 0:
            remainder = 1
        else: remainder = 0

        for x in range(len(primelst)):
            if x % 2 == 0:
                if not (primelst[x].isdigit() and primelst[x + 1].replace('E', '').isdigit()):
                    return None, pos
                res *= (int(primelst[x]) ** int(primelst[x + 1].replace('E', '')))

        for x in remainderlst:
            if not (x.isdigit()):
                return None,pos
            remainder = remainder * self.base + int(x)

        res = res * remainder

        return -res, pos if lst[0] == '-' else res, pos

    def encode(self, vector):
        lst = []
        l = len(vector)
        lst.append("N" + str(l))
        for val in vector:
            lst.extend(self.write_int(val))
        return lst

    def decode(self, lst):
        if len(lst) < 1 or lst[0][0] != "N":
            return None
        nr_lines = int(lst[0][1:])
        h = lst[1:]
        m = []
        for _ in range(nr_lines):
            val, pos = self.parse_int(h)
            if val is None:
                return None
            h = h[pos:]
            m.append(val)
        return m

class WordRun(Encoder):
    """
    Encode a word in 'run' notation
    """
    def __init__(self, params):
        super().__init__()
        self.symbols = ['a', 'b', 'c', 'd', 'e', 'f','g','h'] + [f'r{str(i)}' for i in range(30)]

    def encode(self, sent):
        thischar=0
        counter=1
        runs=[]
        for i in range(len(sent)):
            if sent[i]==thischar:
                counter += 1
            else:
                thischar=sent[i]
                runs += sent[i-1]
                runs += [f'r{counter}']
                counter = 1
        #print(runs)
        return runs

    def decode(self, runs):
        currchar=''
        sent=[]
        for i in range(len(runs)):
            if not any(char.isdigit() for char in runs[i]):
                currchar = runs[i]
            else:
                sent += int(''.join(filter(str.isdigit, runs[i]))) * [currchar]
                currchar = 'X' #use 'X' to stand for a runs tag in situations where output looks like r1r2r1
        return sent

class WordBase(Encoder):
    """
    Encode a word in 'run' notation
    """

    def __init__(self, params):
        super().__init__()
        self.symbols = ['a', 'b', 'c', 'd', 'e', 'f','g','h']

    def encode(self, sent):
        return sent

    def decode(self, runs):
        return runs