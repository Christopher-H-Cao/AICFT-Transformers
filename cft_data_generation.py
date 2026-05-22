from fractions import Fraction
from functools import reduce
from itertools import combinations
import pandas as pd
from itertools import combinations_with_replacement  # Updated import

# tensor product both distinct CFTs and identical CFTs: (2,1,1,1) can appear instead of only (4,3,2,1)
level_data = [
    {'level': 1,
     'conformal_dimensions': [Fraction(0, 1), Fraction(1, 6), Fraction(1, 2), Fraction(5, 8), Fraction(8, 9)],
     'central_charge': Fraction(5, 1)},
    {'level': 2,
     'conformal_dimensions': [Fraction(0, 1), Fraction(1, 20), Fraction(9, 20), Fraction(9, 16), Fraction(4, 5)],
     'central_charge': Fraction(9, 1)},
    {'level': 3,
     'conformal_dimensions': [Fraction(0, 1), Fraction(9, 22), Fraction(45, 88), Fraction(8, 11), Fraction(21, 22)],
     'central_charge': Fraction(135, 11)},
    # Add further levels similarly if needed
]

# Function to tensor combine two rows with a limit of conformal dimensions
def tensor_combine(level_a, level_b, max_dimensions):
    combined_set = set()
    for d1 in level_a['conformal_dimensions']:
        for d2 in level_b['conformal_dimensions']:
            combined_set.add(d1 + d2)
            if len(combined_set) >= max_dimensions:
                break
        if len(combined_set) >= max_dimensions:
            break
    combined_dimensions = sorted(combined_set)

    central_charges_a = level_a.get('central_charges', [level_a.get('central_charge', None)])
    central_charges_b = level_b.get('central_charges', [level_b.get('central_charge', None)])

    # Remove None values in case some levels don't have central_charge defined
    central_charges_a = [c for c in central_charges_a if c is not None]
    central_charges_b = [c for c in central_charges_b if c is not None]

    combined_central_charges = sorted(central_charges_a + central_charges_b, reverse=True)

    return {
        'conformal_dimensions': combined_dimensions,
        'central_charges': combined_central_charges
    }

# Function to tensor combine n theories (multiple rows) with a dimension limit
def tensor_combine_n(level_data, level_numbers, max_dimensions=40):
    selected_levels = [level_data[num - 1] for num in level_numbers]  # Adjust for 0-based indexing
    return reduce(lambda x, y: tensor_combine(x, y, max_dimensions), selected_levels)

# Helper function to convert fractions to explicit "numerator/denominator" form
def fraction_str(frac):
    return f"{frac.numerator}/{frac.denominator}"

# Store combined rows in formatted list
formatted_rows = []

combination_size = 3
all_levels = [level['level'] for level in level_data]

max_combinations = 100
counter = 1

# Use combinations_with_replacement to allow repeated levels
for levels_combination in combinations_with_replacement(all_levels, combination_size):
    if counter > max_combinations:
        break
    combined_row = tensor_combine_n(level_data, levels_combination)

    dimensions_str = ' '.join(fraction_str(d) for d in combined_row['conformal_dimensions'])
    charges_str = ' '.join(fraction_str(c) for c in combined_row['central_charges'])

    formatted_rows.append(f"{counter}|{dimensions_str}\t{charges_str}")
    counter += 1

# Run this cell to make all integers such as 1,2,3,4... to present as 1/1, 2/1, 3/1, 4/1
from fractions import Fraction as _F
class Fraction(_F):
    def __str__(self):
        return f"{self.numerator}/{self.denominator}"
    __repr__ = __str__


# This function loops through all possible representations for each level of CFT
def iter_dynkin_labels(k, r):
    """
    Yield all r-tuples of non-negative integers whose sum is <= k.
    """
    # Base case: only one label left, it must absorb the remainder.
    if r == 1:
        for x in range(k+1):
            yield [x]
    else:
        # Pick the first label from 0..k, then distribute the rest to the other r-1 labels.
        for x in range(k+1):
            for tail in iter_dynkin_labels(k-x, r-1):
                yield [x] + tail



def conformal_dimensions_so7(max_level):
    r = 3  # rank for so(7)
    g_dual_coxeter = 2*r-1  # dual Coxeter number for so(2r)

    # Quadratic form matrix G provided explicitly for SO(10)
    G = [
        [Fraction(1,1), Fraction(1,1), Fraction(1,2)],
        [Fraction(1,1), Fraction(2,1), Fraction(1,1)],
        [Fraction(1,2), Fraction(1,1), Fraction(3,4)],
    ]

    omegas = [[int(i == j) for i in range(r)] for j in range(r)]  # fundamental weights
    rho = [1]*r  # Weyl vector for SO(2r)
    theta = [0,1,0]

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    def inner_product_k(v1,v2):
        return sum(v1[i]*v2[i] for i in range(len(v1)))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()

        # trivial representation dimension always 0
        dims.add(Fraction(0))

        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec, theta) <=k:
                #print(inner_product(lambda_vec,theta), k)
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 10:
                break

        # fundamental weights conformal dimensions
        # for omega in omegas:
        #   if inner_product(omega,theta) <=k:
        #     #print(k, inner_product(omega, theta), omega, theta)
        #     casimir = inner_product(omega, omega) + 2 * inner_product(omega, rho)
        #     h = Fraction(casimir, denominator)
        #     dims.add(h % 1)
        #     if len(dims) >= 20:
        #       break

        # central charge for so(2r)_k
        c = Fraction(k * r * (2 * r + 1), k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'so7,{c}'
            #'central_charge': f'so7'   # exclude central charge
        })

    return results

# Example: Generate results up to level 10
levels_data_so7 = conformal_dimensions_so7(15)
#print(levels_data_so10)


def conformal_dimensions_so8(max_level):
    r = 4  # rank for so(8)
    g_dual_coxeter = 2 * r - 2  # dual Coxeter number for so(2r)

    # Quadratic form matrix G provided explicitly for SO(8)
    G = [
        [Fraction(2,2), Fraction(2,2), Fraction(1,2), Fraction(1,2)],
        [Fraction(2,2), Fraction(4,2), Fraction(2,2), Fraction(2,2)],
        [Fraction(1,2), Fraction(2,2), Fraction(2,2), Fraction(1,2)],
        [Fraction(1,2), Fraction(2,2), Fraction(1,2), Fraction(2,2)],
    ]

    omegas = [[int(i == j) for i in range(r)] for j in range(r)]  # fundamental weights
    rho = [1]*r  # Weyl vector for SO(2r)
    theta = [0,1,0,0]

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    def inner_product_k(v1,v2):
        return sum(v1[i]*v2[i] for i in range(len(v1)))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()

        # trivial representation dimension always 0
        dims.add(Fraction(0))

        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec, theta) <=k:
                #print(inner_product(lambda_vec,theta), k)
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 10:
                break

        # fundamental weights conformal dimensions
        # for omega in omegas:
        #   if inner_product(omega,theta) <=k:
        #     #print(k, inner_product(omega, theta), omega, theta)
        #     casimir = inner_product(omega, omega) + 2 * inner_product(omega, rho)
        #     h = Fraction(casimir, denominator)
        #     dims.add(h % 1)
        #     if len(dims) >= 20:
        #       break

        # central charge for so(2r)_k
        c = Fraction(k * r * (2 * r - 1), k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'so8,{c}'
            #'central_charge': f'so8'   # exclude central charge
        })

    return results

# Example: Generate results up to level 10
levels_data_so8 = conformal_dimensions_so8(15)


def conformal_dimensions_so9(max_level):
    r = 4  # rank for so(9)
    g_dual_coxeter = 2*r-1  # dual Coxeter number for so(2r+1)

    # Quadratic form matrix G provided explicitly for SO(9)
    G = [
        [Fraction(1,1), Fraction(1,1), Fraction(1,1), Fraction(1,2)],
        [Fraction(1,1), Fraction(2,1), Fraction(2,1), Fraction(1,1)],
        [Fraction(1,1), Fraction(2,1), Fraction(3,1), Fraction(3,2)],
        [Fraction(1,2), Fraction(1,1), Fraction(3,2), Fraction(1,1)]
    ]

    omegas = [[int(i == j) for i in range(r)] for j in range(r)]  # fundamental weights
    rho = [1]*r  # Weyl vector for SO(2r+1)
    theta = [0,1,0,0]

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    def inner_product_k(v1,v2):
        return sum(v1[i]*v2[i] for i in range(len(v1)))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()

        # trivial representation dimension always 0
        dims.add(Fraction(0))

        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec, theta) <=k:
                #print(inner_product(lambda_vec,theta), k)
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 10:
                break

        # fundamental weights conformal dimensions
        # for omega in omegas:
        #   if inner_product(omega,theta) <=k:
        #     #print(k, inner_product(omega, theta), omega, theta)
        #     casimir = inner_product(omega, omega) + 2 * inner_product(omega, rho)
        #     h = Fraction(casimir, denominator)
        #     dims.add(h % 1)
        #     if len(dims) >= 20:
        #       break

        # central charge for so(2r)_k
        c = Fraction(k * r * (2 * r + 1), k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'so9,{c}'
            #'central_charge': f'so9'   # exclude central charge
        })

    return results

# Example: Generate results up to level 10
levels_data_so9 = conformal_dimensions_so9(15)
#print(levels_data_so10)

def conformal_dimensions_so10(max_level):
    r = 5  # rank for so(10)
    g_dual_coxeter = 2 * r - 2  # dual Coxeter number for so(2r)

    # Quadratic form matrix G provided explicitly for SO(10)
    G = [
        [Fraction(2,2), Fraction(2,2), Fraction(2,2), Fraction(1,2), Fraction(1,2)],
        [Fraction(2,2), Fraction(4,2), Fraction(4,2), Fraction(2,2), Fraction(2,2)],
        [Fraction(2,2), Fraction(4,2), Fraction(6,2), Fraction(3,2), Fraction(3,2)],
        [Fraction(1,2), Fraction(2,2), Fraction(3,2), Fraction(5,4), Fraction(3,4)],
        [Fraction(1,2), Fraction(2,2), Fraction(3,2), Fraction(3,4), Fraction(5,4)],
    ]

    omegas = [[int(i == j) for i in range(r)] for j in range(r)]  # fundamental weights
    rho = [1]*r  # Weyl vector for SO(2r)
    theta = [0,1,0,0,0]

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    def inner_product_k(v1,v2):
        return sum(v1[i]*v2[i] for i in range(len(v1)))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()

        # trivial representation dimension always 0
        dims.add(Fraction(0))

        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec, theta) <=k:
                #print(inner_product(lambda_vec,theta), k)
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 10:
                break

        # fundamental weights conformal dimensions
        # for omega in omegas:
        #   if inner_product(omega,theta) <=k:
        #     #print(k, inner_product(omega, theta), omega, theta)
        #     casimir = inner_product(omega, omega) + 2 * inner_product(omega, rho)
        #     h = Fraction(casimir, denominator)
        #     dims.add(h % 1)
        #     if len(dims) >= 20:
        #       break

        # central charge for so(2r)_k
        c = Fraction(k * r * (2 * r - 1), k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'so10,{c}'
            #'central_charge': f'so10'   # exclude central charge
        })

    return results

# Example: Generate results up to level 10
levels_data_so10 = conformal_dimensions_so10(15)
#print(levels_data_so10)

def conformal_dimensions_so11(max_level):
    r = 5  # rank for so(9)
    g_dual_coxeter = 2*r-1  # dual Coxeter number for so(2r+1)

    # Quadratic form matrix G provided explicitly for SO(9)
    G = [
        [Fraction(2,2), Fraction(2,2), Fraction(2,2), Fraction(2,2), Fraction(1,2)],
        [Fraction(2,2), Fraction(4,2), Fraction(4,2), Fraction(4,2), Fraction(2,2)],
        [Fraction(2,2), Fraction(4,2), Fraction(6,2), Fraction(6,2), Fraction(3,2)],
        [Fraction(2,2), Fraction(4,2), Fraction(6,2), Fraction(8,2), Fraction(4,2)],
        [Fraction(1,2), Fraction(2,2), Fraction(3,2), Fraction(4,2), Fraction(5,2)],
    ]

    omegas = [[int(i == j) for i in range(r)] for j in range(r)]  # fundamental weights
    rho = [1]*r  # Weyl vector for SO(2r+1)
    theta = [0,1,0,0,0]

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    def inner_product_k(v1,v2):
        return sum(v1[i]*v2[i] for i in range(len(v1)))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()

        # trivial representation dimension always 0
        dims.add(Fraction(0))

        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec, theta) <=k:
                #print(inner_product(lambda_vec,theta), k)
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 10:
                break

        # fundamental weights conformal dimensions
        # for omega in omegas:
        #   if inner_product(omega,theta) <=k:
        #     #print(k, inner_product(omega, theta), omega, theta)
        #     casimir = inner_product(omega, omega) + 2 * inner_product(omega, rho)
        #     h = Fraction(casimir, denominator)
        #     dims.add(h % 1)
        #     if len(dims) >= 20:
        #       break

        # central charge for so(2r)_k
        c = Fraction(k * r * (2 * r + 1), k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'so11,{c}'
            #'central_charge': f'so11'   # exclude central charge
        })

    return results

# Example: Generate results up to level 10
levels_data_so11 = conformal_dimensions_so11(15)
#print(levels_data_so10)

def conformal_dimensions_so12(max_level):
    r = 6  # rank for so(10)
    g_dual_coxeter = 2 * r - 2  # dual Coxeter number for so(2r)

    # Quadratic form matrix G provided explicitly for SO(10)
    G = [
        [Fraction(2,2), Fraction(2,2), Fraction(2,2), Fraction(2,2), Fraction(1,2), Fraction(1,2)],
        [Fraction(2,2), Fraction(4,2), Fraction(4,2), Fraction(4,2), Fraction(2,2), Fraction(2,2)],
        [Fraction(2,2), Fraction(4,2), Fraction(6,2), Fraction(6,2), Fraction(3,2), Fraction(3,2)],
        [Fraction(2,2), Fraction(4,2), Fraction(6,2), Fraction(8,2), Fraction(4,2), Fraction(4,2)],
        [Fraction(1,2), Fraction(2,2), Fraction(3,2), Fraction(4,2), Fraction(3,2), Fraction(2,2)],
        [Fraction(1,2), Fraction(2,2), Fraction(3,2), Fraction(4,2), Fraction(2,2), Fraction(3,2)],
    ]

    omegas = [[int(i == j) for i in range(r)] for j in range(r)]  # fundamental weights
    rho = [1]*r  # Weyl vector for SO(2r)
    theta = [0,1,0,0,0,0]

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    def inner_product_k(v1,v2):
        return sum(v1[i]*v2[i] for i in range(len(v1)))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()

        # trivial representation dimension always 0
        dims.add(Fraction(0))

        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec, theta) <=k:
                #print(inner_product(lambda_vec,theta), k)
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 10:
                break

        # fundamental weights conformal dimensions
        # for omega in omegas:
        #   if inner_product(omega,theta) <=k:
        #     #print(k, inner_product(omega, theta), omega, theta)
        #     casimir = inner_product(omega, omega) + 2 * inner_product(omega, rho)
        #     h = Fraction(casimir, denominator)
        #     dims.add(h % 1)
        #     if len(dims) >= 20:
        #       break

        # central charge for so(2r)_k
        c = Fraction(k * r * (2 * r - 1), k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'so12,{c}'
            #'central_charge': f'so12'   # exclude central charge
        })

    return results

# Example: Generate results up to level 10
levels_data_so12 = conformal_dimensions_so12(15)
#print(levels_data_so12)

# Redefine parameters for so(16)
r_so16 = 8  # rank
dim_so16 = 2 * r_so16 ** 2 - r_so16  # dimension
g_dual_coxeter_so16 = 2 * r_so16 - 2  # dual Coxeter number

# Construct the Gram matrix for so(16)
G_so16 = [
    [Fraction(1,1), Fraction(1,1), Fraction(1,1), Fraction(1,1), Fraction(1,1), Fraction(1,1), Fraction(1,2), Fraction(1,2)],
    [Fraction(1,1), Fraction(2,1), Fraction(2,1), Fraction(2,1), Fraction(2,1), Fraction(2,1), Fraction(1,1), Fraction(1,1)],
    [Fraction(1,1), Fraction(2,1), Fraction(3,1), Fraction(3,1), Fraction(3,1), Fraction(3,1), Fraction(3,2), Fraction(3,2)],
    [Fraction(1,1), Fraction(2,1), Fraction(3,1), Fraction(4,1), Fraction(4,1), Fraction(4,1), Fraction(2,1), Fraction(2,1)],
    [Fraction(1,1), Fraction(2,1), Fraction(3,1), Fraction(4,1), Fraction(5,1), Fraction(5,1), Fraction(5,2), Fraction(5,2)],
    [Fraction(1,1), Fraction(2,1), Fraction(3,1), Fraction(4,1), Fraction(5,1), Fraction(6,1), Fraction(3,1), Fraction(3,1)],
    [Fraction(1,2), Fraction(1,1), Fraction(3,2), Fraction(2,1), Fraction(5,2), Fraction(3,1), Fraction(2,1), Fraction(3,2)],
    [Fraction(1,2), Fraction(1,1), Fraction(3,2), Fraction(2,1), Fraction(5,2), Fraction(3,1), Fraction(3,2), Fraction(2,1)],
]


# Define highest root θ = (0,1,0,...,0)
theta_so16 = [0] * r_so16
theta_so16[1] = 1

# Weyl vector ρ = (1,1,...,1)
rho_so16 = [1] * r_so16

# Inner product using the Gram matrix G_so16
def inner_product_so16(v1, v2):
    return sum(v1[i] * G_so16[i][j] * v2[j] for i in range(r_so16) for j in range(r_so16))

# Compute conformal dimensions and central charge for so(16) up to max_level
def conformal_dimensions_so16(max_level):
    results = []
    for k in range(1, max_level + 1):
        denominator = 2 * (k + g_dual_coxeter_so16)
        dims = set()
        dims.add(Fraction(0))  # Trivial representation

        for lambda_vec in iter_dynkin_labels(k, r_so16):
            if inner_product_so16(lambda_vec, theta_so16) <= k:
                casimir = inner_product_so16(lambda_vec, lambda_vec) + 2 * inner_product_so16(lambda_vec, rho_so16)
                h = Fraction(casimir, denominator)
                dims.add(h)
            if len(dims) >= 10:
                break

        c = Fraction(k * dim_so16, k + g_dual_coxeter_so16)
        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],  # exclude trivial
            'central_charge': f"so16,{c}"
        })

    return results

levels_data_so16 = conformal_dimensions_so16(15)  # Compute up to level 10

def conformal_dimensions_su2(max_level):
    results = []
    counter = 1
    for k in range(1, max_level + 1):
        dims = set()
        dims.add(Fraction(0))  # Trivial representation
        # l runs from 0 to k (inclusive)
        for l in range(0, k+1):
            h = Fraction(l * (l + 2), 4 * (k + 2))
            # If you want to mimic the mod 1 behavior as in the SO(10) example:
            dims.add(h)
            if len(dims) >= 15:
              break
        dims = sorted(dims)
        # Central charge for su(2)_k is c = 3k/(k+2)
        c = Fraction(3 * k, k + 2)
        #print(c)
        results.append({
            'level': k,
            'conformal_dimensions': dims[0:10],
            'central_charge': f'su2,{c}'
            #'central_charge': f'su2'
        })
        counter += 1
    return results

# Generate results for levels 1 to 10
levels_data_su2 = conformal_dimensions_su2(15)

def conformal_dimensions_su3(max_level):
    r = 2  # rank for su(3)
    g_dual_coxeter = r + 1  # dual Coxeter number for su(r+1)
    theta = [1,1]

    # Quadratic form matrix G explicitly for SU(3)
    G = [
        [Fraction(2,3), Fraction(1,3)],
        [Fraction(1,3), Fraction(2,3)],
    ]

    rho = [1,1]  # Weyl vector for SU(3)

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()
        dims.add(Fraction(0))  # Trivial representation

        # Consider all allowed representations with lambda1 + lambda2 <= k
        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec, theta) <= k:
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 15:
                break

        # central charge for su(r+1)_k
        c = Fraction(k * r * (r + 2), k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'su3,{c}'
            #'central_charge': f'su3'
        })

    return results

# Example: Generate results up to level 10
levels_data_su3 = conformal_dimensions_su3(15)[0:]


def conformal_dimensions_su4(max_level):
    r = 3  # rank for su(4)
    g_dual_coxeter = r + 1  # dual Coxeter number for su(r+1)
    theta = [1,0,1]

    # Quadratic form matrix G explicitly for SU(3)
    G = [
        [Fraction(3,4), Fraction(2,4), Fraction(1,4)],
        [Fraction(2,4), Fraction(4,4), Fraction(2,4)],
        [Fraction(1,4), Fraction(2,4), Fraction(3,4)]
    ]

    rho = [1]*r  # Weyl vector for SU(4)

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()
        dims.add(Fraction(0))  # Trivial representation

        # Consider all allowed representations with lambda1 + lambda2 <= k
        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec, theta) <= k:
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 15:
                break

        # central charge for su(r+1)_k
        c = Fraction(k * r * (r + 2), k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'su4,{c}'
            #'central_charge': f'su4'
        })

    return results

# Example: Generate results up to level 10
levels_data_su4 = conformal_dimensions_su4(15)[0:]


# Parameters for su(5)
r_su5 = 4  # rank = 5 - 1
dim_su5 = r_su5 ** 2 + 2 * r_su5  # dimension of su(5)
g_dual_coxeter_su5 = r_su5 + 1  # dual Coxeter number

# Construct the Gram matrix for su(5)
def quadratic_form_su(n):
    G = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            G[i][j] = Fraction(min(i + 1, j + 1) * (n - max(i, j)), n + 1)
    return G

G_su5 = quadratic_form_su(r_su5)

# Define highest root θ = (1, 0, ..., 1)
theta_su5 = [1,0,0,1]

# Weyl vector ρ = (1, 1, ..., 1)
rho_su5 = [1] * r_su5

# Inner product using the Gram matrix G_su5
def inner_product_su5(v1, v2):
    return sum(v1[i] * G_su5[i][j] * v2[j] for i in range(r_su5) for j in range(r_su5))

# Compute conformal dimensions and central charge for su(5) up to max_level
def conformal_dimensions_su5(max_level):
    results = []
    for k in range(1, max_level + 1):
        denominator = 2 * (k + g_dual_coxeter_su5)
        dims = set()
        dims.add(Fraction(0))  # Trivial representation

        for lambda_vec in iter_dynkin_labels(k, r_su5):
            if inner_product_su5(lambda_vec, theta_su5) <= k:
                casimir = inner_product_su5(lambda_vec, lambda_vec) + 2 * inner_product_su5(lambda_vec, rho_su5)
                h = Fraction(casimir, denominator)
                dims.add(h)
            if len(dims) >= 15:
                break

        c = Fraction(k * dim_su5, k + g_dual_coxeter_su5)
        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],  # exclude trivial
            'central_charge': f"su5,{c}"
        })

    return results

levels_data_su5 = conformal_dimensions_su5(15)


def conformal_dimensions_su6(max_level):
    r = 5  # rank for su(4)
    g_dual_coxeter = r + 1  # dual Coxeter number for su(r+1)
    theta = [1,0,0,0,1]

    # Quadratic form matrix G explicitly for SU(3)
    G = [
        [Fraction(5,6), Fraction(4,6), Fraction(3,6), Fraction(2,6), Fraction(1,6)],
        [Fraction(4,6), Fraction(8,6), Fraction(6,6), Fraction(4,6), Fraction(2,6)],
        [Fraction(3,6), Fraction(6,6), Fraction(9,6), Fraction(6,6), Fraction(3,6)],
        [Fraction(2,6), Fraction(4,6), Fraction(8,6), Fraction(8,6), Fraction(4,6)],
        [Fraction(1,6), Fraction(2,6), Fraction(4,6), Fraction(4,6), Fraction(5,6)]
    ]

    rho = [1]*r  # Weyl vector for SU(4)

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()
        dims.add(Fraction(0))  # Trivial representation

        # Consider all allowed representations with lambda1 + lambda2 <= k
        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec, theta) <= k:
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 15:
                break

        # central charge for su(r+1)_k
        c = Fraction(k * r * (r + 2), k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'su6,{c}'
            #'central_charge': f'su6'
        })

    return results

# Example: Generate results up to level 10
levels_data_su6 = conformal_dimensions_su6(15)[0:]


# this block is correct
def conformal_dimensions_sp4(max_level):
    g_dual_coxeter = 3  # sp(4) has dual Coxeter number 3
    r = 2

    # quadratic form matrix G in fundamental weights basis
    G = [[Fraction(1,2), Fraction(1,2)],
         [Fraction(1,2), Fraction(2,2)]]

    omega1 = [1,0]
    omega2 = [0,1]
    rho = [1,1]  # rho = omega1 + omega2
    theta = [2,0]

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(2) for j in range(2))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()

        # trivial representation dimension always 0
        dims.add(Fraction(0))

        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec, theta) <= k:
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 15:
                break

        # for omega in [omega1, omega2]:
        #   if inner_product(omega,theta) <= k:
        #     #print(inner_product(omega, theta), omega, theta)
        #     casimir = inner_product(omega, omega) + 2 * inner_product(omega, rho)
        #     h = Fraction(casimir, denominator)
        #     dims.add(h % 1)
        #     if len(dims) >= 10:
        #       break

        # central charge for sp(2r)_k (here r=2)
        c = Fraction(k * 10, k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'sp4,{c}'
            #'central_charge': f'sp4'
        })

    return results

level_k = 15

# Example: Generate results up to level 10
levels_data_sp4 = conformal_dimensions_sp4(level_k)[:]

def conformal_dimensions_sp6(max_level):
    r = 3  # rank for sp(8)
    g_dual_coxeter = r + 1  # sp(2r) dual Coxeter number is r + 1

    # Quadratic form matrix G for sp(8)
    G = [
        [Fraction(1,2), Fraction(1,2), Fraction(1,2)],
        [Fraction(1,2), Fraction(2,2), Fraction(2,2)],
        [Fraction(1,2), Fraction(2,2), Fraction(3,2)],
    ]

    omegas = [[int(i == j) for i in range(r)] for j in range(r)]  # fundamental weights
    rho = [1]*r  # Weyl vector for sp(2r)
    theta = [2,0,0]

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()

        # trivial representation dimension always 0
        dims.add(Fraction(0))

        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec,theta) <= k:
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 15:
                break

        # for omega in omegas:
        #   if inner_product(omega,theta) <= k:
        #     #print(inner_product(omega, theta), omega, theta)
        #     casimir = inner_product(omega, omega) + 2 * inner_product(omega, rho)
        #     h = Fraction(casimir, denominator)
        #     dims.add(h % 1)
        #     if len(dims) >= 10:
        #       break

        # central charge for sp(2r)_k
        c = Fraction(k * r * (2 * r + 1), k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'sp6,{c}'
            #'central_charge': f'sp6'
        })

    return results

# Example: Generate results up to level 10
levels_data_sp6 = conformal_dimensions_sp6(15)[:]


def conformal_dimensions_sp8(max_level):
    r = 4  # rank for sp(8)
    g_dual_coxeter = r + 1  # sp(2r) dual Coxeter number is r + 1

    # Quadratic form matrix G for sp(8)
    G = [
        [Fraction(1,2), Fraction(1,2), Fraction(1,2), Fraction(1,2)],
        [Fraction(1,2), Fraction(2,2), Fraction(2,2), Fraction(2,2)],
        [Fraction(1,2), Fraction(2,2), Fraction(3,2), Fraction(3,2)],
        [Fraction(1,2), Fraction(2,2), Fraction(3,2), Fraction(4,2)],
    ]

    omegas = [[int(i == j) for i in range(r)] for j in range(r)]  # fundamental weights
    rho = [1]*r  # Weyl vector for sp(2r)
    theta = [2,0,0,0]

    # Compute (lambda, mu)
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    results = []

    for k in range(1, max_level+1):
        denominator = 2 * (k + g_dual_coxeter)
        dims = set()

        # trivial representation dimension always 0
        dims.add(Fraction(0))

        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            if inner_product(lambda_vec,theta) <= k:
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 15:
                break

        # for omega in omegas:
        #   if inner_product(omega,theta) <= k:
        #     #print(inner_product(omega, theta), omega, theta)
        #     casimir = inner_product(omega, omega) + 2 * inner_product(omega, rho)
        #     h = Fraction(casimir, denominator)
        #     dims.add(h % 1)
        #     if len(dims) >= 10:
        #       break

        # central charge for sp(2r)_k
        c = Fraction(k * r * (2 * r + 1), k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],
            'central_charge': f'sp8,{c}'
            #'central_charge': f'sp8'
        })

    return results

# Example: Generate results up to level 10
levels_data_sp8 = conformal_dimensions_sp8(15)[:]


import itertools

# Quadratic form matrix for E6 from the image (scaled by 1/3)
G_E6 = [
    [Fraction(4, 3), Fraction(5, 3), Fraction(6, 3), Fraction(4, 3), Fraction(2, 3), Fraction(3, 3)],
    [Fraction(5, 3), Fraction(10, 3), Fraction(12, 3), Fraction(8, 3), Fraction(4, 3), Fraction(6, 3)],
    [Fraction(6, 3), Fraction(12, 3), Fraction(18, 3), Fraction(12, 3), Fraction(6, 3), Fraction(9, 3)],
    [Fraction(4, 3), Fraction(8, 3), Fraction(12, 3), Fraction(10, 3), Fraction(5, 3), Fraction(6, 3)],
    [Fraction(2, 3), Fraction(4, 3), Fraction(6, 3), Fraction(5, 3), Fraction(4, 3), Fraction(3, 3)],
    [Fraction(3, 3), Fraction(6, 3), Fraction(9, 3), Fraction(6, 3), Fraction(3, 3), Fraction(6, 3)]
]

# Cartan matrix tells us rank r = 6 for E6
r = 6

# Dual Coxeter number for E6 is g = 12
g_dual_coxeter_E6 = 12

# Dimension of E6 is 78
dim_E6 = 78

# Define the Weyl vector ρ as a vector of ones
rho_E6 = [1] * r

# Define the highest root theta for E6 as given: (0, 0, ..., 1)
theta_E6 = [0, 0, 0, 0, 0, 1]

# Construct the standard basis fundamental weights (identity vectors)
omegas_E6 = [[int(i == j) for i in range(r)] for j in range(r)]

# Inner product using the Gram matrix G_E6
def inner_product_E6(v1, v2):
    return sum(v1[i] * G_E6[i][j] * v2[j] for i in range(r) for j in range(r))

# Compute conformal dimensions and central charge for E6 up to given max_level
def conformal_dimensions_e6(max_level):
    results = []
    for k in range(1, max_level + 1):
        denominator = 2 * (k + g_dual_coxeter_E6)
        dims = set()
        dims.add(Fraction(0))  # Trivial representation

        for lambda_vec in iter_dynkin_labels(k, r):
            if inner_product_E6(lambda_vec, theta_E6) <= k:
                casimir = inner_product_E6(lambda_vec, lambda_vec) + 2 * inner_product_E6(lambda_vec, rho_E6)
                h = Fraction(casimir, denominator)
                dims.add(h)
            if len(dims) >= 15:
                break

        c = Fraction(k * dim_E6, k + g_dual_coxeter_E6)
        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],  # exclude the trivial 0
            'central_charge': f"e6,{c}"
        })

    return results

levels_data_e6 = conformal_dimensions_e6(15)[0:]


# Quadratic form matrix for E7 from the image (scaled by 1/2)
G_E7 = [
    [Fraction(4, 2), Fraction(6, 2), Fraction(8, 2), Fraction(6, 2), Fraction(4, 2), Fraction(2, 2), Fraction(4, 2)],
    [Fraction(6, 2), Fraction(12, 2), Fraction(16, 2), Fraction(12, 2), Fraction(8, 2), Fraction(4, 2), Fraction(8, 2)],
    [Fraction(8, 2), Fraction(16, 2), Fraction(24, 2), Fraction(18, 2), Fraction(12, 2), Fraction(6, 2), Fraction(12, 2)],
    [Fraction(6, 2), Fraction(12, 2), Fraction(18, 2), Fraction(15, 2), Fraction(10, 2), Fraction(5, 2), Fraction(9, 2)],
    [Fraction(4, 2), Fraction(8, 2), Fraction(12, 2), Fraction(10, 2), Fraction(8, 2), Fraction(4, 2), Fraction(6, 2)],
    [Fraction(2, 2), Fraction(4, 2), Fraction(6, 2), Fraction(5, 2), Fraction(4, 2), Fraction(3, 2), Fraction(3, 2)],
    [Fraction(4, 2), Fraction(8, 2), Fraction(12, 2), Fraction(9, 2), Fraction(6, 2), Fraction(3, 2), Fraction(7, 2)]
]

# Rank of E7
r_E7 = 7

# Dual Coxeter number for E7 is g = 18
g_dual_coxeter_E7 = 18

# Dimension of E7 is 133
dim_E7 = 133

# Weyl vector ρ for E7 is the vector of all ones
rho_E7 = [1] * r_E7

# Highest root θ for E7 is (1,0,...,0)
theta_E7 = [1] + [0] * (r_E7 - 1)

# Generate fundamental weights for E7
omegas_E7 = [[int(i == j) for i in range(r_E7)] for j in range(r_E7)]

# Inner product using the Gram matrix G_E7
def inner_product_E7(v1, v2):
    return sum(v1[i] * G_E7[i][j] * v2[j] for i in range(r_E7) for j in range(r_E7))

# Compute conformal dimensions and central charge for E7 up to max_level
def conformal_dimensions_e7(max_level):
    results = []
    for k in range(1, max_level + 1):
        denominator = 2 * (k + g_dual_coxeter_E7)
        dims = set()
        dims.add(Fraction(0))  # Trivial representation

        for lambda_vec in iter_dynkin_labels(k, r_E7):
            if inner_product_E7(lambda_vec, theta_E7) <= k:
                casimir = inner_product_E7(lambda_vec, lambda_vec) + 2 * inner_product_E7(lambda_vec, rho_E7)
                h = Fraction(casimir, denominator)
                dims.add(h)
            if len(dims) >= 15:
                break

        c = Fraction(k * dim_E7, k + g_dual_coxeter_E7)
        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],  # exclude trivial
            'central_charge': f"e7,{c}"
        })

    return results

levels_data_e7 = conformal_dimensions_e7(15)[0:]

def conformal_dimensions_e8(max_level):
    """
    Computes (for k=1,...,max_level):
      - The fractional parts of conformal dimensions for the trivial rep
        and each 'fundamental-weight' representation
      - The WZW central charge at level k

    Returns a list of dicts with keys:
      'level', 'conformal_dimensions', 'central_charge'
    """

    # Rank of E8
    r = 8

    # Dual Coxeter number of E8
    g_dual_coxeter = 30

    # Dimension of E8
    dim_e8 = 248

    # Quadratic form matrix (Gram matrix of fundamental weights) for E8
    # As shown in the image:
    G = [
        [Fraction(2), Fraction(3), Fraction(4), Fraction(5), Fraction(6), Fraction(4), Fraction(2), Fraction(3)],
        [Fraction(3), Fraction(6), Fraction(8), Fraction(10), Fraction(12), Fraction(8), Fraction(4), Fraction(6)],
        [Fraction(4), Fraction(8), Fraction(12), Fraction(15), Fraction(18), Fraction(12), Fraction(6), Fraction(9)],
        [Fraction(5), Fraction(10), Fraction(15), Fraction(20), Fraction(24), Fraction(16), Fraction(8), Fraction(12)],
        [Fraction(6), Fraction(12), Fraction(18), Fraction(24), Fraction(30), Fraction(20), Fraction(8), Fraction(12)],
        [Fraction(4), Fraction(8), Fraction(12), Fraction(16), Fraction(20), Fraction(14), Fraction(7), Fraction(10)],
        [Fraction(2), Fraction(4), Fraction(6), Fraction(8), Fraction(10), Fraction(7), Fraction(4), Fraction(5)],
        [Fraction(3), Fraction(6), Fraction(9), Fraction(12), Fraction(15), Fraction(10), Fraction(5), Fraction(8)]
    ]

    # We'll mimic your approach by treating the fundamental weights
    # as the standard basis vectors e_1, ..., e_8
    omegas = [[int(i == j) for i in range(r)] for j in range(r)]

    # A simple placeholder for the Weyl vector (like your so(10) code).
    # In a more precise calculation, one would use the actual expression for ρ.
    rho = [1] * r

    theta = [1,0,0,0,0,0,0,0]

    # Inner product using the Gram matrix G
    def inner_product(v1, v2):
        return sum(v1[i]*G[i][j]*v2[j] for i in range(r) for j in range(r))

    # def inner_product_k(v1,v2):
    #     return sum(v1[i]*v2[i] for i in range(len(v1)))

    results = []

    for k in range(1, max_level + 1):
        denominator = 2 * (k + g_dual_coxeter)

        dims = set()
        # Trivial representation has conformal dimension 0
        dims.add(Fraction(0))

        for lambda_vec in iter_dynkin_labels(k, r):
            # lambda_vec is now a list of length r (e.g. [λ1,λ2,λ3] for su(4))
            #print(lambda_vec)
            if inner_product(lambda_vec, theta) <= k:
                #print(lambda_vec,k)
                casimir = inner_product(lambda_vec, lambda_vec) \
                          + 2*inner_product(lambda_vec, rho)
                h = Fraction(casimir, denominator)
                dims.add(h)

        # if you only want the first N distinct dimensions:
            if len(dims) >= 15:
                break

        # Compute fractional part of h for each fundamental weight
        # for omega in omegas:
        #     if inner_product(omega, theta) <= k:
        #         #print(inner_product(omega, theta), omega, theta)
        #         casimir = inner_product(omega, omega) + 2 * inner_product(omega, rho)
        #         h = Fraction(casimir, denominator)
        #         dims.add(h % 1)

        # Central charge for E8 at level k
        c = Fraction(k * dim_e8, k + g_dual_coxeter)

        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10], # delete the 0/1 trivial spectrum
            'central_charge': f'e8,{c}'
            #'central_charge': f'e8'
        })

    return results

# Example usage: compute up to level 10
levels_data_e8 = conformal_dimensions_e8(15)[:]

# Quadratic form matrix for F4 from the image
G_F4 = [
    [Fraction(2), Fraction(3), Fraction(2), Fraction(1)],
    [Fraction(3), Fraction(6), Fraction(4), Fraction(2)],
    [Fraction(2), Fraction(4), Fraction(3), Fraction(3, 2)],
    [Fraction(1), Fraction(2), Fraction(3, 2), Fraction(1)]
]

# Rank of F4
r_F4 = 4

# Dual Coxeter number for F4 is g = 9
g_dual_coxeter_F4 = 9

# Dimension of F4 is 52
dim_F4 = 52

# Weyl vector ρ for F4 is the vector of all ones
rho_F4 = [1] * r_F4

# Highest root θ for F4 is (1, 0, 0, 0)
theta_F4 = [1] + [0] * (r_F4 - 1)

# Generate fundamental weights for F4
omegas_F4 = [[int(i == j) for i in range(r_F4)] for j in range(r_F4)]

# Inner product using the Gram matrix G_F4
def inner_product_F4(v1, v2):
    return sum(v1[i] * G_F4[i][j] * v2[j] for i in range(r_F4) for j in range(r_F4))

# Compute conformal dimensions and central charge for F4 up to max_level
def conformal_dimensions_f4(max_level):
    results = []
    for k in range(1, max_level + 1):
        denominator = 2 * (k + g_dual_coxeter_F4)
        dims = set()
        dims.add(Fraction(0))  # Trivial representation

        for lambda_vec in iter_dynkin_labels(k, r_F4):
            if inner_product_F4(lambda_vec, theta_F4) <= k:
                casimir = inner_product_F4(lambda_vec, lambda_vec) + 2 * inner_product_F4(lambda_vec, rho_F4)
                h = Fraction(casimir, denominator)
                dims.add(h)
            if len(dims) >= 15:
                break

        c = Fraction(k * dim_F4, k + g_dual_coxeter_F4)
        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],  # exclude trivial
            'central_charge': f"f4,{c}"
        })

    return results

levels_data_f4 = conformal_dimensions_f4(15)[0:]

# Re-import necessary packages after code execution environment reset
import itertools
import pandas as pd

# Quadratic form matrix for G2 from the image (scaled by 1/3)
G_G2 = [
    [Fraction(6, 3), Fraction(3, 3)],
    [Fraction(3, 3), Fraction(2, 3)]
]

# Rank of G2
r_G2 = 2

# Dual Coxeter number for G2 is g = 4
g_dual_coxeter_G2 = 4

# Dimension of G2 is 14
dim_G2 = 14

# Weyl vector ρ for G2 is the vector of all ones
rho_G2 = [1] * r_G2

# Highest root θ for G2 is (1, 0)
theta_G2 = [1, 0]

# Generate fundamental weights for G2
omegas_G2 = [[int(i == j) for i in range(r_G2)] for j in range(r_G2)]

# Inner product using the Gram matrix G_G2
def inner_product_G2(v1, v2):
    return sum(v1[i] * G_G2[i][j] * v2[j] for i in range(r_G2) for j in range(r_G2))

# Compute conformal dimensions and central charge for G2 up to max_level
def conformal_dimensions_g2(max_level):
    results = []
    for k in range(1, max_level + 1):
        denominator = 2 * (k + g_dual_coxeter_G2)
        dims = set()
        dims.add(Fraction(0))  # Trivial representation

        for lambda_vec in iter_dynkin_labels(k, r_G2):
            if inner_product_G2(lambda_vec, theta_G2) <= k:
                casimir = inner_product_G2(lambda_vec, lambda_vec) + 2 * inner_product_G2(lambda_vec, rho_G2)
                h = Fraction(casimir, denominator)
                dims.add(h)
            if len(dims) >= 15:
                break

        c = Fraction(k * dim_G2, k + g_dual_coxeter_G2)
        results.append({
            'level': k,
            'conformal_dimensions': sorted(dims)[0:10],  # exclude trivial
            'central_charge': f"g2,{c}"
        })

    return results

levels_data_g2 = conformal_dimensions_g2(15)[0:]

def conformal_dimensions_minimal_models(max_q):

    results = []
    # Loop over possible q values; for each, p = q+1.
    for q in range(2, max_q + 1):
        p = q + 1
        dims = set()
        dims.add(Fraction(0))
        for r in range(1, q):        # r = 1, 2, ..., q-1
            for s in range(1, p):    # s = 1, 2, ..., p-1 (i.e. 1,...,q)
                h = Fraction((p * r - q * s)**2 - 1, 4 * p * q)
                #print(h)
                dims.add(h)  # Taking modulo 1 as in the previous examples.
                if len(dims) >= 20:
                    break
        dims = sorted(dims)
        c = Fraction(1) - Fraction(6, p * q)
        results.append({
            'level': p,
            'conformal_dimensions': dims[0:10],
            'central_charge': f'minimal,{c}'
            #'central_charge': f'minimal'
        })
    return results

# Generate results for unitary minimal models with q from 2 up to 10.
levels_data_minimal = conformal_dimensions_minimal_models(17)[1:]

#from fractions import Fraction
import pandas as pd

def conformal_dimensions_parafermion(max_k):
    results = []
    for k in range(1, max_k + 1):
        c = Fraction(2 * (k - 1), k + 2)
        dims = set()
        dims.add(Fraction(0))
        for r in range(0, k + 1):
            for s in range(-r + 2, r + 1, 2):  # s = -r+2, -r+4, ..., r
                h = Fraction(r * (r + 2), 4 * (k + 2)) - Fraction(s**2, 4 * k)
                dims.add(h)
        dims = sorted(dims)
        results.append({
            'level': k,
            'central_charge': f'parafermion,{c}',
            'conformal_dimensions': dims[0:10]  # first 15 dimensions
        })
    return results

# Compute for k up to 17
levels_data_parafermion = conformal_dimensions_parafermion(15)[1:]


def conformal_dimensions_N1_minimal_models(max_k):
    results = []
    for k in range(2, max_k + 1):
        c = Fraction(3, 2) - Fraction(12, k * (k + 2))
        dims = set()
        dims.add(Fraction(0))
        for r in range(1, k):  # 1 ≤ r ≤ k-1
            for s in range(1, k + 2):  # 1 ≤ s ≤ k+1
                delta = Fraction(((k + 2) * r - k * s) ** 2 - 4, 8 * k * (k + 2))
                correction = Fraction(1, 32) * (1 - (-1) ** (r + s))
                h = delta + correction
                dims.add(h)
        dims = sorted(dims)
        results.append({
            'level': k,
            'central_charge': f'N1minimal,{c}',
            'conformal_dimensions': dims[0:10]  # first 15 dimensions
        })
    return results

# Compute for k up to 16
levels_data_N1_minimal = conformal_dimensions_N1_minimal_models(17)[1:]

def conformal_dimensions_N2_minimal_models(max_P):
    results = []
    for P in range(1, max_P + 1):
        c = Fraction(3 * P, P + 2)
        dims = set()
        dims.add(Fraction(0))
        for l in range(0, P + 1):  # l = 0, 1, ..., P
            for m in range(-l, l + 1, 2):  # m = -l, -l+2, ..., l
                h = Fraction(l * (l + 2) - m**2, 4 * (P + 2))
                dims.add(h)
        dims = sorted(dims)
        results.append({
            'level': P,
            'central_charge': f'N2minimal,{c}',
            'conformal_dimensions': dims[0:10]  # first 15 dimensions
        })
    return results

# Compute for P up to 17
levels_data_N2_minimal = conformal_dimensions_N2_minimal_models(15)


# Step 1: enter CFT spectrum here
#combined_list = levels_data_minimal + levels_data_su2 + levels_data_su3 + levels_data_sp4 + levels_data_sp8 + levels_data_so10 + levels_data_e8
suN_family = levels_data_su2[:5] + levels_data_su3[:3] + levels_data_su4[:1] + levels_data_su5[:1] + levels_data_su6[:1]
soN_family = levels_data_so7[:1] + levels_data_so8[:2] + levels_data_so9[:2] + levels_data_so10[:1] + levels_data_so11[:1] + levels_data_so12[:1]
spN_family = levels_data_sp4[:2] + levels_data_sp6[:1] + levels_data_sp8[:1]
exp_family = levels_data_e6[:2] + levels_data_e7[:2] + levels_data_e8[:3] + levels_data_f4[:2] + levels_data_g2[:3]
coset_family = levels_data_minimal[:2] + levels_data_N1_minimal[:2] + levels_data_N2_minimal[:3] + levels_data_parafermion[:2]
#u1_family = levels_data_u1_unique[:3] + levels_data_u1_unique[5:6]

combined_list = suN_family + soN_family + spN_family + exp_family + coset_family

print("all theories:",len(combined_list))  # check the number of individual CFTs

# Step 2.1: theories with identical central charge are deleted
seen_charges = set()
filtered_list = []

for entry in combined_list:
    if entry['central_charge'] not in seen_charges:
        seen_charges.add(entry['central_charge'])
        filtered_list.append(entry)

# Step 2.2: delete the theories with the same conformal dimension entry to avoid ill-defined mapping
seen_conformal_dimensions = set()
filtered_list_final = []

for entry in filtered_list:
    dims_key = tuple(entry['conformal_dimensions'])  # Make list hashable
    if dims_key not in seen_conformal_dimensions:
        seen_conformal_dimensions.add(dims_key)
        filtered_list_final.append(entry)  # Keep only the first occurrence

# Step 3: Sort by original level (optional, just for ordering)
filtered_list_final.sort(key=lambda x: x['level'])

# Step 4: Recount levels starting from 1
for new_level, entry in enumerate(filtered_list_final, start=1):
    entry['level'] = new_level

print("identical theories:",len(filtered_list_final))

# hyperparameters
combination_size = 8   # number of individual theories / length of k list
max_combinations = 100000000 - 1   # number of tensor product theories in each length of k
#(obtained 1 million data around 3 minutes)

# Loop the combination_size   21 minutes for 3.7M data  16 minutes for 2.8M data
import random
level_data = filtered_list_final
formatted_rows = []

all_levels = [level['level'] for level in level_data]

for combination_size in range(2, 8):  # Iterate over combination sizes 2 to n
    counter = 1  # Reset counter for this combination_size
    # shuffle the level_data for each combination_size
    #rng = random.Random(0)
    #rng.shuffle(all_levels)

    for levels_combination in combinations_with_replacement(all_levels, combination_size):
        #print(levels_combination)
        if counter > max_combinations:
            break  # Exit this inner loop and continue with next combination_size
        combined_row = tensor_combine_n(level_data, levels_combination)

        dimensions_str = ' '.join(fraction_str(d) for d in combined_row['conformal_dimensions'])
        #charges_str = ' '.join(fraction_str(c) for c in combined_row['central_charges'])
        charges_str = ' '.join(c for c in combined_row['central_charges'])

        formatted_rows.append(f"{counter}|{dimensions_str}\t{charges_str}")
        counter += 1
print(len(formatted_rows))

# Delete identical theories
data = formatted_rows

seen_conformal = set()
unique_lines = []

for line in data:
    # Split by "|" to separate the parts.
    parts = line.split('|')
    # We assume the conformal dimensions are in the second field, and they end at the tab character.
    if len(parts) >= 2:
        conformal_dims = parts[1].split('\t')[0].strip()
        if conformal_dims not in seen_conformal:
            seen_conformal.add(conformal_dims)
            unique_lines.append(line)
print(len(unique_lines))

df_list = pd.DataFrame(unique_lines)
length = int(str(len(df_list))[0])
print(len(df_list))
df_list.to_csv('cft_all_l2-7.csv', index=False)

# with label
from fractions import Fraction

data = unique_lines     # whatever list/iterator you already have

result_list = []
for line in data:
    # 1. split off the leading index
    index, rest = line.split('|', 1)

    # 2. separate the conformal-weights part from the central-charge part
    conformal, central = rest.split('\t')

    # 3. NEW: extract just the fraction after the comma in each token
    #    tokens look like "su2,1/1" or "e8,31/2", etc.
    fractions = []
    for token in central.split():
        # tolerate both “label,fraction” and bare “fraction”
        parts = token.split(',', 1)
        frac_str = parts[-1]          # last piece is always the fraction
        fractions.append(Fraction(frac_str))

    central_sum = sum(fractions)

    # 4. re-assemble the output row (unchanged formatting)
    new_row = f"{index}|{conformal}\t{central}|{central_sum}"
    result_list.append(new_row)

# Filter lines where the total central charge (last field) is <= c_max
c_max = 22
filtered_lines = [line for line in result_list if Fraction(line.split('|')[-1]) < c_max]

# Remove the last field and the preceding pipe by splitting from the right
processed_lines = [line.rsplit('|', 1)[0] for line in filtered_lines]

len(processed_lines)

df_list = pd.DataFrame(processed_lines)
length = int(str(len(df_list))[0])
print(len(df_list))
df_list.to_csv('cft_lib_all_l2-7_c22.csv', index=False)
