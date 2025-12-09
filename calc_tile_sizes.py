import numpy as np
import os

DATATYPE_SIZE = 4 # bytes
def calc_tile_sizes(pre_tile_analysis_file, cache_sizes):
    with open(pre_tile_analysis_file, 'r') as file:
        lines = file.readlines()
    reuse_values = lines[0].split()
    # convert reuse_values to numpy array
    reuse_values = np.array(reuse_values, dtype=float)
    # split up lines[1:] by \n\n terminators
    memory_footprint_polynomial_terms = [[]]
    changeIdx = False
    for i in range(1, len(lines)):
        if lines[i] == '\n' or lines[i] == '':
            changeIdx = True
            pass
        else:
            if changeIdx:
                memory_footprint_polynomial_terms.append([])
                changeIdx = False
            memory_footprint_polynomial_terms[-1].append(np.array(lines[i].split(), dtype=float))
        print("line " + str(i) + " processed")
    
    if memory_footprint_polynomial_terms[-1] == []:
        memory_footprint_polynomial_terms.pop()
    # convert each entry in memory_footprint_polynomial_terms to 2D numpy array
    new_poly_terms = []
    for term in memory_footprint_polynomial_terms:
        current_matrix = term
        current_matrix = np.array(current_matrix, dtype=float)
        new_poly_terms.append(current_matrix)
    
    memory_footprint_polynomial_terms = new_poly_terms
    
    # each polynomial term is of format [[a1, a2, a3...], [b1, b2, b3...]...] ==> (a1 * x1 + a2 * x2 + a3 * x3 + ...) * (b1 * x1 + b2 * x2 + b3 * x3 + ...) * ...
    # for the xi's, use raw values
    xi_values = reuse_values
    term_degrees = []

    for polynomial_term in memory_footprint_polynomial_terms:
        for row in polynomial_term:
            for i in range(len(row)):
                row[i] = row[i] * xi_values[i]
        # multiply the rows together polynomial style
        poly_term_vector = np.matmul(polynomial_term, np.ones((polynomial_term.shape[1], 1), dtype=float)) # column vector
        poly_term_total_coeff = np.prod(poly_term_vector)
        term_degrees.append([poly_term_total_coeff, polynomial_term.shape[0]])
    
    # sum up term coeffs with same degree
    term_coeffs_by_degree = dict()
    for i in range(len(term_degrees)):
        if term_degrees[i][1] not in term_coeffs_by_degree:
            term_coeffs_by_degree[term_degrees[i][1]] = term_degrees[i][0]
        else:
            term_coeffs_by_degree[term_degrees[i][1]] += term_degrees[i][0]
    
    # form polynomial vector
    polynomial_vector = []
    max_degree = max(term_coeffs_by_degree.keys())
    for i in range(max_degree + 1):
        if i in term_coeffs_by_degree:
            polynomial_vector.append(term_coeffs_by_degree[i])
        else:
            polynomial_vector.append(0)
    # now we have the polynomial vector
    # use it to solve for cache_sizes[1]
    polynomial_vector = np.array(polynomial_vector, dtype=float)
    polynomial_vector[0] = -cache_sizes[0] / DATATYPE_SIZE
    polynomial_vector = polynomial_vector[::-1]
    roots = np.roots(polynomial_vector)
    # take the real part of the roots
    roots = np.real(roots)
    # take the positive roots
    roots = roots[roots > 0]
    # take the smallest root
    tile_size_tau = np.min(roots)

    lvl_0_tile_sizes = tile_size_tau * reuse_values
    lvl_0_tile_sizes = lvl_0_tile_sizes.astype(int)

    with open(os.path.join(os.path.dirname(pre_tile_analysis_file), 'tile_size_env_vars.sh'), 'w') as file:
        file.write('#!/bin/bash\n')
        file.write('export NUM_TILE_LEVELS=' + str(len(cache_sizes)) + '\n')
        file.write('export NUM_BANDS=' + str(len(lvl_0_tile_sizes)) + '\n')
        for i in range(len(lvl_0_tile_sizes)):
            file.write('export TILE_LEVEL_0_' + str(i) + '=' + str(lvl_0_tile_sizes[i]) + '\n')

    # need to calculate the next level tile sizes
    # for each successive level, we want to preserve the reuse factors. so,
    # if each dimension of the previous level's tile volume iterates by the same amount, this is achieved.
    # Hence, (smaller cache size) * (iter amount) ^ (num bands) = (larger cache size)
    # then, the new volume has length = prev level length * (iter amount), etc. We give the tile sizes as iter amt, as tiling will scale this automatically due to strides of previous levels.
    num_bands = len(lvl_0_tile_sizes)
    for i in range(1, len(cache_sizes)):
        iter_amount = (cache_sizes[i] / cache_sizes[i-1]) ** (1 / num_bands)
        # write to file
        with open(os.path.join(os.path.dirname(pre_tile_analysis_file), 'tile_size_env_vars.sh'), 'a') as file:
            for j in range(num_bands):
                file.write('export TILE_LEVEL_' + str(i) + '_' + str(j) + '=' + str(int(iter_amount)) + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_tile_analysis_file', type=str)
    parser.add_argument('--cache_sizes', type=str, help='cache sizes in bytes, comma separated')
    args = parser.parse_args()
    calc_tile_sizes(args.pre_tile_analysis_file, [int(size) for size in args.cache_sizes.split(',')])