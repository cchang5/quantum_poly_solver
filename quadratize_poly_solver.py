import pandas as pd

pd.options.display.max_rows = 999
import numpy as np
import poly_brute_force as poly


def constraint_hamiltonian(qubo, num_problem_qubits, num_qubo_qubits):
    # construct constraint equations
    # auxiliary qubit a_ij = b_i b_j is enforced by
    # b_i b_j - 2 b_i a_ij - 2 b_j a_ij + 3 a_ij
    coeff_scale = 10000
    coeff_bb = coeff_scale * 1
    coeff_ba = coeff_scale * -2
    coeff_aa = coeff_scale * 3

    if coeff_bb + 2. * coeff_ba + coeff_aa == 0:
        pass
    else:
        print("constraint equation poorly defined")
        import sys
        sys.exit()

    # constrain b_i b_j
    for index_j in range(num_problem_qubits):
        for index_i in range(index_j):
            qubo[index_i, index_j] = coeff_bb

    # constrain -2 b_i a_ij -2 b_j a_ij
    qubo_to_aux_index = dict()  # maps auxiliary qubit indices (i,j) to qubo matrix index
    accumulate = 0
    row_counter = 0
    triangle_counter = num_problem_qubits - 1
    for index_j in range(num_problem_qubits, num_qubo_qubits):
        qubo[row_counter, index_j] = coeff_ba
        accumulate += 1
        qubo_to_aux_index[(row_counter, accumulate + row_counter)] = index_j
        if accumulate == triangle_counter:
            accumulate = 0
            row_counter += 1
            triangle_counter -= 1
    accumulate = 0
    row_counter = 1
    triangle_counter = num_problem_qubits - 1
    for index_row in range(num_problem_qubits):
        for index_ij in range(triangle_counter):
            index_i = row_counter + index_ij
            index_j = num_problem_qubits + index_ij + accumulate
            qubo[index_i, index_j] = coeff_ba
        accumulate += index_ij + 1
        row_counter += 1
        triangle_counter -= 1

    # constrain 3 a_ij
    for index_ij in range(num_problem_qubits, num_qubo_qubits):
        qubo[index_ij, index_ij] = coeff_aa

    print("constraint hamiltonian")
    print(pd.DataFrame(qubo))

    return qubo, qubo_to_aux_index


def quadratize(reduced_qubo):
    # quadratizes up to 4-body interactions
    # reduction by substitution (Rosenberg 1975)
    # quadratization in discrete optimization and quantum mechanics
    # section V. A.
    # Nike Dattani arXiv: 1901.04405

    # initialize quadratized hamiltonian
    num_problem_qubits = len(reduced_qubo['qubit_residual_dim1'])
    num_auxiliary_qubits = int(num_problem_qubits * (num_problem_qubits - 1) / 2)
    num_qubo_qubits = num_problem_qubits + num_auxiliary_qubits
    quad_qubo = np.zeros((num_qubo_qubits, num_qubo_qubits), float)

    # construct constraint Hamiltonian
    quad_qubo, qubo_to_aux_index = constraint_hamiltonian(quad_qubo, num_problem_qubits, num_qubo_qubits)

    # load reduced qubo into quadratized qubo
    # dim 0
    qubo_constant = reduced_qubo['qubit_residual_dim0'].copy()
    # check if all non-zero entries are remapped
    reduced_qubo['qubit_residual_dim0'] = np.array(0)

    # dim 1
    for index_ij in range(num_problem_qubits):
        quad_qubo[index_ij, index_ij] += reduced_qubo['qubit_residual_dim1'][index_ij]
        # check if all non-zero entries are remapped
        reduced_qubo['qubit_residual_dim1'][index_ij] = 0

    # dim 2
    for index_j in range(num_problem_qubits):
        for index_i in range(index_j):
            quad_qubo[index_i, index_j] += reduced_qubo['qubit_residual_dim2'][index_i, index_j]
            # check if all non-zero entries are remapped
            reduced_qubo['qubit_residual_dim2'][index_i, index_j] = 0

    # dim 3
    for index_k in range(num_problem_qubits):
        for index_j in range(index_k):
            for index_i in range(index_j):
                row_index = index_i
                col_index = qubo_to_aux_index[(index_j, index_k)]
                quad_qubo[row_index, col_index] += reduced_qubo['qubit_residual_dim3'][index_i, index_j, index_k]
                # check if all non-zero entries are remapped
                reduced_qubo['qubit_residual_dim3'][index_i, index_j, index_k] = 0

    # dim 4
    for index_l in range(num_problem_qubits):
        for index_k in range(index_l):
            for index_j in range(index_k):
                for index_i in range(index_j):
                    row_index = qubo_to_aux_index[(index_i, index_j)]
                    col_index = qubo_to_aux_index[(index_k, index_l)]
                    quad_qubo[row_index, col_index] += reduced_qubo['qubit_residual_dim4'][
                        index_i, index_j, index_k, index_l]
                    # check if all non-zero entries are remapped
                    reduced_qubo['qubit_residual_dim4'][index_i, index_j, index_k, index_l] = 0

    # check
    print("quadratized hamiltonian")
    print(pd.DataFrame(quad_qubo))
    print("qubo constant: ", qubo_constant)

    check = sum([sum(reduced_qubo[key].flatten()) for key in reduced_qubo])
    if check == 0:
        print("check if all non-zero entires are remapped:", True)
    else:
        print("check if all non-zero entires are remapped:", False)

    return quad_qubo, qubo_constant, qubo_to_aux_index


def argmin_QUBO(qubo, qubo_constant):
    # this is for an actual quadratic qubo (yes yes qubo = quadratic binary blah blah...)
    num_of_qubits = len(qubo)
    ground_state_eigenvector = poly.int_to_bin(hilbert_index=0, num_of_qubits=num_of_qubits)
    ground_state_eigenvalue = np.einsum('i,ij,j', ground_state_eigenvector.T, qubo,
                                        ground_state_eigenvector) + qubo_constant
    result_eigenvalue = []
    result_eigenvector = []
    for h_idx in range(2 ** num_of_qubits):  # loop over all 2^n possibilities
        eigenvector = poly.int_to_bin(h_idx, num_of_qubits)
        eigenvalue = np.einsum('i,ij,j', eigenvector.T, qubo, eigenvector) + qubo_constant
        result_eigenvalue.append(eigenvalue)
        result_eigenvector.append(eigenvector)
        if eigenvalue < ground_state_eigenvalue:
            ground_state_eigenvalue = eigenvalue
            ground_state_eigenvector = eigenvector
    return ground_state_eigenvector, ground_state_eigenvalue, result_eigenvalue, result_eigenvector


def quadratized_inverse_mapping(eigenvector, eigenvalue, basis_map, qubo_to_aux_index):
    num_problem_qubits = len(basis_map['basis']) * len(basis_map['basis_offset'])

    # reconstruct result
    result = []
    problem_eigenvector = eigenvector[:num_problem_qubits]
    num_equations = len(basis_map['basis_coeff'])
    qubits_per_var = len(basis_map['basis'])
    for idx_params in range(num_equations):
        result.append(
            basis_map['basis_coeff'][idx_params]
            * sum(
                basis_map['basis'] * problem_eigenvector[
                                     idx_params * qubits_per_var:idx_params * qubits_per_var + qubits_per_var])
            + basis_map['basis_offset'][idx_params])

    print("result:", result)
    print("squared residual:", eigenvalue)

    # check constraint equations
    print("check constraints")
    for qubit_idx in range(num_problem_qubits):
        for pair_idx in range(qubit_idx):
            print("x_%s%s == x_%s x_%s: " % (pair_idx, qubit_idx, pair_idx, qubit_idx),
                  eigenvector[qubit_idx] * eigenvector[pair_idx] == eigenvector[
                      qubo_to_aux_index[(pair_idx, qubit_idx)]])

    return result


def main(evaluate=False):
    extended_qubo, triangle_qubo, reduced_qubo, basis_map = poly.import_QUBO()
    quadratized_qubo, qubo_constant, qubo_to_aux_index = quadratize(reduced_qubo)
    if evaluate:
        ground_state_eigenvector, ground_state_eigenvalue, result_eigenvalue, result_eigenvector = argmin_QUBO(
            quadratized_qubo, qubo_constant)
        quadratized_inverse_mapping(ground_state_eigenvector, ground_state_eigenvalue, basis_map, qubo_to_aux_index)
    return quadratized_qubo, qubo_constant, basis_map, qubo_to_aux_index


if __name__ == "__main__":
    main(True)
