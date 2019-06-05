#!/usr/bin/env python
# Input a QUBO instance and solve using brute force

import numpy as np


# define problem
def define_problem():
    # system of equations
    num_equations = 2

    P0 = np.zeros(num_equations)
    P0[0] = -51.
    P0[1] = -46.

    P1 = np.zeros((num_equations, num_equations))
    P1[0, 0] = 2
    P1[0, 1] = 4
    P1[1, 0] = 3
    P1[1, 1] = 2

    P2 = np.zeros((num_equations, num_equations, num_equations))
    P2[0, 0, 0] = 2
    P2[0, 0, 1] = 3
    P2[0, 1, 0] = 0
    P2[0, 1, 1] = 1

    P2[1, 0, 0] = 1
    P2[1, 0, 1] = 2
    P2[1, 1, 0] = 0
    P2[1, 1, 1] = 2

    # search parameters
    qubits_per_var = 2
    basis = np.array([2 ** i for i in range(qubits_per_var)])

    basis_offset = np.array([1.5, 2])
    basis_coeff = np.array([0.5, 1])

    basis_map = {'basis': basis, 'basis_offset': basis_offset, 'basis_coeff': basis_coeff}

    return num_equations, P0, P1, P2, qubits_per_var, basis, basis_offset, basis_coeff, basis_map


def calculate_squared_residuals(P0, P1, P2):
    residual = dict()
    # x labels the states and o labels the operator
    # number of x's corresponds to the rank of the tensor since the state has not been contracted yet
    residual['dim0_o'] = np.einsum('i,i', P0, P0)

    residual['dim1_ox'] = np.einsum('i,ij->j', P0.T, P1)
    residual['dim1_xo'] = np.einsum('ji,i->j', P1.T, P0)

    residual['dim2_oxx'] = np.einsum('i,ijk->jk', P0.T, P2)
    residual['dim2_xox'] = np.einsum('ji,ik->jk', P1.T, P1)
    residual['dim2_xxo'] = np.einsum('kji,i->kj', P2.T, P0)

    residual['dim3_xoxx'] = np.einsum('ji,ikl->jkl', P1.T, P2)
    residual['dim3_xxox'] = np.einsum('kji,il->kjl', P2.T, P1)

    residual['dim4_xxoxx'] = np.einsum('kji,inm->kjnm', P2.T, P2)

    return residual


def calculate_residual_offsets(P0, P1, P2, basis_offset):
    residual_offset = dict()
    ### calculate QUBO offsets
    # x labels the states, o labels the operator, b labels the offset
    # D1
    residual_dim1_ob = np.einsum('i,ij,j', P0.T, P1, basis_offset)

    residual_dim1_bo = np.einsum('j,ji,i', basis_offset.T, P1.T, P0)

    residual_offset['dim1_o'] = residual_dim1_ob + residual_dim1_bo

    # D2
    residual_dim2_obx = np.einsum('i,ijk,j', P0.T, P2, basis_offset)
    residual_dim2_oxb = np.einsum('i,ijk,k', P0.T, P2, basis_offset)
    residual_dim2_obb = np.einsum('i,ijk,j,k', P0.T, P2, basis_offset, basis_offset)

    residual_dim2_box = np.einsum('j,ji,ik', basis_offset.T, P1.T, P1)
    residual_dim2_xob = np.einsum('ji,ik,k', P1.T, P1, basis_offset)
    residual_dim2_bob = np.einsum('j,ji,ik,k', basis_offset.T, P1.T, P1, basis_offset)

    residual_dim2_bxo = np.einsum('k,kji,i', basis_offset.T, P2.T, P0)
    residual_dim2_xbo = np.einsum('j,kji,i', basis_offset.T, P2.T, P0)
    residual_dim2_bbo = np.einsum('k,j,kji,i', basis_offset.T, basis_offset.T, P2.T, P0)

    residual_offset['dim2_ox'] = residual_dim2_obx + residual_dim2_oxb + residual_dim2_box
    residual_offset['dim2_xo'] = residual_dim2_xob + residual_dim2_bxo + residual_dim2_xbo
    residual_offset['dim2_o'] = residual_dim2_obb + residual_dim2_bob + residual_dim2_bbo

    # D3
    residual_dim3_xoxb = np.einsum('ji,ikl,l', P1.T, P2, basis_offset)
    residual_dim3_xobx = np.einsum('ji,ikl,k', P1.T, P2, basis_offset)
    residual_dim3_boxx = np.einsum('j,ji,ikl', basis_offset.T, P1.T, P2)
    residual_dim3_xobb = np.einsum('ji,ikl,k,l', P1.T, P2, basis_offset, basis_offset)
    residual_dim3_boxb = np.einsum('j,ji,ikl,l', basis_offset.T, P1.T, P2, basis_offset)
    residual_dim3_bobx = np.einsum('j,ji,ikl,k', basis_offset.T, P1.T, P2, basis_offset)
    residual_dim3_bobb = np.einsum('j,ji,ikl,k,l', basis_offset.T, P1.T, P2, basis_offset, basis_offset)

    residual_dim3_xxob = np.einsum('kji,il,l', P2.T, P1, basis_offset)
    residual_dim3_xbox = np.einsum('j,kji,il', basis_offset.T, P2.T, P1)
    residual_dim3_bxox = np.einsum('k,kji,il', basis_offset.T, P2.T, P1)
    residual_dim3_xbob = np.einsum('j,kji,il,l', basis_offset.T, P2.T, P1, basis_offset)
    residual_dim3_bxob = np.einsum('k,kji,il,l', basis_offset.T, P2.T, P1, basis_offset)
    residual_dim3_bbox = np.einsum('k,j,kji,il', basis_offset.T, basis_offset.T, P2.T, P1)
    residual_dim3_bbob = np.einsum('k,j,kji,il,l', basis_offset.T, basis_offset.T, P2.T, P1, basis_offset)

    residual_offset['dim3_oxx'] = residual_dim3_boxx
    residual_offset['dim3_xox'] = residual_dim3_xoxb + residual_dim3_xobx + residual_dim3_xbox + residual_dim3_bxox
    residual_offset['dim3_xxo'] = residual_dim3_xxob
    residual_offset['dim3_ox'] = residual_dim3_boxb + residual_dim3_bobx + residual_dim3_bbox
    residual_offset['dim3_xo'] = residual_dim3_xobb + residual_dim3_xbob + residual_dim3_bxob
    residual_offset['dim3_o'] = residual_dim3_bobb + residual_dim3_bbob

    # D4
    residual_dim4_xxoxb = np.einsum('kji,inm,m', P2.T, P2, basis_offset)
    residual_dim4_xxobx = np.einsum('kji,inm,n', P2.T, P2, basis_offset)
    residual_dim4_xxobb = np.einsum('kji,inm,n,m', P2.T, P2, basis_offset, basis_offset)
    residual_dim4_xboxx = np.einsum('j,kji,inm', basis_offset.T, P2.T, P2)
    residual_dim4_xboxb = np.einsum('j,kji,inm,m', basis_offset.T, P2.T, P2, basis_offset)
    residual_dim4_xbobx = np.einsum('j,kji,inm,n', basis_offset.T, P2.T, P2, basis_offset)
    residual_dim4_xbobb = np.einsum('j,kji,inm,n,m', basis_offset.T, P2.T, P2, basis_offset, basis_offset)
    residual_dim4_bxoxx = np.einsum('k,kji,inm', basis_offset.T, P2.T, P2)
    residual_dim4_bxoxb = np.einsum('k,kji,inm,m', basis_offset.T, P2.T, P2, basis_offset)
    residual_dim4_bxobx = np.einsum('k,kji,inm,n', basis_offset.T, P2.T, P2, basis_offset)
    residual_dim4_bxobb = np.einsum('k,kji,inm,n,m', basis_offset.T, P2.T, P2, basis_offset, basis_offset)
    residual_dim4_bboxx = np.einsum('k,j,kji,inm', basis_offset.T, basis_offset.T, P2.T, P2)
    residual_dim4_bboxb = np.einsum('k,j,kji,inm,m', basis_offset.T, basis_offset.T, P2.T, P2, basis_offset)
    residual_dim4_bbobx = np.einsum('k,j,kji,inm,n', basis_offset.T, basis_offset.T, P2.T, P2, basis_offset)
    residual_dim4_bbobb = np.einsum('k,j,kji,inm,n,m', basis_offset.T, basis_offset.T, P2.T, P2, basis_offset,
                                    basis_offset)

    residual_offset['dim4_xoxx'] = residual_dim4_xboxx + residual_dim4_bxoxx
    residual_offset['dim4_xxox'] = residual_dim4_xxobx + residual_dim4_xxoxb
    residual_offset['dim4_oxx'] = residual_dim4_bboxx
    residual_offset['dim4_xox'] = residual_dim4_xboxb + residual_dim4_xbobx + residual_dim4_bxoxb + residual_dim4_bxobx
    residual_offset['dim4_xxo'] = residual_dim4_xxobb
    residual_offset['dim4_ox'] = residual_dim4_bboxb + residual_dim4_bbobx
    residual_offset['dim4_xo'] = residual_dim4_xbobb + residual_dim4_bxobb
    residual_offset['dim4_o'] = residual_dim4_bbobb

    return residual_offset


def combine_residual_offset(residual, residual_offset):
    full_residual = dict()
    # dim 0
    offset_residual_dim0_o = residual['dim0_o'] + residual_offset['dim1_o'] + residual_offset['dim2_o'] + \
                             residual_offset['dim3_o'] + residual_offset['dim4_o']
    full_residual['dim0'] = offset_residual_dim0_o

    # dim1
    offset_residual_dim1_ox = residual['dim1_ox'] + residual_offset['dim2_ox'] + \
                              residual_offset['dim3_ox'] + residual_offset['dim4_ox']
    offset_residual_dim1_xo = residual['dim1_xo'] + residual_offset['dim2_xo'] + \
                              residual_offset['dim3_xo'] + residual_offset['dim4_xo']
    full_residual['dim1'] = offset_residual_dim1_ox + offset_residual_dim1_xo

    # dim 2
    offset_residual_dim2_oxx = residual['dim2_oxx'] + residual_offset['dim3_oxx'] + residual_offset['dim4_oxx']
    offset_residual_dim2_xox = residual['dim2_xox'] + residual_offset['dim3_xox'] + residual_offset['dim4_xox']
    offset_residual_dim2_xxo = residual['dim2_xxo'] + residual_offset['dim3_xxo'] + residual_offset['dim4_xxo']
    full_residual['dim2'] = offset_residual_dim2_oxx + offset_residual_dim2_xox + offset_residual_dim2_xxo

    # dim 3
    offset_residual_dim3_xoxx = residual['dim3_xoxx'] + residual_offset['dim4_xoxx']
    offset_residual_dim3_xxox = residual['dim3_xxox'] + residual_offset['dim4_xxox']
    full_residual['dim3'] = offset_residual_dim3_xoxx + offset_residual_dim3_xxox

    # dim 4
    offset_residual_dim4_xxoxx = residual['dim4_xxoxx']
    full_residual['dim4'] = offset_residual_dim4_xxoxx

    return full_residual


def real_to_qubit_basis(full_residual, num_equations, qubits_per_var, basis, basis_coeff):
    extended_qubo = dict()

    # dimension 0
    extended_qubo['qubit_residual_dim0'] = full_residual['dim0']

    # dimension 1
    extended_qubo['qubit_residual_dim1'] = np.reshape(
        np.einsum('i,j->ij', basis_coeff * full_residual['dim1'], basis), (num_equations * qubits_per_var))

    # dimension 2
    basis_coeff_dim2 = np.einsum('i,j->ij', basis_coeff, basis_coeff)
    basis_dim2 = np.einsum('i,j->ij', basis, basis)

    extended_qubo['qubit_residual_dim2'] = np.reshape(
        np.einsum('ij,kl->ikjl', basis_coeff_dim2 * full_residual['dim2'], basis_dim2),
        (num_equations * qubits_per_var, num_equations * qubits_per_var))

    # dimension 3
    basis_coeff_dim3 = np.einsum('i,j,k->ijk', basis_coeff, basis_coeff, basis_coeff)
    basis_dim3 = np.einsum('i,j,k->ijk', basis, basis, basis)

    extended_qubo['qubit_residual_dim3'] = np.reshape(
        np.einsum('ijk,lmn->iljmkn', basis_coeff_dim3 * full_residual['dim3'], basis_dim3),
        (num_equations * qubits_per_var, num_equations * qubits_per_var, num_equations * qubits_per_var))

    # dimension 4
    basis_coeff_dim4 = np.einsum('i,j,k,l->ijkl', basis_coeff, basis_coeff, basis_coeff, basis_coeff)
    basis_dim4 = np.einsum('i,j,k,l->ijkl', basis, basis, basis, basis)

    extended_qubo['qubit_residual_dim4'] = np.reshape(
        np.einsum('ijkl,mnop->imjnkolp', basis_coeff_dim4 * full_residual['dim4'], basis_dim4),
        (num_equations * qubits_per_var, num_equations * qubits_per_var, num_equations * qubits_per_var,
         num_equations * qubits_per_var))

    return extended_qubo


def accumulate_qubo(extended_qubo):
    triangle_qubo = dict()
    triangle_qubo['qubit_residual_dim0'] = extended_qubo['qubit_residual_dim0'].copy()
    triangle_qubo['qubit_residual_dim1'] = extended_qubo['qubit_residual_dim1'].copy()
    # dim 2
    accumulate_dim2 = np.zeros_like(extended_qubo['qubit_residual_dim2'])
    for index_j in range(len(accumulate_dim2)):
        for index_i in range(len(accumulate_dim2)):
            sorted_index = np.sort([index_i, index_j])
            row_index = sorted_index[0]
            col_index = sorted_index[1]
            accumulate_dim2[row_index, col_index] += extended_qubo['qubit_residual_dim2'][index_i, index_j]
    triangle_qubo['qubit_residual_dim2'] = accumulate_dim2
    # dim 3
    accumulate_dim3 = np.zeros_like(extended_qubo['qubit_residual_dim3'])
    for index_k in range(len(accumulate_dim3)):
        for index_j in range(len(accumulate_dim3)):
            for index_i in range(len(accumulate_dim3)):
                sorted_index = np.sort([index_i, index_j, index_k])
                accumulate_dim3[sorted_index[0], sorted_index[1], sorted_index[2]] += \
                    extended_qubo['qubit_residual_dim3'][index_i, index_j, index_k]
    triangle_qubo['qubit_residual_dim3'] = accumulate_dim3
    # dim 4
    accumulate_dim4 = np.zeros_like(extended_qubo['qubit_residual_dim4'])
    for index_l in range(len(accumulate_dim4)):
        for index_k in range(len(accumulate_dim4)):
            for index_j in range(len(accumulate_dim4)):
                for index_i in range(len(accumulate_dim4)):
                    sorted_index = np.sort([index_i, index_j, index_k, index_l])
                    accumulate_dim4[sorted_index[0], sorted_index[1], sorted_index[2], sorted_index[3]] += \
                        extended_qubo['qubit_residual_dim4'][index_i, index_j, index_k, index_l]
    triangle_qubo['qubit_residual_dim4'] = accumulate_dim4

    return triangle_qubo


def dimensional_reduction(triangle_qubo):
    from sympy.utilities.iterables import multiset_permutations
    # takes upper triangular qubo and reduces the dimensionality of repeated qubits
    # e.g. x_i^n = x_i since x_i in [0, 1]
    reduced_qubo = dict()

    # dim 0
    reduced_qubo['qubit_residual_dim0'] = triangle_qubo['qubit_residual_dim0'].copy()

    # dim 1
    reduced_qubo['qubit_residual_dim1'] = np.zeros_like(triangle_qubo['qubit_residual_dim1'])
    for idx in range(len(reduced_qubo['qubit_residual_dim1'])):
        # dim 1
        reduced_qubo['qubit_residual_dim1'][idx] += triangle_qubo['qubit_residual_dim1'][idx]
        # dim 2
        reduced_qubo['qubit_residual_dim1'][idx] += triangle_qubo['qubit_residual_dim2'][idx, idx]
        # dim 3
        reduced_qubo['qubit_residual_dim1'][idx] += triangle_qubo['qubit_residual_dim3'][idx, idx, idx]
        # dim 4
        reduced_qubo['qubit_residual_dim1'][idx] += triangle_qubo['qubit_residual_dim4'][idx, idx, idx, idx]

    # dim 2
    reduced_qubo['qubit_residual_dim2'] = np.zeros_like(triangle_qubo['qubit_residual_dim2'])
    for idx_j in range(len(reduced_qubo['qubit_residual_dim2'])):
        for idx_i in range(idx_j):
            # dim 2
            reduced_qubo['qubit_residual_dim2'][idx_i, idx_j] += triangle_qubo['qubit_residual_dim2'][idx_i, idx_j]
            # dim 3
            reduced_qubo['qubit_residual_dim2'][idx_i, idx_j] += triangle_qubo['qubit_residual_dim3'][
                idx_i, idx_j, idx_j]
            reduced_qubo['qubit_residual_dim2'][idx_i, idx_j] += triangle_qubo['qubit_residual_dim3'][
                idx_i, idx_i, idx_j]
            # dim 4
            reduced_qubo['qubit_residual_dim2'][idx_i, idx_j] += triangle_qubo['qubit_residual_dim4'][
                idx_i, idx_j, idx_j, idx_j]
            reduced_qubo['qubit_residual_dim2'][idx_i, idx_j] += triangle_qubo['qubit_residual_dim4'][
                idx_i, idx_i, idx_j, idx_j]
            reduced_qubo['qubit_residual_dim2'][idx_i, idx_j] += triangle_qubo['qubit_residual_dim4'][
                idx_i, idx_i, idx_i, idx_j]

    # dim 3
    reduced_qubo['qubit_residual_dim3'] = np.zeros_like(triangle_qubo['qubit_residual_dim3'])
    for idx_k in range(len(reduced_qubo['qubit_residual_dim3'])):
        for idx_j in range(idx_k):
            for idx_i in range(idx_j):
                # dim 3
                reduced_qubo['qubit_residual_dim3'][idx_i, idx_j, idx_k] += triangle_qubo['qubit_residual_dim3'][
                    idx_i, idx_j, idx_k]
                # dim 4
                reduced_qubo['qubit_residual_dim3'][idx_i, idx_j, idx_k] += triangle_qubo['qubit_residual_dim4'][
                    idx_i, idx_i, idx_j, idx_k]
                reduced_qubo['qubit_residual_dim3'][idx_i, idx_j, idx_k] += triangle_qubo['qubit_residual_dim4'][
                    idx_i, idx_j, idx_j, idx_k]
                reduced_qubo['qubit_residual_dim3'][idx_i, idx_j, idx_k] += triangle_qubo['qubit_residual_dim4'][
                    idx_i, idx_j, idx_k, idx_k]

    # dim 4
    reduced_qubo['qubit_residual_dim4'] = np.zeros_like(triangle_qubo['qubit_residual_dim4'])
    for idx_l in range(len(reduced_qubo['qubit_residual_dim4'])):
        for idx_k in range(idx_l):
            for idx_j in range(idx_k):
                for idx_i in range(idx_j):
                    reduced_qubo['qubit_residual_dim4'][idx_i, idx_j, idx_k, idx_l] += \
                        triangle_qubo['qubit_residual_dim4'][idx_i, idx_j, idx_k, idx_l]

    return reduced_qubo


# import the QUBO data and return numpy 2D square array
def import_QUBO():
    ### define problem
    num_equations, P0, P1, P2, qubits_per_var, basis, basis_offset, basis_coeff, basis_map = define_problem()
    ### calculate "qubo" in real number basis
    residual = calculate_squared_residuals(P0, P1, P2)
    residual_offset = calculate_residual_offsets(P0, P1, P2, basis_offset)
    full_residual = combine_residual_offset(residual, residual_offset)
    ### transform "qubo" to qubit basis
    # this can be sent into eval_QUBO() and solved
    extended_qubo = real_to_qubit_basis(full_residual, num_equations, qubits_per_var, basis, basis_coeff)
    ### accumulate extended qubo, to construct only upper triangular tensors
    triangle_qubo = accumulate_qubo(extended_qubo)
    ### make most sparse upper triangular tensors by reducing repeated qubits
    reduced_qubo = dimensional_reduction(triangle_qubo)

    return extended_qubo, triangle_qubo, reduced_qubo, basis_map


# evaluate the QUBO given binary vector "eigenvector" and return energy "eigenvalue"
def eval_QUBO(extended_qubo, eigenvector):
    eigenvalue_dim0 = extended_qubo['qubit_residual_dim0']
    eigenvalue_dim1 = np.einsum('j,j', extended_qubo['qubit_residual_dim1'], eigenvector)
    eigenvalue_dim2 = np.einsum('j,jk,k', eigenvector.T, extended_qubo['qubit_residual_dim2'], eigenvector)
    eigenvalue_dim3 = np.einsum('j,jkl,k,l', eigenvector.T, extended_qubo['qubit_residual_dim3'], eigenvector,
                                eigenvector)
    eigenvalue_dim4 = np.einsum('k,j,kjnm,n,m', eigenvector.T, eigenvector.T, extended_qubo['qubit_residual_dim4'],
                                eigenvector, eigenvector)

    eigenvalue = eigenvalue_dim0 + eigenvalue_dim1 + eigenvalue_dim2 + eigenvalue_dim3 + eigenvalue_dim4
    return eigenvalue


# Convert non-negative n-bit integer to n-bit binary representation and return numpy array
def int_to_bin(hilbert_index, num_of_qubits):
    length = int(hilbert_index).bit_length();  # length of binary conversion
    # Check that the bit length fits the b-bit representation
    if length > num_of_qubits:
        print(" <<Bit length exceeds repreesntation size>>")
        raise ValueError
    x = bin(int(hilbert_index));  # binary converstion returns string x
    y = x[2:length + 2]  # store last l chars of x in y
    eigenvector = np.zeros(num_of_qubits);
    for i in range(len(y)):
        # add the bits from smallest to largest in the last l slots
        eigenvector[num_of_qubits - length + i] = int(y[i]);
    return eigenvector


def argmin_QUBO(extended_qubo):
    num_of_qubits = len(extended_qubo['qubit_residual_dim1'])
    ground_state_eigenvector = int_to_bin(hilbert_index=0, num_of_qubits=num_of_qubits)
    ground_state_eigenvalue = eval_QUBO(extended_qubo, ground_state_eigenvector)
    result_eigenvalue = []
    result_eigenvector = []
    for h_idx in range(2 ** num_of_qubits):  # loop over all 2^n possibilities
        eigenvector = int_to_bin(h_idx, num_of_qubits)
        eigenvalue = eval_QUBO(extended_qubo, eigenvector)
        result_eigenvalue.append(eigenvalue)
        result_eigenvector.append(eigenvector)
        if eigenvalue < ground_state_eigenvalue:
            ground_state_eigenvalue = eigenvalue
            ground_state_eigenvector = eigenvector
    return ground_state_eigenvector, result_eigenvalue, result_eigenvector


def inverse_mapping(eigenvector, basis_map):
    presult = []
    num_equations = len(basis_map['basis_coeff'])
    qubits_per_var = len(basis_map['basis'])
    for idx_params in range(num_equations):
        presult.append(
            basis_map['basis_coeff'][idx_params]
            * sum(
                basis_map['basis'] * eigenvector[
                                     idx_params * qubits_per_var:idx_params * qubits_per_var + qubits_per_var])
            + basis_map['basis_offset'][idx_params])
    return presult


def evaluate_problem(qubo, basis_map, title):
    # Get arg min for extended qubo and compute energy
    ground_state_eigenvector, result_eigenvalue, result_eigenvector = argmin_QUBO(qubo)
    ground_state_eigenvalue = eval_QUBO(qubo, ground_state_eigenvector)
    # Evaluate results
    print(title)
    print("ground state eigenvector = ", ground_state_eigenvector)
    print("ground state eigenvalue  = ", ground_state_eigenvalue)
    print("solution                 = ", inverse_mapping(ground_state_eigenvector, basis_map))
    print()


def main():
    # Get QUBO matrix
    extended_qubo, triangle_qubo, reduced_qubo, basis_map = import_QUBO()
    evaluate_problem(extended_qubo, basis_map, 'extended qubo')
    evaluate_problem(triangle_qubo, basis_map, 'upper triangular qubo')
    evaluate_problem(reduced_qubo, basis_map, 'reduced upper triangular qubo')


if __name__ == '__main__':
    main()
