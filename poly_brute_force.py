#!/usr/bin/env python
# Input a QUBO instance and solve using brute force

import numpy as np


# import the QUBO data and return numpy 2D square array
def import_QUBO():
    ### define problem
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

    ### calculate squared residual matrices
    # x labels the states and o labels the operator
    residual_dim0_o = np.einsum('i,i', P0, P0)

    residual_dim1_ox = np.einsum('i,ij->j', P0.T, P1)
    residual_dim1_xo = np.einsum('ji,i->j', P1.T, P0)

    residual_dim2_oxx = np.einsum('i,ijk->jk', P0.T, P2)
    residual_dim2_xox = np.einsum('ji,ik->jk', P1.T, P1)
    residual_dim2_xxo = np.einsum('kji,i->kj', P2.T, P0)

    residual_dim3_xoxx = np.einsum('ji,ikl->jkl', P1.T, P2)
    residual_dim3_xxox = np.einsum('kji,il->kjl', P2.T, P1)

    residual_dim4_xxoxx = np.einsum('kji,inm->kjnm', P2.T, P2)

    ### define search parameters
    qubits_per_var = 3
    basis = np.array([2 ** i for i in range(qubits_per_var)])

    basis_offset = np.array([1.5, 0])
    basis_coeff = np.array([0.5, 1])

    basis_map = {'basis': basis, 'basis_offset': basis_offset, 'basis_coeff': basis_coeff}

    ### calculate QUBO offsets
    # x labels the states, o labels the operator, b labels the offset
    # D1
    residual_dim1_ob = np.einsum('i,ij,j', P0.T, P1, basis_offset)

    residual_dim1_bo = np.einsum('j,ji,i', basis_offset.T, P1.T, P0)

    residual_dim1_offset_dim0_o = residual_dim1_ob + residual_dim1_bo

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

    residual_dim2_offset_dim1_ox = residual_dim2_obx + residual_dim2_oxb + residual_dim2_box
    residual_dim2_offset_dim1_xo = residual_dim2_xob + residual_dim2_bxo + residual_dim2_xbo
    residual_dim2_offset_dim0_o = residual_dim2_obb + residual_dim2_bob + residual_dim2_bbo

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

    residual_dim3_offset_dim2_oxx = residual_dim3_boxx
    residual_dim3_offset_dim2_xox = residual_dim3_xoxb + residual_dim3_xobx + residual_dim3_xbox + residual_dim3_bxox
    residual_dim3_offset_dim2_xxo = residual_dim3_xxob
    residual_dim3_offset_dim1_ox = residual_dim3_boxb + residual_dim3_bobx + residual_dim3_bbox
    residual_dim3_offset_dim1_xo = residual_dim3_xobb + residual_dim3_xbob + residual_dim3_bxob
    residual_dim3_offset_dim0_o = residual_dim3_bobb + residual_dim3_bbob

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

    residual_dim4_offset_dim3_xoxx = residual_dim4_xboxx + residual_dim4_bxoxx
    residual_dim4_offset_dim3_xxox = residual_dim4_xxobx + residual_dim4_xxoxb
    residual_dim4_offset_dim2_oxx = residual_dim4_bboxx
    residual_dim4_offset_dim2_xox = residual_dim4_xboxb + residual_dim4_xbobx + residual_dim4_bxoxb + residual_dim4_bxobx
    residual_dim4_offset_dim2_xxo = residual_dim4_xxobb
    residual_dim4_offset_dim1_ox = residual_dim4_bboxb + residual_dim4_bbobx
    residual_dim4_offset_dim1_xo = residual_dim4_xbobb + residual_dim4_bxobb
    residual_dim4_offset_dim0_o = residual_dim4_bbobb

    # combine contributions with same contractions with the state vectors
    offset_residual_dim0_o = residual_dim0_o + residual_dim1_offset_dim0_o + residual_dim2_offset_dim0_o + residual_dim3_offset_dim0_o + residual_dim4_offset_dim0_o

    offset_residual_dim1_ox = residual_dim1_ox + residual_dim2_offset_dim1_ox + residual_dim3_offset_dim1_ox + residual_dim4_offset_dim1_ox
    offset_residual_dim1_xo = residual_dim1_xo + residual_dim2_offset_dim1_xo + residual_dim3_offset_dim1_xo + residual_dim4_offset_dim1_xo

    offset_residual_dim2_oxx = residual_dim2_oxx + residual_dim3_offset_dim2_oxx + residual_dim4_offset_dim2_oxx
    offset_residual_dim2_xox = residual_dim2_xox + residual_dim3_offset_dim2_xox + residual_dim4_offset_dim2_xox
    offset_residual_dim2_xxo = residual_dim2_xxo + residual_dim3_offset_dim2_xxo + residual_dim4_offset_dim2_xxo

    offset_residual_dim3_xoxx = residual_dim3_xoxx + residual_dim4_offset_dim3_xoxx
    offset_residual_dim3_xxox = residual_dim3_xxox + residual_dim4_offset_dim3_xxox

    offset_residual_dim4_xxoxx = residual_dim4_xxoxx

    ### expand to qubit basis
    extended_qubo = dict()

    # dimension 0
    extended_qubo['qubit_residual_dim0_o'] = offset_residual_dim0_o

    # dimension 1
    extended_qubo['qubit_residual_dim1_ox'] = np.reshape(
        np.einsum('i,j->ij', basis_coeff * offset_residual_dim1_ox, basis), (num_equations * qubits_per_var))
    extended_qubo['qubit_residual_dim1_xo'] = np.reshape(
        np.einsum('i,j->ij', basis_coeff * offset_residual_dim1_xo, basis), (num_equations * qubits_per_var))

    # dimension 2
    basis_coeff_dim2 = np.einsum('i,j->ij', basis_coeff, basis_coeff)
    basis_dim2 = np.einsum('i,j->ij', basis, basis)

    extended_qubo['qubit_residual_dim2_oxx'] = np.reshape(
        np.einsum('ij,kl->ikjl', basis_coeff_dim2 * offset_residual_dim2_oxx, basis_dim2),
        (num_equations * qubits_per_var, num_equations * qubits_per_var))
    extended_qubo['qubit_residual_dim2_xox'] = np.reshape(
        np.einsum('ij,kl->ikjl', basis_coeff_dim2 * offset_residual_dim2_xox, basis_dim2),
        (num_equations * qubits_per_var, num_equations * qubits_per_var))
    extended_qubo['qubit_residual_dim2_xxo'] = np.reshape(
        np.einsum('ij,kl->ikjl', basis_coeff_dim2 * offset_residual_dim2_xxo, basis_dim2),
        (num_equations * qubits_per_var, num_equations * qubits_per_var))

    # dimension 3
    basis_coeff_dim3 = np.einsum('i,j,k->ijk', basis_coeff, basis_coeff, basis_coeff)
    basis_dim3 = np.einsum('i,j,k->ijk', basis, basis, basis)

    extended_qubo['qubit_residual_dim3_xoxx'] = np.reshape(
        np.einsum('ijk,lmn->iljmkn', basis_coeff_dim3 * offset_residual_dim3_xoxx, basis_dim3),
        (num_equations * qubits_per_var, num_equations * qubits_per_var, num_equations * qubits_per_var))
    extended_qubo['qubit_residual_dim3_xxox'] = np.reshape(
        np.einsum('ijk,lmn->iljmkn', basis_coeff_dim3 * offset_residual_dim3_xxox, basis_dim3),
        (num_equations * qubits_per_var, num_equations * qubits_per_var, num_equations * qubits_per_var))

    # dimension 4
    basis_coeff_dim4 = np.einsum('i,j,k,l->ijkl', basis_coeff, basis_coeff, basis_coeff, basis_coeff)
    basis_dim4 = np.einsum('i,j,k,l->ijkl', basis, basis, basis, basis)

    extended_qubo['qubit_residual_dim4_xxoxx'] = np.reshape(
        np.einsum('ijkl,mnop->imjnkolp', basis_coeff_dim4 * offset_residual_dim4_xxoxx, basis_dim4),
        (num_equations * qubits_per_var, num_equations * qubits_per_var, num_equations * qubits_per_var,
         num_equations * qubits_per_var))

    return extended_qubo, basis_map


# evaluate the QUBO given binary vector "eigenvector" and return energy "eigenvalue"
def eval_QUBO(extended_qubo, eigenvector):
    eigenvalue_dim0_o = extended_qubo['qubit_residual_dim0_o']
    eigenvalue_dim0 = eigenvalue_dim0_o

    eigenvalue_dim1_ox = np.einsum('j,j', extended_qubo['qubit_residual_dim1_ox'], eigenvector)
    eigenvalue_dim1_xo = np.einsum('j,j', eigenvector.T, extended_qubo['qubit_residual_dim1_xo'])
    eigenvalue_dim1 = eigenvalue_dim1_ox + eigenvalue_dim1_xo

    eigenvalue_dim2_oxx = np.einsum('jk,j,k', extended_qubo['qubit_residual_dim2_oxx'], eigenvector, eigenvector)
    eigenvalue_dim2_xox = np.einsum('j,jk,k', eigenvector.T, extended_qubo['qubit_residual_dim2_xox'], eigenvector)
    eigenvalue_dim2_xxo = np.einsum('k,j,kj', eigenvector.T, eigenvector.T, extended_qubo['qubit_residual_dim2_xxo'])
    eigenvalue_dim2 = eigenvalue_dim2_oxx + eigenvalue_dim2_xox + eigenvalue_dim2_xxo

    eigenvalue_dim3_xoxx = np.einsum('j,jkl,k,l', eigenvector.T, extended_qubo['qubit_residual_dim3_xoxx'], eigenvector,
                                     eigenvector)
    eigenvalue_dim3_xxox = np.einsum('k,j,kjl,l', eigenvector.T, eigenvector.T,
                                     extended_qubo['qubit_residual_dim3_xxox'], eigenvector)
    eigenvalue_dim3 = eigenvalue_dim3_xoxx + eigenvalue_dim3_xxox

    eigenvalue_dim4_xxoxx = np.einsum('k,j,kjnm,n,m', eigenvector.T, eigenvector.T,
                                      extended_qubo['qubit_residual_dim4_xxoxx'],
                                      eigenvector,
                                      eigenvector)
    eigenvalue_dim4 = eigenvalue_dim4_xxoxx

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
    num_of_qubits = len(extended_qubo['qubit_residual_dim1_ox'])
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


def bit_to_decimal(bx, basis_map):
    presult = []
    nparams = len(basis_map['basis_coeff'])
    bit_precision = len(basis_map['basis'])
    for i in range(nparams):
        presult.append(
            basis_map['basis_coeff'][i]
            * sum(basis_map['basis'] * bx[i * bit_precision:i * bit_precision + bit_precision])
            + basis_map['basis_offset'][i])
    return presult


def main():
    # Get QUBO matrix
    extended_qubo, basis_map = import_QUBO()
    # Get arg min QUBO and compute energy
    ground_state_eigenvector, result_eigenvalue, result_eigenvector = argmin_QUBO(extended_qubo)
    ground_state_eigenvalue = eval_QUBO(extended_qubo, ground_state_eigenvector)

    # Evaluate results
    print("QUBO")
    print("ground state eigenvector = ", ground_state_eigenvector)
    print("ground state eigenvalue  = ", ground_state_eigenvalue)
    print("solution                 = ", bit_to_decimal(ground_state_eigenvector, basis_map))
    # for idx in range(len(result_eigenvalue)):
    #    print(result_eigenvector[idx], result_eigenvalue[idx])


if __name__ == '__main__':
    main()
