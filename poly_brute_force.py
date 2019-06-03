#!/usr/bin/env python
# Input a QUBO instance and solve using brute force

import numpy as np

# import the QUBO data and return numpy 2D square array
def import_QUBO():
    # define problem
    neq = 2

    P0 = np.zeros(neq)
    P0[0] = -51.
    P0[1] = -46.

    P1 = np.zeros((neq,neq))
    P1[0,0] = 2
    P1[0,1] = 4
    P1[1,0] = 3
    P1[1,1] = 2

    P2 = np.zeros((neq,neq,neq))
    P2[0,0,0] = 2
    P2[0,0,1] = 3
    P2[0,1,0] = 0
    P2[0,1,1] = 1
    
    P2[1,0,0] = 1
    P2[1,0,1] = 2
    P2[1,1,0] = 0
    P2[1,1,1] = 2

    # construct squared residual matrices
    # R = R0 + R1 + R2 + ....
    residual_dim0 = np.einsum('i,i', P0, P0)

    residual_dim1_blu = np.einsum('i,ij->j', P0.T, P1)
    residual_dim1_gre = np.einsum('ji,i->j', P1.T, P0)
    
    residual_dim2_blu = np.einsum('i,ijk->jk', P0.T, P2)
    residual_dim2_gre = np.einsum('ji,ik->jk', P1.T, P1)
    residual_dim2_yel = np.einsum('kji,i->kj', P2.T, P0)

    residual_dim3_gre = np.einsum('ji,ikl->jkl', P1.T, P2)
    residual_dim3_yel = np.einsum('kji,il->kjl', P2.T, P1)

    residual_dim4 = np.einsum('kji,inm->kjnm', P2.T, P2)

    # define search
    nbit = 3
    basis = np.array([2**i for i in range(nbit)])

    basis_offset = np.array([0,0])
    basis_coeff = np.ones(neq)

    basis_maps = {'basis':basis, 'basis_offset':basis_offset, 'basis_coeff':basis_coeff}
    
    # define QUBO shifts
    #B0_1 = (np.einsum('ij,j->i',P1,amin)+np.einsum('ji,j->i',P1,amin))/2.
    #B0_2 = (np.einsum('ijk,j,k->i',P2,amin,amin)+np.einsum('kij,j,k->i',P2,amin,amin)+np.einsum('jki,j,k->i',P2,amin,amin))/3.

    ##Am1 = np.einsum('i,j->ij',amin,amin)
    ##Ps1 = np.einsum('ijk->ij',P2)+np.einsum('kij->ij',P2)+np.einsum('jki->ij',P2)
    ##B1_2 = Am1*Ps1/3.
    #B1_2 = (np.einsum('ijk,k->ij',P2,amin)+np.einsum('kij,j->ij',P2,amin)+np.einsum('jki,j->ij',P2,amin))/3.
    #
    ## combine P and B
    #S0 = P0-B0_1-B0_2
    #S1 = P1+B1_2
    #S2 = P2

    offset_residual_dim0 = residual_dim0

    offset_residual_dim1_blu = residual_dim1_blu
    offset_residual_dim1_gre = residual_dim1_gre

    offset_residual_dim2_blu = residual_dim2_blu
    offset_residual_dim2_gre = residual_dim2_gre
    offset_residual_dim2_yel = residual_dim2_yel

    offset_residual_dim3_gre = residual_dim3_gre
    offset_residual_dim3_yel = residual_dim3_yel

    offset_residual_dim4 = residual_dim4

    # expand to qubit basis
    extended_qubo = dict()

    # dimension 0
    extended_qubo['qubit_residual_dim0'] = offset_residual_dim0

    # dimension 1
    extended_qubo['qubit_residual_dim1_blu'] = np.reshape(np.einsum('i,j->ij',basis_coeff*offset_residual_dim1_blu, basis), (neq*nbit))
    extended_qubo['qubit_residual_dim1_gre'] = np.reshape(np.einsum('i,j->ij',basis_coeff*offset_residual_dim1_gre, basis), (neq*nbit))
    
    # dimension 2
    basis_coeff_dim2 = np.einsum('i,j->ij', basis_coeff, basis_coeff)
    basis_dim2 = np.einsum('i,j->ij', basis, basis)
    
    extended_qubo['qubit_residual_dim2_blu'] = np.reshape(np.einsum('ij,kl->ikjl',basis_coeff_dim2*offset_residual_dim2_blu, basis_dim2), (neq*nbit, neq*nbit))
    extended_qubo['qubit_residual_dim2_gre'] = np.reshape(np.einsum('ij,kl->ikjl',basis_coeff_dim2*offset_residual_dim2_gre, basis_dim2), (neq*nbit, neq*nbit))
    extended_qubo['qubit_residual_dim2_yel'] = np.reshape(np.einsum('ij,kl->ikjl',basis_coeff_dim2*offset_residual_dim2_yel, basis_dim2), (neq*nbit, neq*nbit))

    # dimension 3
    basis_coeff_dim3 = np.einsum('i,j,k->ijk', basis_coeff, basis_coeff, basis_coeff)
    basis_dim3 = np.einsum('i,j,k->ijk', basis, basis, basis)

    extended_qubo['qubit_residual_dim3_gre'] = np.reshape(np.einsum('ijk,lmn->iljmkn',basis_coeff_dim3*offset_residual_dim3_gre, basis_dim3), (neq*nbit, neq*nbit, neq*nbit))
    extended_qubo['qubit_residual_dim3_yel'] = np.reshape(np.einsum('ijk,lmn->iljmkn',basis_coeff_dim3*offset_residual_dim3_yel, basis_dim3), (neq*nbit, neq*nbit, neq*nbit))

    # dimension 4
    basis_coeff_dim4 = np.einsum('i,j,k,l->ijkl', basis_coeff, basis_coeff, basis_coeff, basis_coeff)
    basis_dim4 = np.einsum('i,j,k,l->ijkl', basis, basis, basis, basis)

    extended_qubo['qubit_residual_dim4'] = np.reshape(np.einsum('ijkl,mnop->imjnkolp', basis_coeff_dim4*offset_residual_dim4, basis_dim4), (neq*nbit, neq*nbit, neq*nbit, neq*nbit))
    
    return extended_qubo, basis_maps

# evaluate the QUBO given binary vector b and return energy F
def eval_QUBO(extended_qubo, eigenvector):
    D0 = extended_qubo['qubit_residual_dim0']

    D1_blu = np.einsum('j,j', extended_qubo['qubit_residual_dim1_blu'], eigenvector)
    D1_gre = np.einsum('j,j', eigenvector.T, extended_qubo['qubit_residual_dim1_gre'])
    D1 = D1_blu + D1_gre

    D2_blu = np.einsum('jk,j,k', extended_qubo['qubit_residual_dim2_blu'], eigenvector, eigenvector)
    D2_gre = np.einsum('j,jk,k', eigenvector.T, extended_qubo['qubit_residual_dim2_gre'], eigenvector)
    D2_yel = np.einsum('k,j,kj', eigenvector.T, eigenvector.T, extended_qubo['qubit_residual_dim2_yel'])
    D2 = D2_blu + D2_gre + D2_yel

    D3_gre = np.einsum('j,jkl,k,l', eigenvector.T, extended_qubo['qubit_residual_dim3_gre'], eigenvector, eigenvector)
    D3_yel = np.einsum('k,j,kjl,l', eigenvector.T, eigenvector.T, extended_qubo['qubit_residual_dim3_yel'], eigenvector)
    D3 = D3_gre + D3_yel

    D4 = np.einsum('k,j,kjnm,n,m', eigenvector.T, eigenvector.T, extended_qubo['qubit_residual_dim4'], eigenvector, eigenvector)

    eigenvalue = D0 + D1 + D2 + D3 + D4
    return eigenvalue

# Convert non-negative n-bit integer to n-bit binary representation and return numpy array
def int_to_bin(hilbert_index, num_of_qubits):
    length = int(hilbert_index).bit_length(); # length of binary conversion
    # Check that the bit length fits the b-bit representation
    if length > num_of_qubits:
        print(" <<Bit length exceeds repreesntation size>>")
        raise ValueError
    x = bin(int(hilbert_index)); # binary converstion returns string x
    y =x[2:length+2] # store last l chars of x in y
    b = np.zeros(num_of_qubits);
    for i in range(len(y)):
        b[num_of_qubits-length+i] = int(y[i]); # add the bits from smallest to largest in the last l slots
    return b

# Find argument that minimizes QUBO F(b) = b^T Q b and return as numpy array
def argmin_QUBO(extended_qubo):
    num_of_qubits = len(extended_qubo['qubit_residual_dim1_blu'])
    ground_state_eigenvector = int_to_bin(hilbert_index=0, num_of_qubits=num_of_qubits)
    ground_state_eigenvalue = eval_QUBO(extended_qubo, ground_state_eigenvector)
    result_eigenvalue = []
    result_eigenvector = []
    for h_idx in range(2**num_of_qubits): # loop over all 2^n possibilities
        eigenvector = int_to_bin(h_idx, num_of_qubits)
        eigenvalue = eval_QUBO(extended_qubo, eigenvector)
        result_eigenvalue.append(eigenvalue)
        result_eigenvector.append(eigenvector)
        if eigenvalue < ground_state_eigenvalue:
            ground_state_eigenvalue = eigenvalue
            ground_state_eigenvector = eigenvector
    return ground_state_eigenvector, result_eigenvalue, result_eigenvector

def posterior(bx, basis_map):
    presult = []
    nparams = len(basis_map['basis_coeff'])
    bit_precision = len(basis_map['basis'])
    for i in range(nparams):
        presult.append(basis_map['basis_coeff'][i]*sum(basis_map['basis']*bx[i*bit_precision:i*bit_precision+bit_precision])+basis_map['basis_offset'][i])
        print(basis_map['basis_offset'][i])
    return presult

#############################################
if __name__=='__main__':
    # Get QUBO matrix
    extended_qubo, basis_map = import_QUBO()
    # Get arg min QUBO and compute energy
    bx, resultE, resultb = argmin_QUBO(extended_qubo)
    Fx = eval_QUBO(extended_qubo, bx)
    
    # Evaluate results
    print("QUBO")
    print("b1 = ", bx)
    print("F(b1) = ", Fx)
    print("P(p|D)=", posterior(bx, basis_map))
    #for idx in range(len(resultE)):
    #    print(resultb[idx], resultE[idx])
