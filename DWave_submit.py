#!/usr/bin/env python
# Script to solve QUBO problem using DWSOlveQUBO class
import timeit

start = timeit.default_timer()

from DWSolveQUBO import DWSolveQUBO
import quadratize_poly_solver as quad
import numpy as np

# Declare QUBO data as matrix
qubo, qubo_constant, basis_map, qubo_to_aux_index = quad.main()

# Declare size, assume square matrix QUBO
print("-------------------------")
print("solving on DWave annealer")

n = len(qubo)

# Declare dictionary for QUBO data
qubo_dict = {}

# The QUBO matrix in dictionary form where each qubit(diagonal) and coupling(off-diagonal) weight is assigned to
# physical qubits on the hardware (minor embedding).
qubo_dict.update({(i, j): qubo[i][j] for i in range(n) for j in range(n)})

# Send the DWSolveQUBO class the QUBO_dict; initialization
print
"Solving QUBO"
solve_qubo = DWSolveQUBO(qubo, qubo_dict)

# From DWSolveQUBO get the answer; this is where computation happens
solve_qubo.solvequbo()

# post-process the energies; doesn't do anything at the moment
dwave_total_energies = map(lambda x: x, solve_qubo.dwave_energies)

# This is the lowest-energy observed unembedded results
rqubo_answer = solve_qubo.qubo_ans
# print "QUBO answer is ", rqubo_answer

# Corresponding QUBO value converted to floats
rqubo_energy = np.array([float(dwave_total_energies[i]) for i in range(len(dwave_total_energies))]) + qubo_constant
# print "QUBO min energy is ", min(rqubo_energy)

minenergy_idx = np.argsort(rqubo_energy)
qubo_answer = np.array(rqubo_answer)[minenergy_idx][0]
qubo_energy = np.array(rqubo_energy)[minenergy_idx][0]

# reconstruct solution
quad.quadratized_inverse_mapping(qubo_answer, qubo_energy, basis_map, qubo_to_aux_index)

end = timeit.default_timer()
print("time (s):", end - start)
