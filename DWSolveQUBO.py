#!/usr/bin/env python
########################################################################################################################
#                                              DW Solve QUBO
########################################################################################################################
# The following is a class which solves QUBO's on the D-Wave. It takes a QUBO dictionary containing the qubit and
# coupler weight assigned to the physical hardware qubits (Q= {(0, 4) = 23.3}). It returns the energy and bit string.
########################################################################################################################
from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.remote import RemoteConnection
from dwave_sapi2.embedding import find_embedding
from dwave_sapi2.util import qubo_to_ising
from dwave_sapi2.embedding import embed_problem
from dwave_sapi2.core import solve_ising
from dwave_sapi2.embedding import unembed_answer
import numpy as np

class DWSolveQUBO:
    def __init__(self, qubo, qubo_dict):
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # INITIALIZATION:
        # get qubo and qubo_dict from QUBO_linear.py
        self.qubo = qubo
        self.qubo_dict = qubo_dict
        # D-Wave remote connection
        self.url = 'https://cloud.dwavesys.com/sapi'
        with open('./apikey.txt') as apikeyfile:
            apikey = apikeyfile.readline()
        self.token = apikey
        # create a remote connection
        self.conn = RemoteConnection(self.url, self.token)
        # NB auto_scale is set TRUE so you SHOULD NOT have to rescale the h and J (manual rescaling is optional and
        # included in this program.)
        # answer_mode: raw, histogram
        self.params = {"annealing_time": 1, "answer_mode": "raw", "auto_scale": True, "postprocess": "",
                       "num_reads": 2000, "num_spin_reversal_transforms":10}
        print(self.params)
        # get the solver
        self.solver = self.conn.get_solver('DW_2000Q_2_1')
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # EMBEDDING CONTROLS:
        # this logical value indicates whether to clean up the embedding. AKA removing physical variables that are
        # adjacent to a single variable in the same chain or not adjacent to any variables in other chains.
        self.clean = False
        # this logical value indicates whether to smear an embedding to increase the chain size so that the h values do
        # not exceed the scale of J values relative to h_range and J_range respectively.
        self.smear = False
        # a list representing the range of h values, these values are only used when smear = TRUE
        self.h_range = [-1, 1]
        self.J_range = [-1, 1]
        # SOLVE_ISING VARIABLES:
        # the hardware adjacency matrix
        self.Adjacency = None
        # the embedding
        self.Embedding = None
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # D-WAVE VARIABLES:
        # h is the vector containing the linear ising coefficients
        self.h = None
        self.h_max = None
        # J is the matrix containing the quadratic ising coefficients in dictionary form where each qubit and coupler
        # value is assigned to qubits on the physical hardware
        self.J = None
        self.J1 = None
        # ising_offset is a constant which shifts all ising energies
        self.ising_offset = None
        # embedded h values
        self.h0 = None
        self.h1 = None
        # embedded J values
        self.j0 = None
        # strong output variable couplings
        self.jc = None
        # what the d-wave returns from solve_ising method
        self.dwave_return = None
        # the unembedded version of what the d-wave returns
        self.unembed = None
        # ising answer
        self.ising_ans = None
        self.ising_energies = None
        self.h_energy = None
        self.J_energy = None
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # QUBO RESULT VARIABLES:
        # qubo answer
        self.qubo_ans = None
        self.qubo_energy = None
        self.dwave_energies = None

    def solvequbo(self):
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # EMBEDDING:
        # gets the hardware adjacency for the solver in use.
        self.Adjacency = get_hardware_adjacency(self.solver)
        # gets the embedding for the D-Wave hardware
        self.Embedding = find_embedding(self.qubo_dict, self.Adjacency)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # CONVERSIONS AND RESCALING:
        # convert qubo to ising
        (self.h, self.J, self.ising_offset) = qubo_to_ising(self.qubo_dict)
        # Even though auto_scale = TRUE, we are rescaling values
        # Normalize h and J to be between +/-1
        self.h_max = max(map(abs, self.h))

        if len(self.J.values()) > 0:
            j_max = max([abs(x) for x in self.J.values()])
        else:
            j_max = 1
        # In [0,1], this scales down J values to be less than jc
        j_scale = 0.8

        # Use the largest large value
        if self.h_max > j_max:
            j_max = self.h_max

        # This is the actual scaling
        rescale = j_scale / j_max
        self.h1 = map(lambda x:  rescale * x, self.h)

        if len(self.J.values()) > 0:
            self.J1 = {key: rescale*val for key, val in self.J.items()}
        else:
            self.J1 = self.J
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # EMBEDDING:
        # gets the hardware adjacency for the solver in use.
        self.Adjacency = get_hardware_adjacency(self.solver)
        # gets the embedding for the D-Wave hardware
        self.Embedding = find_embedding(self.qubo_dict, self.Adjacency)
        # Embed the rescale values into the hardware graph
        [self.h0, self.j0, self.jc, self.Embedding] = embed_problem(self.h1, self.J1, self.Embedding, self.Adjacency,
                                                                    self.clean, self.smear, self.h_range, self.J_range)
        # embed_problem returns two J's, one for the biases from your problem, one for the chains.
        self.j0.update(self.jc)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # SOLVE PROBLEM ON D-WAVE:
        # generate the embedded solution to the ising problem.
        self.dwave_return = solve_ising(self.solver, self.h0, self.j0, **self.params)
        #print("dwave_return")
        #print(self.dwave_return['solutions'])
        # the unembedded answer to the ising problem.
        unembed = np.array(unembed_answer(self.dwave_return['solutions'], self.Embedding, broken_chains="minimize_energy",
                                 h=self.h, j=self.J)) #[0]
        # convert ising string to qubo string
        ising_ans = [list(filter(lambda a: a != 3, unembed[i])) for i in range(len(unembed))]
        #print(ising_ans)
        #print("ISING ANS")
        # Because the problem is unembedded, the energy will be different for the embedded, and unembedded problem.
        # ising_energies = dwave_return['energies']
        self.h_energy = [sum(self.h1[v] * val for v, val in enumerate(unembed[i])) for i in range(len(unembed))]
        self.J_energy = [sum(self.J1[(u, v)] * unembed[i,u] * unembed[i,v] for u, v in self.J1) for i in range(len(unembed))]
        self.ising_energies = np.array(self.h_energy) + np.array(self.J_energy)
        #print(self.h_energy)
        #print(self.J_energy)
        #print(self.ising_energies)
        #print("ENERGIES")
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # CONVERT ANSWER WITH ENERGY TO QUBO FORM:
        # Rescale and add back in the ising_offset and another constant
        self.dwave_energies = self.ising_energies/rescale + self.ising_offset #[map(lambda x: (x / rescale + self.ising_offset), self.ising_energies[i]) for i in range(len(self.ising_energies))]
        # QUBO RESULTS:
        self.qubo_ans = (np.array(ising_ans)+1)/2 #[map(lambda x: (x + 1) / 2, ising_ans[i]) for i in range(len(ising_ans))]

