# quantum_poly_solver

## Description
Scripts that demonstrate a direct algorithm for solving systems of polynomial equations using quantum annealing. These scripts are provided as supplemental material to [SREP-19](https://www.nature.com/srep/). Evaluation on a commercially available D-Wave Quantum Annealer is also provided given a valid api-key.

- `poly_brute_force.py`
  - `define_problem()` </br> Defines the system of polynomial equations in the form of Eq. (1). The definition of the search space is provided by Eq. (2).
  - `argmin_QUBO(qubo)` </br> valuates the system of equtions in the qubit basis by brute force. The full 2<sup>n</sup> Hilbert-space is explicitly evaluted, and sorted for the ground state eigenvalue and eigenvector. 
  - `main()` </br> Runs the example discussed in the paper.
  - Three representations of the objective function is provided:
    - `extended qubo` is in general a set of dense tensors that define the system of equations in the qubit basis.
    - `upper triangular qubo` accumulates the entries in `extended qubo` such that all entries of i < j < k = 0
    - `reduced upper triangular qubo` reduces the order of repeated indices (e.g. Q<sup>(2)</sup><sub>11</sub> → Q<sup>(1)</sup><sub>1</sub>)
    
- `quadratize_poly_solver.py`
  - `quadratize(qubo)` </br> Maps a quadratic system of equations with a reduced upper triangular tensor representation up to rank 3, down to a rank 2 tensor representation (QUBO). The quadratization is performed using reduction-by-substitution [Rosenberg 1975, [Dattani 2019](https://arxiv.org/abs/1901.04405)].
  - `argmin_QUBO(qubo)` </br> Evalues the resulting QUBO with brute force.
  - `main()` </br> Runs the example discussed in the paper.

- `DWave_submit.py`
  - Depends on `dwave_sapi2` library and can be downloaded at [Qubist](https://cloud.dwavesys.com/qubist/downloads/) (behind a login but with a Boost license).
  - `apikey-example.txt` </br> A plain text file to store a valid api key for remote access to a DWave annealer. Rename as `apikey.txt` for the script to access it.
  - Running this script will submit the quadratized QUBO generated in `quadratized_poly_solver.py` to the annealer. Annealer parameters are defined in `DWSolveQUBO.py` under `DWSolveQUBO.params`.

## Authors
* [@cchang5](https://github.com/cchang5)
* [@travishumble](https://github.com/travishumble)

## Copyright Notice
