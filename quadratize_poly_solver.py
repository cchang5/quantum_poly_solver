import numpy as np
import poly_brute_force as poly

def quadratize(extended_qubo):
    # quadratizes a system of O(2) polynomial equations
    # reduction by substitution (Rosenberg 1975)
    # quadratization in discrete optimization and quantum mechanics
    # section V. A.
    # Nike Dattani arXiv: 1901.04405
    print(extended_qubo)

def main():
    extended_qubo, basis_map = poly.import_QUBO()
    qubo = quadratize(extended_qubo)

if __name__=="__main__":
    main()