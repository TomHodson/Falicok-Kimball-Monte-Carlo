#cython: boundscheck=False, wraparound=False, infer_types=True, initializedcheck=False, cdivision=True

#pull in the random number generation code from the standard lib
cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval) nogil

def py_drand48():
    return drand48()

def py_srand48(seed):
    srand48(seed)

cimport cython
from libc.math cimport exp, log
import numpy as np
import scipy as sp
from scipy import linalg

#TODO change the code to just rotate a single row of this matrix when needed rather than generating it all the time
def interaction_matrix(N, alpha, V, normalise = True, dtype = np.float64):
    row0 = np.abs((N/np.pi * np.sin(np.pi * np.arange(1,N, dtype = dtype)/N)) ** (-alpha) )
    row0 = np.concatenate([[0,], row0])# put the first 0 in by hand
    if normalise: row0 = row0 / np.sum(row0)
    row0 = V * row0
    return linalg.circulant(row0)

cpdef initialise_state_representations(long [::1] state, double [:, ::1] interaction_matrix):
    'initialise useful representations of the state'
    cdef int N = state.shape[0]

    alternating_signs = np.ones(N, dtype = np.int64)
    ut = np.ones(N, dtype = np.int64)
    t = np.ones(N, dtype = np.int64)


    cdef long [::1] alt_v = alternating_signs
    cdef long [::1] ut_v = ut
    cdef long[::1]  t_v = t
    cdef long[::1] st_v = state


    cdef int s = 1
    cdef int i, j
    for i in range(N):
        s = -s
        alt_v[i] = s
        ut_v[i] = (2*st_v[i] - 1)
        t_v[i] = s * ut_v[i]

    cdef double [:] background = (interaction_matrix @ t)
    return alternating_signs, ut, t, background

cpdef double c_classical_energy(
                                    double mu,
                                    long[::1] state,
                                    long[::1] t,
                                    double [::1] background) nogil:
    'compute the energy of the f electrons'
    cdef int N = state.shape[0]
    cdef double F = 0
    cdef int i
    for i in range(N):
        F += - mu * state[i] + t[i] * background[i]

    return F

cdef void invert_site_inplace(
                        long i,
                        long [::1] alternating_signs,
                        long [::1] state,
                        long [::1] ut,
                        long [::1] t,
                        double [::1] background,
                        double [:, ::1] interaction_matrix,
                       ) nogil:
    'invert site i and update the useful state representations in place'
    cdef long N = state.shape[0]

    state[i] = 1 - state[i]
    ut[i] = -ut[i]
    t[i] = -t[i]

    #these expresions only work if we're already flipped the spin
    cdef long dni = ut[i]
    cdef long dti = 2 * t[i]

    cdef int j
    for j in range(N):
        background[j] += interaction_matrix[i, j] * dti

cpdef double incremental_energy_difference(long i,
                                    double mu,
                                    long[::1] ut,
                                    long[::1] t,
                                    double [::1] background) nogil:
    'compute the energy difference for the site i WHICH HAS BEEN FLIPPED ALREADY'
    cdef long dni = ut[i] #the changes are simply related to the value after flipping too
    cdef long dti = 2*t[i]
    cdef double dF_f =  - mu * dni + 2 * dti * background[i]
    return dF_f

cpdef double average_quantum_energy(double beta, double[::1] eigenvalues):
    cdef long N = eigenvalues.shape[0]
    cdef double energy = 0
    cdef int i
    for i in range(N):
      energy += log(1 + exp(-beta * eigenvalues[i]))

    return 1/beta * energy

#cpdef diagonalise(state, double U, double [::1] eigenvalues, double complex [:, ::1] eigenvectors):
#          N = state.shape[0]
#          diagonal = U*state
#          offdiagonal = -np.ones(len(diagonal)-1) #setting t=1 here
#          eigenvalues[:], eigenvectors[:, :] = sp.linalg.eigh_tridiagonal(d = diagonal, e = offdiagonal)

def diagonalise(state, U, eigenvalues, eigenvectors):
          N = state.shape[0]
          diagonal = U*state
          offdiagonal = -np.ones(len(diagonal)-1) #setting t=1 here
          eigenvalues[:], eigenvectors[:, :] = sp.linalg.eigh_tridiagonal(d = diagonal, e = offdiagonal)


cpdef void quantum_cython_mcmc_helper(
                    #outputs
                    double [::1] classical_energies,
                    double [::1] quantum_energies,
                    long [::1] numbers,
                    long [::1] magnetisations,

                    #inputs
                    long [::1] state,
                    long [::1] alternating_signs,
                    long [::1] ut,
                    long [::1] t,
                    double [::1] background,
                    double [:, ::1] interaction_matrix,

                    #parameters
                    long N_steps = 10**5,
                    long N_system = 100,
                    double mu = 0,
                    double beta = 0.1,
                    double V=1,
                    double alpha=1.5,
                    double U=1,
                   ):

      cdef double [::1] eigenvalues = np.zeros(N_system, dtype = np.double);
      cdef double complex [:, ::1] eigenvectors = np.zeros((N_system, N_system), dtype = np.complex128);

      #diagonalise H and put the answers into eigenvalues and eigenvectors
      diagonalise(state, U, eigenvalues, eigenvectors)

      #calculate the first entry in each array of output and make variables to track them
      cdef double classical_energy = c_classical_energy(mu, state, t, background)
      cdef double quantum_energy = average_quantum_energy(beta, eigenvalues)
      cdef long number = np.sum(state)
      cdef long magnetisation = np.sum(t)

      classical_energies[0] = classical_energy
      quantum_energies[0] = quantum_energy
      numbers[0] = number
      magnetisations[0] = magnetisation

      cdef double new_quantum_energy #this is the only variable that might drift, since we're calcualting it anew each time, may aswell just use it

      #variables to track changes in the above until the move is either accepte or rejected
      cdef double quantum_dF #change in quantum free energy
      cdef double classical_dF #change in quantum free energy
      cdef long dn #the change in the particle number, 0 or 1
      cdef long dt # the change in the magnetisation at this site

      cdef long site #the site we're considering flipping
      cdef int i

      for i in range(1,N_steps):
         for site in range(N_system):
             #flip the site
             invert_site_inplace(site, alternating_signs, state, ut, t, background, interaction_matrix)

             #diagonalise H
             diagonalise(state, U, eigenvalues, eigenvectors)

             #calculate all the changes, quantum_dF is the only one that can't be done incrementally
             new_quantum_energy = average_quantum_energy(beta, eigenvalues)
             quantum_dF = new_quantum_energy - quantum_energy
             classical_dF = incremental_energy_difference(site, mu, ut, t, background)
             dn = ut[site]
             dt = 2 * t[site]

             dF = classical_dF + quantum_dF

             #if we must reject this move
             if dF > 0 and exp(- beta * dF) < drand48():
                 #change the site back and don't change the variables
                 invert_site_inplace(site, alternating_signs, state, ut, t, background, interaction_matrix)
             else:
                 #keep the site as it is and update the variables
                 classical_energy += classical_dF
                 quantum_energy = new_quantum_energy #different because this calculation isn't incremental
                 number += dn
                 magnetisation += dt

         classical_energies[i] = classical_energy
         quantum_energies[i] = quantum_energy
         numbers[i] = number
         magnetisations[i] = magnetisation

def quantum_cython_mcmc(N_steps = 10**4,
                N_system = 1000,
                mu = 0,
                beta = 0.1,
                V=1,
                alpha=1.5,
                U=1,
                return_names = False,
                sample_output = False,
                **kwargs,
               ):

    #python setup code goes here
    state = np.arange(N_system, dtype = np.int64) % 2
    M = interaction_matrix(N=N_system, alpha=alpha, V=V)

    energies = np.zeros(shape = N_steps, dtype = np.float64)
    numbers = np.zeros(shape = N_steps, dtype = np.int64)
    magnetisations = np.zeros(shape = N_steps, dtype = np.int64)

    alternating_signs, ut, t, background = initialise_state_representations(state, interaction_matrix=M)

    #if sample_output is true, we just want to get the shape of the data
    if not sample_output:
        quantum_cython_mcmc_helper(
                    #outputs
                    energies,
                    numbers,
                    magnetisations,

                    #inputs
                    state,
                    alternating_signs,
                    ut,
                    t,
                    background,
                    M,

                    #parameters
                    N_steps,
                    N_system,
                    mu,
                    beta,
                    V,
                    alpha,
                    u,
                   )

    return ('energies', 'numbers', 'magnetisations'), (energies, numbers, magnetisations)

if __name__ == 'main':
    print('hello!')
