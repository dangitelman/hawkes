# Author: Dan Gitelman <dan.gitelman@gmail.com>
# A set of helper functions for Hawkes Process parameter estimation.

import numpy as np

cpdef calc_decays(t,u,N,U,beta):
    """Construct decays matrix, sped up using cython.
    
    Parameters
    ----------
    t : ndarray of floats, shape=(N,)
        Array of event arrival times.
        
    u : ndarray of ints, shape=(N,)
        Array of event dimensions.
        
    N : int
        Number of events in Hawkes PP.
        
    U : int
        Number of dimensions in Hawkes PP.
        
    beta : float, optional
            Hawkes Process exponential decay rate parameter.
            
    Returns
    -------
    decays_mat : array_like, size=(N,U)
        Matrix of decay coefficients, filled with
        summation of decay values of prior events in
        each dimension at every event time.
    """
    
    # calculate incremental decay of exponential excitation
    # kernel between events in point process
    decays_mat = np.zeros((N,U))
    time_decay = np.exp(-beta*np.diff(t))
    
    # determine decay coefficients along every dimension
    # at each event for use in EM algorithm
    decays_mat[np.arange(1,N),u[:-1]] = time_decay
    __calc_decays(decays_mat,time_decay,N,U)
    
    # return filled decays matrix
    return decays_mat

cdef void __calc_decays(double[:,:] decays_mat,
                        double[:] time_decay, int N, int U):
    """Iteratively fill decays matrix.
    """
    
    # iterate over each event and dimension,
    # fill each element with relevant decay value
    cdef int i
    for i in range(N-1):
        for j in range(U):
            decays_mat[i+1,j] += decays_mat[i,j]*time_decay[i]
            
cpdef simulate_cython(params,N,U):
    """Simulate Hawkes point-process, sped up using cython.
    
    Parameters
    ----------
    params : tuple
        Tuple of Hawkes process parameters. Must take
        the form (mu,alpha,beta), where mu (ndarray of
        floats) contains the background rate parameters,
        alpha (ndarray of floats) contains the excitation
        rate parameters, and beta (float) is the decay
        parameter.
        
    N : int
        Number of events in Hawkes PP.
        
    U : int
        Number of dimensions in Hawkes PP.
        
    Returns
    -------
    t : ndarray of floats, shape=(N,)
        Array of event arrival times.
        
    u : ndarray of ints, shape=(N,)
        Array of event dimensions.
    """
    
    # extract parameters
    mu,alpha,beta = params
    
    # create arrival time and dimension arrays
    t = np.zeros(N)
    u = np.zeros(N,dtype=np.int8)
    
    # call simulation function in C
    __simulate_cython(t,u,mu,alpha,beta,N,U)
    
    # return simulated PP
    return t,u
    
    
cdef void __simulate_cython(double[:] t, signed char[:] u,
                            double[:] mu, double[:,:] alpha,
                            double beta, int N, int U):
    """Sequentially simulate events in Hawkes PP.
    """
    
    # cython variable declarations
    cdef int j,k
    cdef int loop = 0
    cdef int count = 0
    cdef double ti = 0
    cdef double e_const = np.e
    cdef double p,w,lam_bar,e_neg_bw
    cdef double sum_baseline = np.sum(mu)
    cdef double[:] summand = np.zeros(U)
    cdef double[:] cumsum = np.zeros(U)
    cdef double[:,:] beta_alpha = np.zeros((U,U))
    for i in range(U):
        for j in range(U):
            beta_alpha[i,j] = alpha[i,j]*beta
    
    # cache randomly generate uniform r.v's (2 times
    # as many as needed to allow for rejection tolerance)
    cdef double[:] D = np.random.uniform(size=N*2)
    cdef double[:] log_rands = -np.log(np.random.uniform(size=N*2))
    
    # Ogata-motivated thinning algorithm for multivariate
    # Hawkes PP simulation
    while count < N:
        # generate exponential arrival parametrized by maximum
        # possible interarrival Hawkes intensity
        lam_bar = sum_baseline
        for j in range(U):
            lam_bar += summand[j]
        w = log_rands[loop]/lam_bar
        ti += w
        e_neg_bw = e_const**(-beta*w)
        for j in range(U):
            summand[j] *= e_neg_bw
            cumsum[j] = mu[j] + summand[j]
            if j > 0:
                cumsum[j] += cumsum[j-1]
                
        # generate uniform and accept with probability
        # sum of cumsum divided by upper bound lam_bar
        p = D[loop]*lam_bar
        if p <= cumsum[-1]:
            # assign dimension
            k = 0
            while p > cumsum[k]:
                k += 1
            # store generated event
            t[count] = ti
            u[count] = k
            count += 1
            for j in range(U):
                summand[j] += beta_alpha[j,k]
        # advance loop attempts
        loop += 1