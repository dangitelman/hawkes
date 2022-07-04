# Author: Dan Gitelman <dan.gitelman@gmail.com>
# Hawkes Process parameter estimation.

import numpy as np
from time import time
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt

from helpers import calc_decays,simulate_cython

class HawkesEM:
    """A Hawkes Process parameter estimator.
    
    Description
    ----------
    Excitatory point processes describe random event
    sequences in which new arrivals increase rates of future
    occurrence. The excitation framework has significant
    applications to fields that feature elements of temporal
    clustering across different dimensions; examples include
    modeling of earthquake aftershocks, crime rates, and
    financial contagion.

    The multivariate Hawkes Process put forward in Hawkes (1971)
    is a popular model for excitatory point processes. Its
    conditional intensity function (CIF) leverages
    a Poissonian background process to generate immigrant
    events with independent arrival times; immigrants then
    excite arrivals of offspring events, which themselves
    generate more offspring.

    Alternatively, the generating process can be reframed as
    a mixture of independent Poisson processes, namely the
    immigrant-generating background rate and past arrivals’
    time-decaying excitatory influence. The Hawkes process
    therefore lends itself to a branching structure, whereby
    each arrival is spawned by a single generating process in
    the Poissonian mixture.

    For the purposes of this class, we consider the Hawkes
    process with exponential memory kernel and stationary
    background rate.
            
    Attributes
    ----------
    t : ndarray of floats, shape=(N,)
        Array of event arrival times.
        
    u : ndarray of ints, shape=(N,)
        Array of event dimensions.
        
    U : int
        Number of dimensions in Hawkes PP.
        
    N : int
        Number of events in Hawkes PP.
        
    T : float
        Hawkes PP event horizon time.
        
    N_dim : int
        Number of events in Hawkes PP along each dimension.
    """
    
    def simulate_data(self,params,N):
        """Simulate Hawkes point-process.
        
        Parameters
        ----------
        params : tuple
            Tuple of Hawkes process parameters.
        
        N : int
            Number of events to be simulated in Hawkes PP.
        """
        
        # ensure dimensions match
        mu,alpha = params[:2]
        U = mu.size
        assert alpha.shape == (U,U)
        assert U < 256
        
        # ensure spectral radius of excitation coefficient
        # matrix is less than 1 (such that process is stable)
        spectral_rad = np.linalg.eig(alpha)[0].real[0]
        assert spectral_rad < 1
        
        # simulate Hawkes point-process
        t,u = self._simulate_data(params,N,U)
        
        # ensure dimension validity
        assert all([(i in u) for i in range(U)])
        
        # fill attributes
        self.t = t; self.u = u
        self.U = U; self.N = N; self.T = t[-1]
        _,self.N_dim = np.unique(u,return_counts=True)
    
    @staticmethod
    def _simulate_data(params,N,U):
        """Simulate events in Hawkes PP using cython code.
        """
        return simulate_cython(params,N,U)
        
    def load_data(self,t,u):
        """Load existing point process event data.
        
        Parameters
        ----------
        t : ndarray of floats, shape=(,)
            Array of event arrival times. Must be in increasing
            order.
        
        u : ndarray of ints, shape=(,)
            Array of event dimensions. Must take on all integer
            values in the range [0,max_val), where max_val is
            the maximum value of elements in the array.
        """
        
        # ensure event array sizes match
        assert t.size == u.size
        
        # ensure event arrival times are increasing
        assert np.all(np.diff(t) >= 0)
        
        # ensure dimension validity
        U = np.max(u) + 1
        assert np.all(u >= 0)
        assert all([(i in u) for i in range(U)])
        
        # fill attributes
        self.t = t; self.u = u
        self.U = U; self.N = t.size; self.T = t[-1]
        _,self.N_dim = np.unique(u,return_counts=True)
        
    def _prep(self,beta):
        """Calculate decay coefficients and weights for
        log-likelihood calculation.
        """
        
        # determine decay coefficients along every dimension
        # at each event for use in EM algorithm
        decays_mat = calc_decays(self.t,self.u,self.N,self.U,beta)
        
        # find log-likelihood calculation weights
        wgts = 1-np.exp(-beta*(self.T-self.t))
        
        # cache EM coefficients
        decays = []; loglike_wgt = np.zeros(self.U)
        for i in range(self.U):
            # mask out relevant dimension
            mask = (self.u==i)
            # fill values
            decays.append(decays_mat[mask])
            loglike_wgt[i] = np.sum(wgts[mask])
        
        # return EM coefficients
        return decays,loglike_wgt
    
    @staticmethod
    def _ckpt_verbose(loglike,epoch,inc,start_time):
        """Print time elapsed and loglike difference at each
        checkpoint if verbose set to True"""
        
        # calculate loglike difference only of at least two
        # measurements have been recorded
        new_time = time()
        if epoch >= inc*2:
            print("Epoch {:5d} -> {:6.2f}s - diff: {:.2e}".format(
                epoch,new_time-start_time,
                (loglike[-1]-loglike[-2])/inc))
        else:
            print("Epoch {:5d} -> {:6.2f}s {}".format(
                epoch,new_time-start_time,'-'*16))
    
    def _EM(self,beta,eps,inc,verbose):
        """Generate Hawkes Process latent variable estimates using
        Expectation-Maximization (EM) algorithm, given exponential
        decay rate hyperparameter beta.
        """
        
        # set random starting values
        mu = np.random.uniform(0,1,size=self.U)
        alpha = np.random.uniform(0,1,size=(self.U,self.U))
        
        # prepare EM coefficients
        if verbose: start_time = time()
        decays,loglike_wgt = self._prep(beta)
        if verbose:
            print("Preparation -> {:6.2f}s {}".format(
                time()-start_time,'-'*16))
        
        # iteratively perform EM updates until
        # stopping criterion is reached
        epoch = 0; loglike = []; stop_criterion = False
        while not stop_criterion:
            # advance an epoch
            epoch += 1; check_loglike = (epoch % inc == 0)
            if check_loglike: loglike.append(0)
            
            # update background and excitation rates
            # sequentially for each all U processes
            for i in range(self.U):
                # peform iteration of EM algorithm
                base_rate,regularizer = self._E_step(mu,alpha,beta,decays,i)
                mu[i],alpha[i] = self._M_step(mu,alpha,beta,decays,base_rate,regularizer,i)
                
                # maximize conditional intensities of observed points
                if check_loglike: loglike[-1] += np.sum(np.log(regularizer))
            
            # calculate log-likelihood and check stop condition
            if check_loglike:
                # penalize intensities at non-arrival times
                loglike[-1] -= self.T*np.sum(mu) + np.sum(alpha@loglike_wgt)
                
                # print time elapsed and loglike difference
                # at each checkpoint if verbose
                if verbose: self._ckpt_verbose(loglike,epoch,inc,start_time)
                    
                # update stopping criterion
                if epoch >= inc*2:
                    if loglike[-1]-loglike[-2] < eps*inc:
                        stop_criterion = True
                        
        # return parameter estimates and final log-likelihood
        return mu,alpha,loglike[-1]
    
    def _E_step(self,mu,alpha,beta,decays,i):
        """Expectation step"""
        base_rate = self._get_base_rate(mu,i)
        A = decays[i]@(alpha[i]*beta)
        regularizer = base_rate + A
        return base_rate,regularizer
    
    def _M_step(self,mu,alpha,beta,decays,base_rate,regularizer,i):
        """Maximization step"""
        inv_reg = 1/regularizer
        q_sum = np.sum(base_rate*inv_reg)
        p_sum = alpha[i]*beta*(decays[i].T@inv_reg)

        # update params
        mu_i = q_sum/self.T
        alpha_i = p_sum/self.N_dim
        return mu_i,alpha_i
    
    @staticmethod
    def _get_base_rate(mu,i):
        """Return base rate in E-step"""
        # use homogenous background rate
        base_rate = mu[i]
        return base_rate
    
    @staticmethod
    def _beta_obj(beta_hat,f_EM,verbose,start_time):
        """Objective function for decay rate search."""
        
        # get log-likelihood from EM procedure if beta > 0,
        # otherwise minval
        if beta_hat <= 0:
            loglike = -1e6
        else:
            _,_,loglike = f_EM(beta_hat)
            
        # print state if verbose
        if verbose:
            print("beta = {:7.4f} -> {:6.2f}s - loglike: {:.6e}".format(
                beta_hat,time()-start_time,
                loglike))
            
        # return negative log-likelihood
        return -loglike
    
    def fit(self,beta=None,eps=0.01,inc=20,verbose=False):
        """Generate Hawkes Process latent variable estimates using
        Expectation-Maximization (EM) algorithm.
        
        Parameters
        ----------
        beta : float, optional (default=None)
            Model hyperparameter specifying exponential rate of
            excitation decay. If not specified, will optimize
            for beta and return estimate of log-likelihood
            maximizing decay rate.
        
        eps : float, optional (default=0.01)
            Stopping criterion threshold for increases in
            complete log-likelihood values.
        
        inc : int, optional (default=20)
            Interval number of epochs at which log-likelihood
            will be calculated and stopping criterion will be
            evaluated.
            
        verbose : bool, optional (default=False)
            Provide details of EM algorithm progress.
        
        Returns
        -------
        mu : ndarray of floats, shape=(U,)
            Estimated Hawkes Process background rate coefficients.
        
        alpha : ndarray of floats, shape=(U,U)
            Estimated Hawkes Process excitation rate coefficients.
            
        beta : float, optional
            Estimated Hawkes Process exponential decay rate
            parameter. Returned only if beta is not a pre-specified
            hyperparameter.
        """
        
        # use beta hyperparameter if pre-specified, otherwise
        # search for optimal beta
        if beta == None:
            # initial guess
            b0 = np.array([0])
                
            # objective function
            f_EM = lambda beta_hat: self._EM(beta_hat,eps,inc,verbose=False)
            obj = lambda beta_hat: self._beta_obj(beta_hat[0],f_EM,verbose,start_time)
            
            # optimize of candidate betas
            if verbose: start_time = time()
            res = minimize(obj,b0,method='COBYLA',tol=eps)
            
            # perform EM using cross-validated beta
            beta_hat = res.x
            mu,alpha,_ = f_EM(beta_hat)
            
            # return parameter estimates
            return mu,alpha,beta_hat
        else:
            # peform EM using hyperparameter beta
            mu,alpha,_ = self._EM(beta,eps,inc,verbose)
            
            # return parameter estimates
            return mu,alpha
        
    def _plot_args(self,kwargs,hdelta,vdelta):
        """Parse arguments for plotting functions"""
        
        # set horizon value
        if 'horizon' in kwargs.keys():
            horizon = kwargs['horizon']
        else:
            horizon = self.T
        
        # set dimension labels
        if 'dim_labels' in kwargs.keys():
            dim_labels = kwargs['dim_labels']
        else:
            dim_labels = np.arange(self.U,dtype=np.int)
            
        # set figure size
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (hdelta,self.U*vdelta)
            
        return horizon,dim_labels,figsize
        
    
    def plot_events(self,**kwargs):
        """Plot event times of point processes across each dimension.
        
        Parameters
        ----------
        **kwargs : keyword args, optional
        Optional arguments to pass to method.
        Examples of optional kwargs:
        
          - horizon : float
            Time horizon value for plot; x-axis runs over interval
            [0,horizon).
        
          - dim_labels : ndarray, shape=(U,)
            Array of dimension names to label each point process.
            
          - figsize : ndarray of ints, shape=(2,)
            Size of resulting plot in the form (horiz_dim,vert_dim).
        """
        
        # get dimension labels and plot size
        horizon,dim_labels,figsize = self._plot_args(kwargs,hdelta=15,vdelta=0.75)
        fig, ax = plt.subplots(self.U,sharex='col',figsize=figsize)
        
        # mask out events in interval
        in_horizon = self.t <= horizon
        event_times = self.t[in_horizon]
        event_dims = self.u[in_horizon]
        
        # plot each point process
        for i in range(self.U):
            # plot event times
            ts = event_times[event_dims==i]
            ys = np.zeros(len(ts))
            ax[i].plot(ts, ys, 'o', color='blue', alpha=0.2)
            
            # print dimension labels and set axis limits 
            ax[i].set_yticks([0])
            ax[i].set_yticklabels(['$p_{%s}$' % dim_labels[i]])
            ax[i].set_xlim([0,horizon])
            
        # set title and axis label
        ax[0].set_title('Event Arrival Times')
        ax[-1].set_xlabel('Time')
        fig.tight_layout()
        plt.show()
        
    def plot_intensities(self,params,n_intvl=1000,**kwargs):
        """Plot Hawkes intensities of point processes across each dimension.
        
        Parameters
        ----------
        params :
        
        n_intvl :
        
        **kwargs : keyword args, optional
        Optional arguments to pass to method.
        Examples of optional kwargs:
        
          - dim_labels : ndarray, shape=(U,)
            Array of dimension names to label each point process.
            
          - figsize : ndarray of ints, shape=(2,)
            Size of resulting plot in the form (horiz_dim,vert_dim).
        """
        
        # extract Hawkes parameters
        mu,alpha,beta = params
        
        # get dimension labels and plot size
        horizon,dim_labels,figsize = self._plot_args(kwargs,hdelta=15,vdelta=1.5)
        fig, ax = plt.subplots(self.U,sharex='col',figsize=figsize)
        
        # mask out events in interval
        in_horizon = self.t <= horizon
        event_times = self.t[in_horizon]
        event_dims = self.u[in_horizon]
        
        # create even partitions in [0,T] and calculate increment size
        intervals = np.linspace(0,horizon,n_intvl)
        delta = horizon/(n_intvl-1)
        
        # calculate decay values for intensity calculations;
        decays_mat = np.zeros((n_intvl,self.U))
        t_rnd = np.ceil(event_times/delta).astype(int)-1
        for i in range(len(t_rnd)):
            inc_decay = np.exp(-beta*(t_rnd[i]*delta-event_times[i]))
            decays_mat[t_rnd[i],self.u[i]] += inc_decay
        time_decay = np.exp(-beta*delta)
        for li in range(n_intvl-1):
            decays_mat[li+1] += decays_mat[li]*time_decay
        
        # plot intensities across each dimension
        for i in range(self.U):
            # calculate intensities and plot
            intensities = mu[i] + decays_mat@(beta*alpha[i])
            ax[i].plot(intervals,intensities,color='blue')
            
            # plot events points on intensity graph
            t_rnd_i = t_rnd[event_dims==i]
            ax[i].scatter(t_rnd_i*delta,intensities[t_rnd_i],color='blue',alpha=0.2)
            
            # set axis label and limits
            ax[i].set_xlim([0,horizon])
            ax[i].set_ylabel('$\lambda_{%s}(t)$' % dim_labels[i])
        
        # set title and axis label
        ax[0].set_title('Hawkes Intensities')
        ax[-1].set_xlabel('Time')
        fig.tight_layout()
        plt.show()
    
class TimeVarHawkesEM(HawkesEM):
    """An estimator of Hawkes Process parameters with time-varying
    background rates.
    
    Description
    ----------
    Excitatory point processes describe random event
    sequences in which new arrivals increase rates of future
    occurrence. The excitation framework has significant
    applications to fields that feature elements of temporal
    clustering across different dimensions; examples include
    modeling of earthquake aftershocks, crime rates, and
    financial contagion.

    The multivariate Hawkes Process put forward in Hawkes (1971)
    is a popular model for excitatory point processes. Its
    conditional intensity function (CIF) leverages
    a Poissonian background process to generate immigrant
    events with independent arrival times; immigrants then
    excite arrivals of offspring events, which themselves
    generate more offspring.

    Alternatively, the generating process can be reframed as
    a mixture of independent Poisson processes, namely the
    immigrant-generating background rate and past arrivals’
    time-decaying excitatory influence. The Hawkes process
    therefore lends itself to a branching structure, whereby
    each arrival is spawned by a single generating process in
    the Poissonian mixture.

    For the purposes of this class, we consider the Hawkes
    process with exponential memory kernel and time-varying
    background rate.
            
    Attributes
    ----------
    t : ndarray of floats, shape=(N,)
        Array of event arrival times.
        
    u : ndarray of ints, shape=(N,)
        Array of event dimensions.
        
    phi_t : ndarray of floats, shape=(N,)
        Time-varying factor applied to event arrival times.
        
    phi_avg : float
        Average of time-varying factor on interval [0,T).
        
    U : int
        Number of dimensions in Hawkes PP.
        
    N : int
        Number of events in Hawkes PP.
        
    T : float
        Hawkes PP event horizon time.
        
    N_dim : int
        Number of events in Hawkes PP along each dimension.
    """
    
    def _simulate_data(self,params,N,U):
        """Simulate events in Hawkes PP with variable
        background rate.
        """
        
        # extract parameters
        mu,alpha,beta,bkgd_params = params
        phi,phi_max,phi_avg = bkgd_params
        
        # create arrival time and dimension arrays
        t = np.zeros(N)
        u = np.zeros(N,dtype=np.int8)
        phi_t = np.zeros(N)
        
        # initialize variables and arrays
        ti = 0
        count = 0
        loop = 0
        summand = np.zeros(U)
        
        # cache some arrays and matrices
        beta_alpha = beta*alpha
        sum_baseline = phi_max*np.sum(mu)

        # cache randomly generate uniform r.v's (2 times
        # as many as needed to allow for rejection tolerance)
        log_rands = -np.log(np.random.uniform(size=N*2))
        D = np.random.uniform(size=N*2)

        # Ogata-motivated thinning algorithm for multivariate
        # Hawkes PP simulation
        with tqdm(total=N) as pbar:
            while count < N:
                # generate exponential arrival parametrized by maximum
                # possible interarrival Hawkes intensity
                lam_bar = sum_baseline + np.sum(summand)
                w = log_rands[loop]/lam_bar
                summand *= np.exp(-beta*w)
                ti += w

                # generate uniform and accept with probability
                # sum of cumsum divided by upper bound lam_bar
                phi_ti = phi(ti)
                lam_arr = phi_ti*mu + summand
                p = D[loop]*lam_bar
                cumsum = np.cumsum(lam_arr)
                if p <= cumsum[-1]:
                    # assign dimension
                    k = 0
                    while p > cumsum[k]:
                        k += 1
                    # store generated event
                    t[count] = ti
                    u[count] = k
                    phi_t[count] = phi_ti
                    summand += beta_alpha[:,k]
                    count += 1
                    pbar.update(1)
                # advance loop attempts
                loop += 1
                
        # normalize static background rate and mask by dimension
        phi_t /= phi_avg
        self.phi_avg = phi_avg
        self.phi_t = [phi_t[u==i] for i in range(U)]
        return t,u
    
    def load_data(self,t,u,bkgd_params):
        """Load existing point process event data.
        
        Parameters
        ----------
        t : ndarray of floats, shape=(,)
            Array of event arrival times. Must be in increasing
            order.
        
        u : ndarray of ints, shape=(,)
            Array of event dimensions. Must take on all integer
            values in the range [0,max_val), where max_val is
            the maximum value of elements in the array.
            
        bkgd_params : tuple
            Tuple of time-varying factor parameters (phi,phi_max,
            phi_avg), where phi is the function, phi_max is its
            maximal value over [0,T), and phi_avg is its average
            over the same interval.
        """
        
        # call parent load_data method
        super().load_data(t,u)
        
        # tack on time-varying attributes
        phi,phi_max,phi_avg = bkgd_params
        phi_t = np.array([phi(ti) for ti in tqdm(t)])/phi_avg
        self.phi_avg = phi_avg
        self.phi_t = [phi_t[u==i] for i in range(self.U)]
        
    def _EM(self,beta,eps,inc,verbose):
        # call parent _EM method
        mu,alpha,loglike = super()._EM(beta,eps,inc,verbose)
        
        # normalize background rate by average of time-varying
        # factor
        mu = mu/self.phi_avg
        return mu,alpha,loglike
    
    def _get_base_rate(self,mu,i):
        """Return base rate in E-step"""
        # use time_varying factor
        base_rate = mu[i]*self.phi_t[i]
        return base_rate