

```python
from hawkes_estimator import *
from numpy.random import uniform
from numpy import fill_diagonal,sqrt,ndarray
import matplotlib.pyplot as plt
```

# Multivariate Hakwes Process

### Generate Example Hawkes Parameters


```python
DIMS = 4 # num dimensions
BACKGROUND_RATE_BOUNDS = [0.4,0.6] # mu bounds
SELF_EXCITATION_RATE_BOUNDS = [0.3,0.4] # alpha matrix diagonal bounds
MUTUAL_EXCITATION_RATE_BOUNDS = [0.05,0.1] # alpha matrix non-diagonal bounds
DECAY_BOUNDS = [0.5,0.6] # beta bounds
```


```python
# generate random parameters given above bounds
def generate_params(U,mu_bnd,alpha_diag_bnd,alpha_nondiag_bnd,beta_bnd):
    mu = uniform(mu_bnd[0],mu_bnd[1],U)
    alpha = uniform(alpha_nondiag_bnd[0],alpha_nondiag_bnd[1],(U,U))
    fill_diagonal(alpha,uniform(alpha_diag_bnd[0],alpha_diag_bnd[1],U))
    beta = uniform(beta_bnd[0],beta_bnd[1])
    return mu,alpha,beta
```


```python
mu,alpha,beta = generate_params(DIMS,BACKGROUND_RATE_BOUNDS,SELF_EXCITATION_RATE_BOUNDS,MUTUAL_EXCITATION_RATE_BOUNDS,DECAY_BOUNDS)
```

### Simulate Hawkes Process


```python
hwk = HawkesEM()
```


```python
hwk.simulate_data(params=(mu,alpha,beta),N=100000)
# hwk.load_data(t,u) <- if importing data from elsewhere
```


```python
hwk.plot_events(horizon=100)
```


![png](README_files/README_9_0.png)


### Estimate Hawkes Parameters (Known Decay)


```python
mu_hat,alpha_hat = hwk.fit(beta,eps=0.001,inc=20,verbose=True)
```

    Preparation ->   0.07s ----------------
    Epoch    20 ->   0.13s ----------------
    Epoch    40 ->   0.18s - diff: 1.29e+01
    Epoch    60 ->   0.25s - diff: 2.92e+00
    Epoch    80 ->   0.32s - diff: 9.64e-01
    Epoch   100 ->   0.40s - diff: 3.91e-01
    Epoch   120 ->   0.45s - diff: 1.84e-01
    Epoch   140 ->   0.51s - diff: 9.44e-02
    Epoch   160 ->   0.57s - diff: 4.95e-02
    Epoch   180 ->   0.62s - diff: 2.55e-02
    Epoch   200 ->   0.69s - diff: 1.28e-02
    Epoch   220 ->   0.75s - diff: 6.18e-03
    Epoch   240 ->   0.80s - diff: 2.92e-03
    Epoch   260 ->   0.87s - diff: 1.34e-03
    Epoch   280 ->   0.92s - diff: 6.07e-04
    


```python
def RMSE(pred,actual):
    return sqrt(((pred-actual)**2).mean())
```


```python
mu_RMSE = RMSE(mu,mu_hat)
alpha_RMSE = RMSE(alpha,alpha_hat)
```


```python
print("Background Rates RMSE = {}".format(round(mu_RMSE,2)))
print("Excitation Rates RMSE = {}".format(round(alpha_RMSE,2)))
```

    Background Rates RMSE = 0.02
    Excitation Rates RMSE = 0.01
    


```python
params = mu_hat,alpha_hat,beta
hwk.plot_intensities(params,horizon=200)
```


![png](README_files/README_15_0.png)


### Estimate Hawkes Parameters (Unknown Decay)


```python
mu_hat,alpha_hat,beta_hat = hwk.fit(eps=0.001,inc=20,verbose=True)
```

    beta =  0.0000 ->   0.00s - loglike: -1.000000e+06
    beta =  1.0000 ->   0.65s - loglike: -7.983871e+04
    beta =  2.0000 ->   1.12s - loglike: -8.048171e+04
    beta =  1.5000 ->   1.62s - loglike: -8.018769e+04
    beta =  0.7500 ->   2.30s - loglike: -7.967384e+04
    beta =  0.5000 ->   3.08s - loglike: -7.960321e+04
    beta =  0.2500 ->   4.36s - loglike: -7.994111e+04
    beta =  0.3750 ->   5.29s - loglike: -7.967855e+04
    beta =  0.5625 ->   5.94s - loglike: -7.960249e+04
    beta =  0.5938 ->   6.57s - loglike: -7.960809e+04
    beta =  0.5469 ->   7.63s - loglike: -7.960105e+04
    beta =  0.5312 ->   8.28s - loglike: -7.960062e+04
    beta =  0.5156 ->   8.92s - loglike: -7.960131e+04
    beta =  0.5234 ->   9.50s - loglike: -7.960082e+04
    beta =  0.5352 ->  10.14s - loglike: -7.960063e+04
    beta =  0.5293 ->  10.98s - loglike: -7.960065e+04
    beta =  0.5323 ->  11.79s - loglike: -7.960061e+04
    beta =  0.5333 ->  12.51s - loglike: -7.960061e+04
    beta =  0.5343 ->  13.24s - loglike: -7.960063e+04
    


```python
mu_RMSE = RMSE(mu,mu_hat)
alpha_RMSE = RMSE(alpha,alpha_hat)
beta_RMSE = RMSE(beta,beta_hat)
```


```python
print("Background Rates RMSE = {}".format(round(mu_RMSE,2)))
print("Excitation Rates RMSE = {}".format(round(alpha_RMSE,2)))
print("Decay Rate RMSE = {}".format(round(alpha_RMSE,2)))
```

    Background Rates RMSE = 0.02
    Excitation Rates RMSE = 0.01
    Decay Rate RMSE = 0.01
    


```python
params = mu_hat,alpha_hat,beta_hat
hwk.plot_intensities(params,horizon=200)
```


![png](README_files/README_20_0.png)


# Multivariate Hakwes Process with Time-Varying Background Rate

### Generate Example Time-Varying Hawkes Parameters


```python
# using mu, alpha, beta from above example**
# example time-varying background rate function
freqs = np.array([100,200])/(2*np.pi)
amp = np.array([0.3,0.3])
offset = 100
def f(x):
    def fi(x):
        return np.sum(amp*np.cos((x-offset)/freqs)) + np.sum(amp)
    if type(x) != ndarray:
        return fi(x)
    else:
        return np.array([fi(xi) for xi in x])
```


```python
# background rate average and maximum
x = np.arange(0,1000,0.1)
favg = np.mean(f(x))
fmax = np.max(f(x))*1.1
bkgd_params = (f,fmax,favg)
```


```python
# plot time-varying background rate
fig,ax = plt.subplots()
ax.plot(x, f(x), color='blue')
ax.set_title('Time-Varying Background Rate')
ax.set_xlabel('Time')
ax.set_ylabel('$\phi(t)$')
fig.tight_layout()
```


![png](README_files/README_25_0.png)


### Simulate Hawkes Process


```python
hwk = TimeVarHawkesEM()
```


```python
hwk.simulate_data(params=(mu,alpha,beta,bkgd_params),N=100000)
# hwk.load_data(t,u,bkgd) <- if importing data from elsewhere
```

    100%|████████████████████████████████████████████████████████████████████████████████| 100000/100000 [00:18<00:00, 5547.04it/s]
    


```python
hwk.plot_events(horizon=200)
```


![png](README_files/README_29_0.png)


### Estimate Hawkes Parameters (Known Decay)


```python
mu_hat,alpha_hat = hwk.fit(beta,eps=0.001,inc=20,verbose=True)
```

    Preparation ->   0.04s ----------------
    Epoch    20 ->   0.12s ----------------
    Epoch    40 ->   0.18s - diff: 1.10e+01
    Epoch    60 ->   0.25s - diff: 1.82e+00
    Epoch    80 ->   0.32s - diff: 7.58e-01
    Epoch   100 ->   0.38s - diff: 4.52e-01
    Epoch   120 ->   0.45s - diff: 2.13e-01
    Epoch   140 ->   0.52s - diff: 7.33e-02
    Epoch   160 ->   0.57s - diff: 2.06e-02
    Epoch   180 ->   0.63s - diff: 5.17e-03
    Epoch   200 ->   0.70s - diff: 1.24e-03
    Epoch   220 ->   0.75s - diff: 2.87e-04
    


```python
mu_RMSE = RMSE(mu,mu_hat)
alpha_RMSE = RMSE(alpha,alpha_hat)
```


```python
print("Background Rates RMSE = {}".format(round(mu_RMSE,2)))
print("Excitation Rates RMSE = {}".format(round(alpha_RMSE,2)))
```

    Background Rates RMSE = 0.02
    Excitation Rates RMSE = 0.01
    


```python
params = mu_hat,alpha_hat,beta
hwk.plot_intensities(params,horizon=200)
```


![png](README_files/README_34_0.png)


### Estimate Hawkes Parameters (Unknown Decay)


```python
mu_hat,alpha_hat,beta_hat = hwk.fit(eps=0.001,inc=20,verbose=True)
```

    beta =  0.0000 ->   0.00s - loglike: -1.000000e+06
    beta =  1.0000 ->   0.50s - loglike: -1.174311e+05
    beta =  2.0000 ->   0.90s - loglike: -1.185477e+05
    beta =  1.5000 ->   1.36s - loglike: -1.180430e+05
    beta =  0.7500 ->   1.91s - loglike: -1.171316e+05
    beta =  0.5000 ->   2.49s - loglike: -1.169806e+05
    beta =  0.2500 ->   3.44s - loglike: -1.175616e+05
    beta =  0.3750 ->   4.03s - loglike: -1.170928e+05
    beta =  0.5625 ->   4.60s - loglike: -1.169883e+05
    beta =  0.4688 ->   5.21s - loglike: -1.169895e+05
    beta =  0.5156 ->   5.87s - loglike: -1.169796e+05
    beta =  0.5312 ->   6.45s - loglike: -1.169807e+05
    beta =  0.5234 ->   7.02s - loglike: -1.169799e+05
    beta =  0.5117 ->   7.59s - loglike: -1.169796e+05
    beta =  0.5176 ->   8.20s - loglike: -1.169796e+05
    beta =  0.5146 ->   8.75s - loglike: -1.169796e+05
    


```python
mu_RMSE = RMSE(mu,mu_hat)
alpha_RMSE = RMSE(alpha,alpha_hat)
beta_RMSE = RMSE(beta,beta_hat)
```


```python
print("Background Rates RMSE = {}".format(round(mu_RMSE,2)))
print("Excitation Rates RMSE = {}".format(round(alpha_RMSE,2)))
print("Decay Rate RMSE = {}".format(round(alpha_RMSE,2)))
```

    Background Rates RMSE = 0.02
    Excitation Rates RMSE = 0.01
    Decay Rate RMSE = 0.01
    


```python
params = mu_hat,alpha_hat,beta_hat
hwk.plot_intensities(params,horizon=200)
```


![png](README_files/README_39_0.png)

