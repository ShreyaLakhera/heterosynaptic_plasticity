import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import sys
import json

def gN(vm):
    return alphaN*vm + betaN

def gV(vm):
    return alphaV*vm 

def df_dt(U, t, k):

    # to-do: convert the for loop into matrix form!

    # U = states of all neurons: [Vm_0, Vm_1, ..., Vm_N, Ca_0, Ca_1, ..., Ca_N, Y_0, Y_1, ..., Y_N, W_0, W_1, ..., W_N] 
    # Vm_i = membrane potential at i-th spine
    # Ca_i = Calcium concentration at i-th spine
    # Y_i = Interim weight of synapse at i-th spine
    # W_i = Weight of synapse at i-th spine

    # k = constants: []

    dU_dt = []
    [N, tau_m, tau_c, tau_A, tau_N, tau_BP, tau_I, tau_E, tau_y, gammaA, gammaN, gammaBP, gammaE, gammaI, C_p, C_d, Ca_threshold_p, Ca_threshold_d, B_p, B_d, y_threshold, AM, xI_dI, xE_dE, spikes] = k

    for _i in range(N):
        _vm = U[]
        _ca = U[]
        _xA = U[]
        _xN = U[]
        _xBP = U[]
        _xI = U[]
        _xE = U[]
        _y = U[]
        _w = U[]

	# sum of inh inputs
        _sigI = 0.
	for _j in np.where(AM[_i,:]>0)[0]:
            _sigI += xI_dI[_j] 

	# sum of exc inputs
        _sigE = 0.
	for _j in np.where(AM[_i,:]>0)[0]:
            _sigE += xE_dI[_j] 

        # del(t - spike)
	_spk = spikes[_i]
        if t in _spk:
            _del_spk = 1.
        else:
            _del_spk = 0.
        
        _dvm = -(_vm/tau_m) + gammaA*(_xA) + gammaN*gN(_vm)*_xN + gammaBP*_xBP - gammaI*sigI + gammaE*sigE
        _dca = -(_ca/tau_c) + gN(_vm)*_xN + gV(_vm)
        _dxA = -(_xA/tau_A) + _del_spk 
        _dxN = -(_xN/tau_N) + _del_spk
        _dxBP = -(_xBP/tau_BP) + _del_spk
        _dxI = -(_xI/tau_I) + _del_spk
        _dxE = -(_xE/tau_E) + _del_spk
        _dy = -(_y/tau_y) + C_p*np.sign(_ca - Ca_threshold_p) - C_d*np.sign(_ca - Ca_threshold_d)
        _dw = B_p*np.sign(_y - y_threshold) - B_d*np.sign(-_y - y_threshold)
 
    return

def test_f(U, k):
    # just exp decay for testing the integrator
    [t, tau, spk] = k
    dspk = 0.
    if t in spk:
        dspk = 1.
    dU = - (U/tau) + dspk 
    return dU

N = 2 #number of spines

tau_m = 
tau_c = 
tau_A = 
tau_N = 
tau_BP = 
tau_I = 
tau_E = 
tau_y = 

gamma_A = 
gamma_N = 
gamma_BP = 
gamma_I = 
gamma_E = 

C_p = 
C_d = 
B_p = 
B_d = 
Ca_threshold_p = 
Ca_threshold_d = 
y_threshold = 

dt = 0.1 # time step (ms)
t_sim = 200.

spiketimes = np.array([])
U0 = np.zeroes((9*N, 1))
ts = np.arange(0., t_sim, dt)
k = [N, tau_m, tau_c, tau_A, tau_N, tau_BP, tau_I, tau_E, tau_y, gammaA, gammaN, gammaBP, gammaE, gammaI, C_p, C_d, Ca_threshold_p, Ca_threshold_d, B_p, B_d, y_threshold, AM, xI_dI, xE_dE, spiketimes]

U = spi.odeint(df_dt, U0, ts, k)
print(np.shape(U)) # should be (9*N, t_sim/dt)

# plot
