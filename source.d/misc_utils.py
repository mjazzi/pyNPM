
################################################################################
# Utilities to perform signal processing / compute distances  
################################################################################

import numpy as np
import scipy.stats

################################################################################
# Pre-process data for wasserstein distance: 
#   find scaling constant c and normalize	
def preProcessWasserstein(mean_ref, std_ref=np.array([]), mean_shift=None): 
  try:
    import jax.numpy as jnp 
    if not(mean_shift): 
      mean_shift = 1.8*jnp.abs(jnp.min(mean_ref))
    shifted_mean_ref = mean_ref +	mean_shift  
    if mean_ref.ndim <=2:
      norm_mean_ref= shifted_mean_ref/jnp.sum(shifted_mean_ref, axis=1).reshape(mean_ref.shape[0],1)  
    else: 
      norm_mean_ref= shifted_mean_ref/jnp.sum(shifted_mean_ref, axis=1).reshape(mean_ref.shape[0],1,mean_ref.shape[2])  
    norm_std_ref=[]    
    if std_ref.any():
      if std_ref.ndim <=2: 
        norm_std_ref = std_ref/jnp.sum(std_ref, axis=1).reshape(std_ref.shape[0], 1)  
      else: 
        norm_std_ref = std_ref/jnp.sum(std_ref, axis=1).reshape(std_ref.shape[0], 1, std_ref.shape[2])
    return norm_mean_ref, norm_std_ref, mean_shift 
  except ImportError:
    print('Module jax.numpy is not available - aborting.')
    exit()

                                     
################################################################################
# Obtain normalizing constants when using wasserstein distance 
def computeWassersteinNormalizingConstants(norm_mean_ref, norm_std_ref, t):
  try:
    import ot
    c_mean = np.zeros(norm_mean_ref.shape[0::2])
    c_std = np.zeros(norm_std_ref.shape[0::2])
    for i in range(norm_mean_ref.shape[2]):
        c_mean[:,i] = ot.wasserstein_1d(np.tile(t.reshape(-1,1), (1,norm_mean_ref.shape[0])), np.tile(t.reshape(-1,1), (1,norm_mean_ref.shape[0])),(1/len(t))* np.ones(norm_mean_ref.shape[:2]).T, norm_mean_ref[:,:,i].T,p=2)
        c_std[:,i] = ot.wasserstein_1d(np.tile(t.reshape(-1,1), (1,norm_mean_ref.shape[0])), np.tile(t.reshape(-1,1), (1,norm_mean_ref.shape[0])),(1/len(t))* np.ones(norm_mean_ref.shape[:2]).T, norm_std_ref[:,:,i].T,p=2)
    return c_mean, c_std
  except ImportError:
    print('Module ot is not available - aborting.')
    exit()

############################################################################### 
# Automate width of target stdev intervals 
def computeFluctuations(signal_ref, rom_ref):
  N = signal_ref.shape[0]
  sigma = np.zeros(signal_ref.shape)
  qualityFlag = True
  for i in range(signal_ref.shape[1]):
      incr = (abs(signal_ref[:,i]-rom_ref[:,i]))/50
      to_modify = set(np.arange(N))
      sigma_col_prev = np.zeros((N))
      sigma_col = np.zeros((N))
      exact_indices = np.where(signal_ref[:,i] == rom_ref[:,i])[0]
      sigma_col[exact_indices] = 0
      to_modify = to_modify.difference(set(exact_indices))
      while not(len(to_modify)==0):
          lb_o = scipy.stats.norm.ppf(0.025, loc=signal_ref[:,i], scale=sigma_col)
          ub_o = scipy.stats.norm.ppf(0.975, loc=signal_ref[:,i], scale=sigma_col)
          belong = (lb_o < rom_ref[:,i]) * (rom_ref[:,i] < ub_o)
          for j in to_modify:
              if belong[j]:
                  to_modify = to_modify.difference(set([j]))
              else:
                  sigma_col[j] = sigma_col_prev[j]+incr[j]
          sigma_col_prev = sigma_col
      if (sigma_col == incr).any():
          qualityFlag = False 
      sigma[:,i] = sigma_col
  return sigma, qualityFlag
