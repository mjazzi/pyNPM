import numpy as np
import os
import sys
import time
import re
import multiprocessing
from joblib import Parallel, delayed
from scipy.io import mmread, loadmat
from scipy.optimize import minimize, Bounds, basinhopping, brute

from matplotlib import pyplot as plt

from NodeProcessor import NodeProcessor
from npm import GtoW2, generateG
from aeros_utils import file_format, new_read_dydv_files, referenceValues, readROBPlus

################################################################################
# Read PBS information to initialize the list of nodes and number of cores
def getPBSInfo():
  filename = os.getenv('PBS_NODEFILE')
  if filename == None:
    return 0, []
  f = open(filename, 'r')
  nodesn = a=f.readlines()
  unodesn = list(set(nodesn))
  unodes = []
  for unoden in unodesn:
    unodes.append(unoden.strip())
  ppn = int(os.getenv('PBS_NUM_PPN'))
  return ppn, unodes

################################################################################
# Read SLURM information to initialize the list of nodes and number of cores
def getSLURMInfo():
  unodes  = os.getenv('SLURM_JOB_NODELIST')
  if unodes == None:
    return 0, []
  if isinstance(unodes, str):
    nodes = [unodes]
  else:
    nodes = list(unodes)
  nodenames=[]
  for node in nodes:
    if node.find('[')!=-1:
      idx = node.find('[')
      prefix = node[:idx]
      numbers = re.findall(r'\d+',node[idx+1:-1])
      a=re.findall(r'\d+-\d+',node[idx+1:-1])
      if a:
        for ran in a:
          bounds = re.findall(r'\d+', ran)
          interval = np.arange(int(ran[:2]), int(ran[-2:])+1)
          for num in interval:
            numbers+=[str(num).zfill(2)]
          numbers = list(set(numbers))
      for number in numbers:
        nodenames += [(prefix + number)]
    else:
      nodenames+=[node]
  ppn = int(os.getenv('SLURM_CPUS_ON_NODE'))
  return ppn, nodenames

################################################################################
# Initialization of the hyperparameters and their bounds
def initHyperParameters(Gparams, stochasticParameters):

  # Initialization of the hyperparameters
  s_0 = stochasticParameters.s_0
  beta_0 = stochasticParameters.beta_0
  m_sigma   = int(Gparams.n * (Gparams.n + 1) / 2) # hyperparameters of sigma
  Rrow = np.zeros(m_sigma, dtype = int) # rows of the entries in sigma
  Rcolumn = np.zeros(m_sigma, dtype = int) # columns of the entries in sigma
  Rsigma_0 = np.zeros(m_sigma) # values of sigma stored in a vector
  sigma_0_full = stochasticParameters.diag_coef * np.identity(Gparams.n)
  nbsigma = m_sigma
  ind = 0
  for j in range(Gparams.n):
    for k in range(j, Gparams.n):
      Rrow[ind] = j
      Rcolumn[ind] = k
      Rsigma_0[ind] = sigma_0_full[j,k]
      ind = ind + 1
  
  # Initialization of the hyperparameters vector alpha_0
  m_alpha = nbsigma + 2
  alpha_0 = np.zeros(m_alpha)
  alpha_0[0:nbsigma] = Rsigma_0
  alpha_0[nbsigma] = s_0
  alpha_0[nbsigma+1] = beta_0

  # Initialization of the hyperparameter bounds
  m_alpha = nbsigma + 2
  lb = np.zeros(m_alpha)
  ub = np.zeros(m_alpha)
  diag_min = stochasticParameters.diag_min
  extra_diag_min = stochasticParameters.extra_diag_min
  diag_max = stochasticParameters.diag_max
  extra_diag_max = stochasticParameters.extra_diag_max
  for ind in range(nbsigma):
    j = Rrow[ind]
    k = Rcolumn[ind]
    if j == k:
      ll = diag_min
      uu = diag_max
    if j != k:
      ll = extra_diag_min
      uu = extra_diag_max
    lb[ind] = ll
    ub[ind] = uu
  lb[nbsigma] = stochasticParameters.lb_s  
  ub[nbsigma] = stochasticParameters.ub_s 
  lb[nbsigma + 1] = stochasticParameters.lb_beta 
  ub[nbsigma + 1] = stochasticParameters.ub_beta 
  return Rrow, Rcolumn, alpha_0, lb, ub

################################################################################
# Initialization of the parameters of matrix G 
def initGparams(Vshape, x,  Gparams):
  # Initialize parameters needed for G (random matrix) - notation follows appendix D
  # of Farhat and Soize (2017) 
  Gparams.n = Vshape[1] 
  Gparams.N0 = int(Vshape[0] / Gparams.m)  
  Gparams.N = Gparams.m * Gparams.N0 

  for i in range(Gparams.d):
    Gparams.L[i] = np.ptp(x[:, i])

  sum_nu = np.zeros(Gparams.nu_p)
  for nu in range(Gparams.nu_p):
    if Gparams.K_nu[nu] < 0:
      kap = - Gparams.K_nu[nu]
    else:
      kap =  Gparams.K_nu[nu]
    sum_nu[nu] = 2 * (1 - kap) / Gparams.nu_p
  Gparams.R2sum_nu = np.sqrt(2 * sum_nu)
  return Gparams

################################################################################
# Convert vector alpha into hyperparameters sigma, s, beta
def alphaToHyperparameters(alpha, n, Rrow, Rcolumn):
  nbsigma = len(Rrow)
  sigma = np.zeros((n, n))
  sigma[Rrow, Rcolumn] = alpha[0:nbsigma]
  s     = alpha[nbsigma]
  beta  = alpha[nbsigma+1]
  return sigma, s, beta

################################################################################
class RunParameters:
  def __init__(self):
    self.exec_nodes = [] # Names of nodes that have been allocated for the job
    self.ncores_per_node = 0
    # Read the PBS info to be used to run ROM realizations using Aero-s
    self.ncores_per_node, self.exec_nodes = \
      getPBSInfo()
    if self.ncores_per_node == 0:
      # Alternatively, read the Slurm info 
      self.ncores_per_node, self.exec_nodes = \
        getSLURMInfo()
 
################################################################################
# Define an auxiliary object for the cost function optimization
class ObjectiveF:
  def __init__(self, N_sim, rng, costFunction):
    self.N_sim = N_sim
    self.rng = rng
    self.costFunction = costFunction
    self.counter = int(0)
    self.batch_size = N_sim

  def objective(self, alpha):
    cycle = self.counter % (self.N_sim / self.batch_size)
    batch = self.batches[np.arange(self.batch_size * cycle, \
                         self.batch_size * (cycle+1), dtype = int)]
    return self.costFunction(alpha, batch)

  def callback(self, params):
    print("Iteration %d" % self.counter)
    print("alpha =  {}".format(params), flush = True)
    return

  def setBatch(self, batch_size):
    self.batch_size = batch_size
    self.batches = self.rng.permutation(self.N_sim)
   
  def callback_sgd(self, params):
    print("Iteration %d" % self.counter)
    print("alpha =  {}".format(params), flush = True)
    self.counter = self.counter + 1
    if self.counter % self.batch_size == 0:
      # Reset batches
      self.batches = self.rng.permutation(self.N_sim)
    return

  def callback_sgd_tr(self, params, optim):
    print("Iteration %d" % self.counter)
    print("alpha =  {}".format(params), flush = True)
    self.counter = self.counter + 1
    if self.counter % self.batch_size == 0:
      # Reset batches
      self.batches = self.rng.permutation(self.N_sim)
    return

################################################################################
# Cleanup temporary files on a compute node
def cleanNode(node):
  command = 'ssh -o ForwardX11=no %s "rm -rf /dev/shm/shrom*"' % node
  os.system(command)

################################################################################
# Function to convert results of Parallel (list) to arrays 
def convertParallelListToArray(res, ixDim):
  ni = len(ixDim)
  na = 0
  for i in range(len(res)):
    if len(res[i]) > na:
      na = len(res[i])
      ix = i
  resA = []
  for ii in range(na - ni):
    if len(ixDim) == 1:
      resA.append(np.zeros(((ixDim[0],) + res[ix][ii + ni].shape)))
    else:
      resA.append(np.zeros(((ixDim[0], ixDim[1], ) + res[ix][ii + ni].shape)))
  print('len(resA): ', len(resA))
  for i in range(len(res)):
    for ii in range(len(res[i]) - ni):
      if len(ixDim) == 1:
        resA[ii][res[i][0]] = res[i][ii + ni]
      else:
        resA[ii][res[i][0], res[i][1]] = res[i][ii + ni]
  return tuple(resA)

################################################################################
class SROMBase:

  def __init__(self, problemPath):
    # Specify directory that contain problem-specific input_dir and output files
    self.problemPath = problemPath
    sys.path.append(problemPath)
    # Initialize problem dependent parameters
    from parameters import ProblemParameters, IOparameters, \
                           GParams, StochasticParameters
    self.IOParameters = IOparameters()
    self.problemParameters = ProblemParameters(self.IOParameters)
    self.stochasticParameters = StochasticParameters()
    self.Gparams = GParams()

    self.runParameters = RunParameters()
    return

################################################################################
# Make SROM realizations and write the corresponding Aero-S input files
# Standalone function as Parallel supposedly packages the whole object otherwise
def makeSROMAeroS(j, problemParameters, IOParameters, Gparams, 
  ArrayRand1IDENT, ArrayRand2IDENT, sigma, s, beta, 
  probed_dofs, root, proot, croot2, batch):

  from aeros_runs import aeros_buildROM, aeros_buildHROM

  print('batch', batch, flush = True)

  derivative_flag = len(np.nonzero(batch == j)[0]) > 0 and \
                   problemParameters.derivative_flag
  if not derivative_flag:
    G = generateG(Gparams, beta,
        problemParameters.x, problemParameters.Dof, problemParameters.Regular, 
                  ArrayRand1IDENT[:,:,:,:,j], ArrayRand2IDENT[:,:,:,:,j])
    WW = GtoW2(G, sigma, s, problemParameters.V, problemParameters.B)

    if len(problemParameters.R) !=0:
       W = WW @ problemParameters.R
    else:
       W = WW

    Vk_unc = W[problemParameters.uncdof,:]
    Vk_probed = W[probed_dofs,:]
    Vk_full = W
    res = (j, Vk_unc, Vk_probed, Vk_full)
  else:
    G, dGdB  = generateG(Gparams, beta,
         problemParameters.x, problemParameters.Dof, problemParameters.Regular, 
         ArrayRand1IDENT[:,:,:,:,j], ArrayRand2IDENT[:,:,:,:,j], True)
    WW, dVkda_s_, dVkda_B_, dVkda_Sig_ = GtoW2(G, sigma, s,
         problemParameters.V, problemParameters.B, dGdB)

    if len(problemParameters.R) !=0:
       W = WW @ problemParameters.R
       dVkda_s = dVkda_s_ @ problemParameters.R
       dVkda_B = dVkda_B_ @ problemParameters.R
       dVkda_Sig = np.einsum('ija,jb->iba', dVkda_Sig_, problemParameters.R);
    else:
       W = WW
       dVkda_s = dVkda_s_
       dVkda_B = dVkda_B_
       dVkda_Sig = dVkda_Sig_

    # componentwise derivatives
    Vk_unc = W[problemParameters.uncdof,:]
    Vk_probed = W[probed_dofs,:]
    Vk_full = W

    dVkds_unc = dVkda_s[problemParameters.uncdof, :]
    dVkdB_unc = dVkda_B[problemParameters.uncdof, :]
    dVkdSig_unc = dVkda_Sig[problemParameters.uncdof, :, :]

    dVkds_probed = dVkda_s[probed_dofs, :]
    dVkdB_probed = dVkda_B[probed_dofs, :]
    dVkdSig_probed = dVkda_Sig[probed_dofs,:, :]
    res = (j, Vk_unc, Vk_probed, Vk_full,
                dVkds_unc, dVkdB_unc, dVkdSig_unc,
                dVkds_probed, dVkdB_probed, dVkdSig_probed)
    
  for i in range(problemParameters.mu.shape[0]):
    k = i * problemParameters.N_sim + j 
    # Storage directory for k-th realization
    croot = '%s/%d' % (root, k)
    paths = [proot, croot, croot2]
    os.mkdir(croot)
    
    # Write SROM - use the same file for both sampled and unsampled
    fid = open('%s/SROB_dir.orthonormalized.out' % croot, 'w')
    print('%d\n%d' % (Gparams.n, len(problemParameters.ROB_nodes)), file = fid)
    np.savetxt(fid, problemParameters.ROB_nodes+ 1, fmt = '  %d')
    for ii in range(Gparams.n):
      fid.write('   %e\n' % problemParameters.ROB_tag[ii])
      if Gparams.m == 3:
        cW = np.reshape(W[:, ii], (-1, 3), order = 'C')
        np.savetxt(fid, np.hstack((cW, np.zeros(cW.shape))))
      else:
        np.savetxt(fid, np.reshape(W[:, ii], (-1, 6), order = 'C'))
    fid.close()
    
    # Generate input_dir files and a batch file to run them
    if not problemParameters.hyperreduction_flag:
      aeros_buildROM(problemParameters.mu[i,:], problemParameters, IOParameters, paths)
      
      # Create batch files
      f = open('%s/batch' % croot,'w')
      f.write('touch  %s/start\n' % croot2)
      f.write('%s e %s/%s.out >& %s/%s\n' % (IOParameters.rob_exec, croot2,
               IOParameters.SROB_dir,croot2, IOParameters.ROB_log_file))
      f.write('mv %s/%s.out.out %s/%s\n' % (croot2, IOParameters.SROB_dir,
                                             croot2,IOParameters.SROB_dir))
      f.write('%s -v 2  %s/%s >& %s/%s\n' % (IOParameters.program_exec, croot2,
            IOParameters.ROM_output, croot2,IOParameters.ROM_log_file))
      f.write('cp %s/dis* %s/%d \n' % (croot2, root, k))
      f.write('cp %s/acc* %s/%d \n' % (croot2, root, k))
      f.write('cp %s/vel* %s/%d \n' % (croot2, root, k))
      f.write('cp %s/log* %s/%d \n' % (croot2, root, k))
      f.write('touch  %s/done%d\n' % (root,k))
      f.close()
    else:
      if problemParameters.sampled_mesh_flag:
        os.system('cp %s/Mesh/samplmsh.new%d.elementmesh.inc %s/samplmsh.new.elementmesh.inc '% (IOParameters.output_ID, i, croot))
      else:
        aeros_spnnlsnew(Gparams.n, problemParameters.mu[i,:],problemParameters, IOParameters, paths)

      aeros_buildHROM(problemParameters.mu[i,:], problemParameters, IOParameters,
                      derivative_flag, k, paths)
      f = open('%s/batch' % croot, 'w')
      f.write(' touch  %s/start\n' % croot2)
      f.write('%s e %s/%s.out >& %s/%s\n' % (IOParameters.rob_exec, croot2, IOParameters.SROB_dir, croot2, IOParameters.ROB_log_file))
      f.write('mv %s/%s.out.out %s/%s\n' % (croot2, IOParameters.SROB_dir, croot2, IOParameters.SROB_dir))
      
      if not problemParameters.sampled_mesh_flag:
        f.write('%s -v 2  %s/%s >& %s/%s\n' % (IOParameters.program_exec,
         croot2, IOParameters.spnnls_output_2, croot2, IOParameters.spnnls_log_file_2))
      f.write('%s -v 2  %s/%s >& %s/%s\n' % (IOParameters.program_exec,
         croot2, IOParameters.HROM_output, croot2, IOParameters.HROM_log_file))
      f.write('cp %s/dis* %s/%d \n' % (croot2, root, k))
      f.write('cp %s/acc* %s/%d \n' % (croot2, root, k))
      f.write('cp %s/vel* %s/%d \n' % (croot2, root, k))
      f.write('cp %s/log* %s/%d \n' % (croot2, root, k))
      f.write(' touch  %s/done%d\n' % (root, k))
      f.close()
  return res 

################################################################################
# Post-process SROM realization results
# Standalone function as Parallel packages the whole object otherwise
def postProAeroS(i, j, filename, filename2, derivative_flag, linear_flag, si,
            Vk_probed,
            dVkds_unc, dVkdB_unc, dVkdSig_unc,
            dVkds_probed, dVkdB_probed, dVkdSig_probed,
            n, N, uncdof):
  f = open(filename, 'r')
  f.readline()
  ncoord = int(f.readline())
  # vanilla Python code - something faster needed
  array = []
  for line in f:
    for x in line.split():
      array.append(x)
  v = np.array(array, dtype=float)
  f.close()
  if not linear_flag:
    coord = np.reshape(v, (ncoord + 1, -1), order = 'F')
    sol = (Vk_probed[j,:,:] @ coord[1:,si:])

    if derivative_flag:
      y_r = coord[1:,:]
      tau = coord.shape[1]
      dyn_dV = new_read_dydv_files(filename2, n, N, uncdof, tau)
      dy_ds = np.einsum('ijkl,kl->ij', dyn_dV, dVkds_unc[j,:,:])
      dy_dB = np.einsum('ijkl,kl->ij', dyn_dV,dVkdB_unc[j,:,:])
      dy_dSig = np.einsum('ijkl,klm->ijm', dyn_dV, dVkdSig_unc[j,:,:,:])
      dVyds = dVkds_probed[j,:,:] @ y_r[:,si:] + \
                   Vk_probed[j,:,:] @ dy_ds[:,si:]
      dVydB = dVkdB_probed[j,:,:] @ y_r[:,si:] + \
                   Vk_probed[j,:,:] @ dy_dB[:,si:]
      dVydSig = np.einsum('ikm,kj->ijm', dVkdSig_probed[j,:,:,:], y_r[:,si:]) +\
                np.einsum('ik,kmj->imj', Vk_probed[j,:,:], dy_dSig[:,si:,:])
      return (i, j, sol, dVyds, dVydB, dVydSig)
    else:
      return (i, j, sol)
  else:
    coord = np.reshape(v, (ncoord * problemParameters.n_dofs_per_node + 1, -1),
                         order = 'F')
    ycoord = coord[1:,:]
    return (i, j, ycoord[probed_dofs,:].T)

################################################################################
class SROMAeroS(SROMBase):

  def __init__(self, problemPath):
    super().__init__(problemPath)
    # Add problem dependent code into the search path 
    sys.path.append('./%s/Scripts' %  self.IOParameters.input_ID)
    print('appending ','./%s/Scripts' %  self.IOParameters.input_ID)

    # Copy Aero-S executable into working directory
    eroot = '../Executables'
    os.system('cp %s %s' % (self.IOParameters.program_exec_original, eroot))
    os.system('cp %s %s' % (self.IOParameters.rob_exec_original, eroot))

    # Read reference and ROM/HROM solution: determine target mean and standard
    # deviation for NPM
    observed_point = self.problemParameters.mu

    self.std_ref, self.mean_ref, self.mean_rom, self.t, self.mean_ref_full_ = \
       referenceValues(self.IOParameters, self.problemParameters,
                       observed_point)
    self.c_mean = np.sum(self.mean_ref ** 2)
    self.c_std = np.sum(self.std_ref ** 2)
    
    # Read the ROB, coordinate set, Dirichlet BC, generate K,M
    self.problemParameters.V, self.problemParameters.ROB_nodes, \
      self.problemParameters.ROB_tag, self.problemParameters.x, BC, \
      self.problemParameters.K, self.problemParameters.M, \
      self.problemParameters.R = \
     readROBPlus(self.problemParameters, self.IOParameters,
                 observed_point, eroot)
    uncdofFlag = np.ones(self.problemParameters.V.shape[0])
    uncdofFlag[((BC[:,0] - 1) *
                    self.problemParameters.n_dofs_per_node +
                 BC[:,1] - 1).astype(int)] = 0
    self.problemParameters.uncdof = np.squeeze(np.argwhere(uncdofFlag))

    # finish initializing Gparams
    initGparams(self.problemParameters.V.shape, self.problemParameters.x,
                self.Gparams)

    # Dof numbering
    self.problemParameters.Dof = np.reshape(np.array(
                 range(self.Gparams.N)), (self.Gparams.N0, self.Gparams.m),'C')
    
    # Generate B matrix based on the Dirichlet boundary condition
    self.problemParameters.B = np.zeros((self.Gparams.N, BC.shape[0]))
    for i in range(BC.shape[0]):
      self.problemParameters.B[(self.Gparams.m * 
        (BC[i, 0] - 1) + BC[i, 1] - 1).astype(int), i] = 1
    self.problemParameters.Regular = np.ones(self.Gparams.N0)
    self.problemParameters.Regular[(BC[:, 0] - 1).astype(int)] = 0  # RT - assumption that all dofs of a node are constrained

  ##############################################################################
  # optimization function
  def optimize(self , nit = 60, _alpha_0 = []):

    self.Rrow, self.Rcolumn, alpha_0, lb, ub = \
      initHyperParameters(self.Gparams, self.stochasticParameters)
    if (_alpha_0 != []):
      alpha_0 = _alpha_0
   
    # Random numbers to generate G
    rng = np.random.default_rng(12345)
    self.ArrayRand1IDENT = rng.random((self.Gparams.d, self.Gparams.nu_p,
                                  self.Gparams.m, self.Gparams.n,
                                  self.problemParameters.N_sim))
    self.ArrayRand2IDENT = rng.random((self.Gparams.d, self.Gparams.nu_p,
                                  self.Gparams.m, self.Gparams.n,
                                  self.problemParameters.N_sim))

    bnds = Bounds(lb, ub)
    if not self.problemParameters.use_R:
      R = []

    # Cost function weights and full reference mean 
    if hasattr(self.problemParameters, "distance_metric"):
      if self.problemParameters.distance_metric == "wasserstein":
        from misc_utils import preProcessWasserstein, \
                                 computeWassersteinNormalizingConstants
        self.norm_mean_ref, self.norm_std_ref, self.shift = \
          preProcessWasserstein(self.mean_ref, self.std_ref)
        self.c_mean, self.c_std = computeWassersteinNormalizingConstants(
          np.array(self.norm_mean_ref), np.array(self.norm_std_ref), self.t)
        self.mean_ref = np.array(self.norm_mean_ref)
        self.mean_std = np.array(self.norm_std_ref)

    if hasattr(self.stochasticParameters, "weight"):
      w = self.stochasticParameters.weight
    elif hasattr(self.stochasticParameters, "weight_mean"):
      w = self.stochasticParameters.weight_mean
    self.coeff_mean = w / self.c_mean
    try: 
      if self.problemParameters.composite_cost_flag:
        self.w_p = self.stochasticParameters.weight_proj
        w += self.w_p
        self.mean_ref_full = self.mean_ref_full_[
                                            self.problemParameters.uncdof,:,:]
    except AttributeError as error:
      self.w_p = 0
      self.mean_ref_full = []
    self.coeff_std = (1-w) / self.c_std

    of = ObjectiveF(self.problemParameters.N_sim, rng, self.costFunction)

    if hasattr(self.problemParameters, "batch_size"):
      of.setBatch(self.problemParameters.batch_size)
    else:
      of.setBatch(self.problemParameters.N_sim)
    
    # Optimize the cost function
    if self.problemParameters.derivative_flag:
      jac = True
    else:
      jac = '2-point'
    
    opt_params = basinhopping(of.objective, alpha_0, minimizer_kwargs={'method':'SLSQP', 'jac':jac, 'callback':of.callback_sgd, 'bounds':bnds}, disp=True, stepsize=2, T=2)
    
    # Print and save the optimal values
    print('params=', opt_params.x)
    np.save('opt_params.npy', opt_params.x)
    return opt_params.x
#   return alpha_0

  ##############################################################################
  # postprocessing
  def postprocessing(self, alpha, n_realiz):

    self.Rrow, self.Rcolumn, alpha_0, lb, ub = \
      initHyperParameters(self.Gparams, self.stochasticParameters)
    sigma, s, beta = alphaToHyperparameters(alpha, self.Gparams.n,
                                            self.Rrow, self.Rcolumn)
    
    rng = np.random.default_rng(12345)
    self.problemParameters.N_sim = n_realiz
    self.ArrayRand1IDENT = rng.random((self.Gparams.d, self.Gparams.nu_p,
                                  self.Gparams.m, self.Gparams.n,
                                  self.problemParameters.N_sim))
    self.ArrayRand2IDENT = rng.random((self.Gparams.d, self.Gparams.nu_p,
                                  self.Gparams.m, self.Gparams.n,
                                  self.problemParameters.N_sim))
    batch = np.arange(0, dtype = int)
    
    trealiz = time.time()
    sol = self.makeAndRunRealizations(sigma, s, beta, batch)[0]
    print(self.problemParameters.N_sim, ' realizations time: ',
          time.time() - trealiz, flush = True)
    
    
    rng = np.random.default_rng(12345)
    self.problemParameters.N_sim = 16
    self.ArrayRand1IDENT2 = rng.random((self.Gparams.d, self.Gparams.nu_p,
                                  self.Gparams.m, self.Gparams.n,
                                  self.problemParameters.N_sim))
    self.ArrayRand2IDENT2 = rng.random((self.Gparams.d, self.Gparams.nu_p,
                                  self.Gparams.m, self.Gparams.n,
                                  self.problemParameters.N_sim))
    
    trealiz = time.time()
    sol_rom = self.makeAndRunRealizations(sigma, 1e-6, beta, batch)[0]
    mean_rom2 = np.sum(sol_rom,axis=1)/ self.problemParameters.N_sim
    print(self.problemParameters.N_sim, ' realizations time: ',
          time.time() - trealiz, flush = True)
    return sol, mean_rom2

  ##############################################################################
  # Example of plotting the results
  def plot(self, sol, mean_ref, mean_rom, i_mu, start, pc, dof, filename):

    sol_inf = np.quantile(sol[i_mu,:,:,:], 1 - pc, axis = 0)
    sol_sup = np.quantile(sol[i_mu,:,:,:], pc, axis = 0)
    
    xTest = np.arange(start, mean_ref.shape[1])
    mean = np.sum(sol,axis=1)/sol.shape[1]

    plt.figure(figsize=(9,5))
    plt.plot(xTest, mean_ref[dof,start:,i_mu], 'r-', label='HDM')
    plt.plot(xTest, mean_rom[i_mu,dof,:], 'b-',  label='ROM')
    plt.plot(xTest,  mean[i_mu,dof,:] , 'm-',  label='Mean SROM')
    ax = plt.gca()
    ax.fill_between(xTest, sol_inf[dof,:], sol_sup[dof,:], color='c', alpha=.2)
    plt.xlabel('$t$')
    plt.legend(loc='upper left')
    plt.savefig(filename)
    plt.close()
    return 

  ##############################################################################
  # NPM cost function
  def costFunction(self, alpha, batch):
 
    sigma, s, beta = alphaToHyperparameters(alpha, self.Gparams.n,
                                            self.Rrow, self.Rcolumn)
    N_sim = self.problemParameters.N_sim
    nbsigma = len(self.Rrow)
    if hasattr(self.problemParameters, "composite_cost_flag"):
      composite_cost_flag = self.problemParameters.composite_cost_flag
    else:
      composite_cost_flag = False
    if hasattr(self.problemParameters, "distance_metric"):
      distance_metric = self.problemParameters.distance_metric
    else:
      distance_metric = "l2"

    trealiz = time.time()
    # Generate the Monte-Carlo realizations of the solution 
    sol, dVyds, dVydB, dVydSig, Vk_unc, dVds, dVdB, dVdSig = \
       self.makeAndRunRealizations(sigma, s, beta, batch)
    print(N_sim, ' realizations time: ', time.time() - trealiz, flush = True)

    # Add modules for Wasserstein's derivatives
    if hasattr(self.problemParameters, "distance_metric"):
      if self.problemParameters.distance_metric == "wasserstein":
        try:
          from jax import jacfwd
          import ot
          import torch
          from misc_utils import preProcessWasserstein, \
                                 computeWassersteinNormalizingConstants
        except ImportError:
          print('Module(s) required for Wasserstein not available - aborting.')
          exit()
  
    Cost_mean = 0.0
    Cost_variance = 0.0
  
    meanderivative_term_s = 0
    stdevderivative_term_s = 0
    meanderivative_term_B = 0
    stdevderivative_term_B = 0
    meanderivative_term_Sig = np.zeros(nbsigma)
    stdevderivative_term_Sig = np.zeros(nbsigma)
  
    tmean= time.time()
    solSq = sol * sol

    for i in range(self.problemParameters.m_mu):
      solSum = np.sum(sol[i,:,:,:], axis = (0))
      solSumSq = np.sum(solSq[i,:,:,:], axis = (0))
  
  
      if self.problemParameters.derivative_flag:
        solSum_forderiv = np.sum(sol[i,batch,:,:], axis = (0))
        solSumSq_forderiv = np.sum(solSq[i,batch,:,:], axis = (0))

        # w.r.t s
        sumDerivSol_s = np.sum(dVyds[i,batch,:,:], axis = (0))
        dEosq_ds = np.sum(2.0 * sol[i,batch,:,:] * dVyds[i,batch,:,:], axis = (0))
  
        # w.r.t B
        sumDerivSol_B = np.sum(dVydB[i,batch,:,:], axis = (0))
        dEosq_dB = np.sum(2.0 * sol[i,batch,:,:] * dVydB[i,batch,:,:], axis = (0))
  
        # w.r.t Sig
        sumDerivSol_Sig = np.sum(dVydSig[i,batch,:,:,:], axis = (0))
        dEosq_dSig = np.sum(2.0 * np.tile(np.expand_dims(sol[i,batch,:,:],
            axis=3), (1, 1, 1, nbsigma)) * dVydSig[i,batch,:,:,:], axis = (0))
  
      # Compute the mean and the variance
      expectancy = solSum / N_sim
      square_expectancy = expectancy ** 2
      expectancy2 = solSumSq / N_sim
      H = expectancy2 - square_expectancy
      var = np.sqrt(np.abs(expectancy2 - square_expectancy))

      if composite_cost_flag:
        orth_proj_sum = 0
        dOrthds = 0
        dOrthdB = 0
        dOrthdSig = 0
        for j in range(N_sim):
          Vb = Vk_unc[j,:,:]
          projection = -Vb @ (Vb.T@self.mean_ref_full[:,:,0])
          orth_proj  = projection + self.mean_ref_full[:,:,0]
          orth_proj_sum += orth_proj
          derivative_flag = len(np.nonzero(batch == j)[0]) > 0 and self.problemParameters.derivative_flag
          if derivative_flag:
            dOrthdNorm=-2*orth_proj
            dOrthds +=  np.sum(dOrthdNorm * ((dVds[j,:,:]@Vb.T + Vb@dVds[j,:,:].T) @ self.mean_ref_full[:,:,0]), axis=(0,1))
            dOrthdB +=  np.sum(dOrthdNorm * ((dVdB[j,:,:]@Vb.T + Vb@dVdB[j,:,:].T) @ self.mean_ref_full[:,:,0]), axis=(0,1))
            dOrthdSig=np.sum(np.tile(np.expand_dims(dOrthdNorm, axis=2), (1,1,nbsigma))*np.einsum('ijk,jl->ilk',np.einsum('ijk,jl->ilk', dVdSig[j,:,:,:], Vb.T) + np.einsum('ij,jlk->ilk', Vb, np.transpose(dVdSig[j,:,:,:], (1,0,2))), self.mean_ref_full[:,:,0]), axis=(0,1))
        if self.problemParameters.derivative_flag:
          dOrthds /= len(batch)
          dOrthdB /= len(batch)
          dOrthdSig /= len(batch)
        exp_orth_proj = orth_proj_sum/N_sim
  
      if self.problemParameters.derivative_flag:
        expectancy_sgd = solSum_forderiv / len(batch)
        square_expectancy_sgd =  expectancy_sgd ** 2
        expectancy2_sgd = solSumSq_forderiv / len(batch)
        H_sgd=expectancy2_sgd - square_expectancy_sgd
        var_sgd = np.sqrt(abs(H_sgd))
  
        var_p = 0.5 / var_sgd
        dabsHdH = np.sign(H_sgd) # RT - zero becomes zero, not 1, does it matter?
        elementwise_deriv = var_p * dabsHdH
  
        # w.r.t. s
        dEosq_ds = 1.0 / len(batch) * dEosq_ds
        dWds = sumDerivSol_s / len(batch)
        dWsq_ds = 2.0 * expectancy_sgd * dWds
        dHds = dEosq_ds - dWsq_ds
        dvar_ds = elementwise_deriv * dHds
        # w.r.t. B
        dEosq_dB = 1.0 / len(batch) * dEosq_dB
        dWdB = sumDerivSol_B / len(batch)
        dWsq_dB = 2.0 * expectancy_sgd * dWdB
        dHdB = dEosq_dB - dWsq_dB
        dvar_dB = elementwise_deriv * dHdB
        # w.r.t. Sig
        dEosq_dSig = 1.0 / len(batch) * dEosq_dSig
        dWdSig = sumDerivSol_Sig / len(batch)
        dWsq_dSig = 2.0 * np.tile(np.expand_dims(expectancy_sgd, axis=2), (1,1,nbsigma)) * dWdSig
        dHdSig = dEosq_dSig - dWsq_dSig 
        dvar_dSig = np.tile(np.expand_dims(elementwise_deriv, axis=2), (1,1,nbsigma)) * dHdSig

      if distance_metric == "l2": 
        cost_mean = np.sum(self.coeff_mean.reshape(-1,1)*(self.mean_ref[:,:,i] - expectancy) ** 2,
                                     axis = (0, 1))
        cost_std = np.sum(self.coeff_std.reshape(-1,1)*(self.std_ref[:, :, i] - var) ** 2,
                                       axis = (0, 1))
      elif distance_metric == "wasserstein":
        norm_expectancy = np.array(preProcessWasserstein(expectancy, mean_shift = self.shift)[0]) 
        norm_var = np.array(preProcessWasserstein(var, mean_shift = 0)[0])
        time_wassInput= np.tile(self.t.reshape(-1,1), (1,expectancy.shape[0])) 
        expectancy_wassInput = norm_expectancy.T
        mean_ref_wassInput = self.mean_ref[:,:,i].T 
        std_ref_wassInput = self.std_ref[:,:,i].T 
        var_wassInput = norm_var.T
        fluctuation_mean = ot.wasserstein_1d(time_wassInput,time_wassInput,expectancy_wassInput,mean_ref_wassInput,p=2)
        fluctuation_std = ot.wasserstein_1d(time_wassInput,time_wassInput,var_wassInput,std_ref_wassInput,p=2)
        cost_mean = np.sum(self.coeff_mean*fluctuation_mean)
        cost_std = np.sum(self.coeff_std*fluctuation_std)
        
        if self.problemParameters.derivative_flag:
          time_wassInput_wgrad = torch.tensor(time_wassInput)
          norm_expectancy_wgrad = np.array(preProcessWasserstein(expectancy_sgd, mean_shift=self.shift)[0])
          norm_var_wgrad = np.array(preProcessWasserstein(var_sgd, mean_shift = 0)[0])
          expectancy_wassInput_wgrad = torch.tensor(norm_expectancy_wgrad.T).requires_grad_(True)
          mean_ref_wassInput_wgrad = torch.tensor(mean_ref_wassInput) 
          std_ref_wassInput_wgrad = torch.tensor(std_ref_wassInput) 
          var_wassInput_wgrad = torch.tensor(norm_var_wgrad.T).requires_grad_(True)
          fluctuation_mean_wgrad = ot.wasserstein_1d(time_wassInput_wgrad,time_wassInput_wgrad,expectancy_wassInput_wgrad,mean_ref_wassInput_wgrad,p=2) 
          fluctuation_std_wgrad = ot.wasserstein_1d(time_wassInput_wgrad,time_wassInput_wgrad,var_wassInput_wgrad,std_ref_wassInput_wgrad,p=2) 
        
          dWassdNormExp = torch.autograd.grad(fluctuation_mean_wgrad, expectancy_wassInput_wgrad, torch.ones_like(fluctuation_mean_wgrad))[0] 
          dWassdNormVar = torch.autograd.grad(fluctuation_std_wgrad, var_wassInput_wgrad, torch.ones_like(fluctuation_std_wgrad))[0]
          dNormExpdExp = np.array(jacfwd(preProcessWasserstein, argnums=0)(expectancy_sgd, mean_shift = self.shift)[0])
          dNormVardVar = np.array(jacfwd(preProcessWasserstein, argnums=0)(var_sgd)[0]) 
    
          dWassdExp = np.zeros(dWassdNormExp.T.shape)
          dWassdVar = np.zeros(dWassdNormVar.T.shape)
          for pdof in range(expectancy.shape[0]):       
            dNormExpdExp_pdof = dNormExpdExp[pdof, :, pdof,:]
            dWassdExp[pdof,:]= dNormExpdExp_pdof.T @ dWassdNormExp.detach().numpy()[:,pdof]
            dNormVardVar_pdof = dNormVardVar[pdof,:,pdof,:] 
            dWassdVar[pdof,:] = dNormVardVar_pdof.T @ dWassdNormVar.detach().numpy()[:,pdof] 
  
      # Compute the mean and variance cost associated with this parameter point
      Cost_mean = Cost_mean + cost_mean
      Cost_variance = Cost_variance + cost_std

      if self.problemParameters.derivative_flag:
        if distance_metric == "wasserstein":
          meanderivative_term_s = meanderivative_term_s + np.sum(self.coeff_mean * dWassdExp * dWds, axis = (0,1))
          stdevderivative_term_s = stdevderivative_term_s + np.sum(self.coeff_std * dWassdVar * dvar_ds, axis = (0,1))
  
          meanderivative_term_B = meanderivative_term_B + np.sum(self.coeff_mean * dWassdExp * dWdB, axis = (0,1))
          stdevderivative_term_B = stdevderivative_term_B + np.sum(self.coeff_std * dWassdVar * dvar_dB, axis = (0,1))
  
          meanderivative_term_Sig = meanderivative_term_Sig + np.sum(
           np.tile(np.expand_dims(self.coeff_mean * dWassdExp, axis=2), (1,1,nbsigma))  * dWdSig, axis = (0, 1))
          stdevderivative_term_Sig = stdevderivative_term_Sig + np.sum(
           np.tile(np.expand_dims(self.coeff_std * dWassdVar, axis=2), (1,1,nbsigma)) * dvar_dSig, axis = (0, 1))

        else:
          # w.r.t s
          meanderivative_term_s = meanderivative_term_s + np.sum(
                 -2.0 * self.coeff_mean * (self.mean_ref[:,:,i] - expectancy_sgd) * dWds, axis = (0, 1))
          stdevderivative_term_s = stdevderivative_term_s + np.sum(
                 -2.0 * self.coeff_std * (self.std_ref[:,:,i] - var_sgd) * dvar_ds, axis = (0, 1))
          # w.r.t B
          meanderivative_term_B = meanderivative_term_B + np.sum(
                 -2.0 * self.coeff_mean * (self.mean_ref[:,:,i] - expectancy_sgd) * dWdB, axis = (0, 1))
          stdevderivative_term_B = stdevderivative_term_B + np.sum(
                 -2.0 * self.coeff_std * (self.std_ref[:,:,i] - var_sgd) * dvar_dB, axis = (0, 1))
          # w.r.t Sig
          meanderivative_term_Sig = meanderivative_term_Sig + np.sum(
             np.tile(np.expand_dims(-2.0 * self.coeff_mean * (self.mean_ref[:,:,i] - expectancy_sgd), axis=2), (1,1,nbsigma))  * dWdSig, axis = (0, 1))
          #meanderivative_term_Sig = np.einsum('ij,ijk->k', -2*(mean_ref[:,:,i] - expectancy_sgd), dWdSig)
          stdevderivative_term_Sig = stdevderivative_term_Sig + np.sum(
             np.tile(np.expand_dims(-2.0 * self.coeff_std * (self.std_ref[:,:,i] - var_sgd), axis=2), (1,1,nbsigma)) * dvar_dSig, axis = (0, 1))
  
    print('Mean computation time: ', time.time() - tmean)
    print('Cost variance: ', Cost_variance, '  Cost_mean: ', Cost_mean)
    print('Scaled Cost variance: ', Cost_variance/self.coeff_std,
        '  Scaled Cost_mean: ', Cost_mean/self.coeff_mean, flush = True)

    J = Cost_mean + Cost_variance
    if composite_cost_flag:
      J += self.w_p * np.linalg.norm(exp_orth_proj, 'fro')**2
  
    if self.problemParameters.derivative_flag:
  
      # Component-wise
      # w.r.t s
      dJmean_ds = meanderivative_term_s
      dJstdev_ds = stdevderivative_term_s
      dJ_ds= dJmean_ds + dJstdev_ds
  
      # w.r.t B
      dJmean_dB = meanderivative_term_B
      dJstdev_dB = stdevderivative_term_B
      dJ_dB= dJmean_dB + dJstdev_dB
  
      # w.r.t Sig
      dJmean_dSig = meanderivative_term_Sig
      dJstdev_dSig = stdevderivative_term_Sig
      dJ_dSig = dJmean_dSig + dJstdev_dSig

      if composite_cost_flag:
        dJ_ds += self.w_p * dOrthds
        dJ_dB += self.w_p * dOrthdB
        dJ_dSig += self.w_p * dOrthdSig
  
      dJ_da = np.hstack((dJ_dSig, dJ_ds, dJ_dB))
  
      print('alpha = ' , alpha)
      print('J =  ', '%.17e' % J, dJ_da, flush = True)
      print('dJ_da', dJ_da)
      return (J, dJ_da)
    else:
      print('alpha = ' , alpha)
      print('J =  ', '%.17e' % J, flush = True)
      return J

  ##############################################################################
  # Creates and runs realizations of the stochastic ROM
  def makeAndRunRealizations(self, sigma, s, beta, batch):
 
     parallelFlag = True
    if parallelFlag:
      nj = int(self.runParameters.ncores_per_node/2)
    else:
      nj = 1

    # This function parallelizes the computation of ROB realizations and then
    # runs the corresponding ROMs concurrently using Aero-s in the shared memory 
    # filesystems (/dev/shm)  of the compute nodes to minimize I/O bottlenecks;
    # the function should work if called in parallel and launched with disjoint
    # exec_nodes for evaluating multiple hyper-parameter choices at a time
    #
    # Inputs
    #
    # Parameters that control where to run Aero-s
    # runParameters.exec_nodes - pre-allocated nodes where Aero-s is run
    # runParameters.ncores_per_node - number of cores per compute node
    #
    # Construction of realizations of G
    # Gparams - structure with parameters of G
    # x - coordinates of nodes where to generate G
    # Dof, Regular - parameters for the construction of G (see Appendix D)
    # ArrayRand1IDENT, ArrayRand2IDENT - random arrays
    # 
    # Hyperparameters
    # sigma, s, beta
    # 
    # V - ROB
    #
    # ROM/Aero-S parameters
    # problemParameters - structure with problem dependent data/parameters
    # problemParameters.m_mu - number of parameter points where the SROM is evaluated
    # problemParameters.rayleigh_damping - Rayleigh damping coefficients
    # problemParameters.hyperreduction_flag - use-hyperreduction flag
    # problemParameters.sampled_mesh_flag - using V on a sampled mesh only
    # problemParmameters.K,problemData.M - cell arrays of stiffness and mass matrices (only for problemData.sampled_mesh_flag==true)
    # problemParameters.uncdof - unconstrained degrees of freedom
    # problemParameters.probed_dofs - dofs needed for the computation of QoIs
    # problemParameters.N_sim - number of realizations per each parameter point
    #
    # Outputs
    # sol - array that stores the solutions at the probed dofs
    
    # Determine probed degrees of freedom (whether on sampled or full mesh)
    if not self.problemParameters.sampled_mesh_flag:
      probed_dofs = np.copy(self.problemParameters.probed_dofs)
    else:
      probed_dofs = np.copy(self.problemParameters.probed_dofs_s)
    
    # Initialize local variables 
    exec_nodes = self.runParameters.exec_nodes
    ncores_per_node= self.runParameters.ncores_per_node
  
    f_unc = 0
    y_0_unc = 0
    if self.problemParameters.derivative_flag and \
       self.problemParameters.linear_flag:
      f_unc = self.problemParameters.f[self.problemParameters.uncdof]
      y_0_unc = self.problemParameters.y_0[self.problemParameters.uncdof]
    
    tinit = time.time()

    # Create unique directory to store files for the SROM realizations
    nodep = NodeProcessor(self.problemParameters.N_sim,
                          self.problemParameters.m_mu,
            self.runParameters.exec_nodes, self.runParameters.ncores_per_node,
                       self.IOParameters.output_ID)
    root, proot, croot2 = nodep.getDirs()
    
    # Generate SROB_dir realizations (in parallel on the local node)
    js = [j for j in range(self.problemParameters.N_sim)]
    res = Parallel(n_jobs = nj) \
                (delayed(makeSROMAeroS)(j,
               self.problemParameters, self.IOParameters, self.Gparams, 
               self.ArrayRand1IDENT, self.ArrayRand2IDENT, sigma, s, beta,
               probed_dofs, root, proot, croot2, batch) for j in js)
    # Convert list to arrays
    if self.problemParameters.derivative_flag and len(batch) > 0:
      Vk_unc, Vk_probed, Vk_full, dVkds_unc, dVkdB_unc, dVkdSig_unc, \
      dVkds_probed, dVkdB_probed, dVkdSig_probed = convertParallelListToArray(
         res, (self.problemParameters.N_sim,))
    else:
      Vk_unc, Vk_probed, Vk_full =  convertParallelListToArray(res,
        (self.problemParameters.N_sim,))
      dVkds_unc = dVkdB_unc = dVkdSig_unc = dVkds_probed = dVkdB_probed = \
         dVkdSig_probed = np.array([])
    print('SROM time: ', time.time() - tinit, flush = True)
  
    # Cleanup shared memory directories on the compute nodes
    # Transfer & run the batch files on the compute nodes
    nodep.runJobs(parallelFlag, batch)
  
    # Post-process the results
    tpost = time.time()
    data = []
    for i in range(self.problemParameters.m_mu):
      for j in range(self.problemParameters.N_sim):
        k = j * self.problemParameters.m_mu+ i
        croot = '%s/%d' % (root, k)
        filename = '%s/%s' % (croot, file_format(self.problemParameters.mu[i,:], 'dis'))
        filename2 = '%s/%s.%d' % (self.IOParameters.output_ID,
                                  self.IOParameters.exact_grad_output, k)
        derivative_flag = len(np.nonzero(batch == j)[0]) > 0 and \
                         self.problemParameters.derivative_flag
        data.append((i, j, filename, filename2, derivative_flag))

    sol_ = Parallel(n_jobs = nj) \
          (delayed(postProAeroS)(
              datum[0], datum[1], datum[2], datum[3], datum[4],
              self.problemParameters.linear_flag,
              self.problemParameters.sol_start_idx,
              Vk_probed,
              dVkds_unc, dVkdB_unc, dVkdSig_unc,
              dVkds_probed, dVkdB_probed, dVkdSig_probed, 
              self.Gparams.n, self.problemParameters.V.shape[0],
              self.problemParameters.uncdof)
           for datum in data)

    if self.problemParameters.derivative_flag and len(batch) > 0:
      sol, dVyds, dVydB, dVydSig = convertParallelListToArray(sol_, \
        (self.problemParameters.m_mu, self.problemParameters.N_sim))
    else:
      sol = convertParallelListToArray(sol_,
         (self.problemParameters.m_mu, self.problemParameters.N_sim))[0]
  
    # Delete the directory with the realizations input files and results
    os.system('rm -rf %s' % root)
    print('Post-pro time: ', time.time() - tpost, flush = True)
  
    if not self.problemParameters.derivative_flag or len(batch) == 0:
      return sol, [], [], [], Vk_unc, [], [], []
    else:
      return sol, dVyds, dVydB, dVydSig, Vk_unc, dVkds_unc, dVkdB_unc, dVkdSig_unc
################################################################################
