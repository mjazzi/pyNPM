################################################################################
# Problem-dependent definitions
################################################################################

import numpy as np
import os


class ProblemParameters:
  """Class to hold problem parameters."""
  linear_flag = 0
  rayleigh_damping = [0.0, 0.0]
  hyperreduction_flag = True
  sampled_mesh_flag = True 
  use_R = True
  n_dofs_per_node = 3
  probed_node_s = np.array([6, 17, 71, 165], dtype = int) - 1  # probed node numbers in sampled mesh numbering
  probed_node = np.array([593, 1891, 13769, 28977], dtype = int) - 1 # probed node number in full mesh numbering
  probed_dofs = np.hstack((self.probed_node * 3 + i for i in range(3)))
  probed_dofs_s = np.hstack((self.probed_node_s * self.n_dofs_per_node + i for i in range(3)))
  mu = np.array([[1.00, 1.00]]) # Observed location (for NPM) 
  m_mu = mu.shape[0]
  N_sim= 72 # number of MC realizations for computing the objective function
  sol_start_idx = 1
  derivative_flag = True
  composite_cost_flag = True
  max_svd_points = 500
  svd_energy = 0.25
  poscfg_flag = True # collected training snapshots were computed on different meshes when constructing ROB
  hyperreduction_tol = 0.01 
  stacked_training_flag = True 
  sgd_gridsearch = False
  distance_metric = "wasserstein"
  batch_size = 24
  def __init__(self, io):
    self.M = []
    self.K = []

class GreedyParameters:
  """Class to hold greedy parameters."""
  grid_size = 3
  max_ROB_points = 8
  residual_tol = 4.0e-1   
  exact_residual_flag = True 
  lower_bound = [0.95, 0.95]
  upper_bound = [1.05, 1.05]

class IOparameters:
  """Class to hold input/output parameters."""
  base_dir = '%s/..' % os.getcwd()
  problem_ID = 0;
  input_ID = '../input%d.d' % problem_ID # Aero-S problem input location
  output_ID = '../output%d.d' % problem_ID # Outputs from greedy procedure and NPM location
  program_exec_original ='~mjazzi/aero-s/bin/aeros'
  rob_exec_original='$GROUP_HOME/bin/rob'
  program_exec = '%s/Executables/aeros' % base_dir # location of the aero-s files
  rob_exec = '%s/Executables/rob' % base_dir
  mpirun = '/home/pavery/openmpi-1.8.3-install/bin/mpirun'
  ROM_results_dir = 'Results/ROM'
  HDM_results_dir ='Results/HDM'
  HROM_results_dir = 'Results/HROM'

  # Aero-s input file names
  ROM_output = 'nozzle.modelII' 
  HROM_output = 'nozzle.modelIII' 
  ROM_post_output = 'nozzle.modelIIpost'
  HROM_post_output = 'nozzle.modelIII.post' 
  HDM_output = 'HDMscript' 
  SVD_output = 'nozzle.svd'
  concat_output = 'nozzle.concatenate'
  spnnls_output = 'nozzle.spnnls'
  spnnls_output_2 = 'nozzle.spnnls.new' 
  mass_stiff_output = 'nozzle.massandstiff2'
  exact_grad_output='exactgrad'

  basis_dir = 'basis_dir/ROB'
  orth_basis_dir = 'basis_dir/ROB.orthonormalized'
  input_dir = 'Input'
  MFTT ='Input/MFTT'

  # Aero-s log file names
  ROM_post_log_file ='log_postROM' 
  HROM_post_log_file ='log_postHROM'
  ROM_log_file ='log_rom' 
  HROM_log_file ='log_hrom' 
  HDM_log_file  = 'log_HDM'
  SVD_log_file ='log_svd'
  ROB_log_file ='log_rob'
  concat_log_file ='log_conc'
  spnnls_log_file ='log_spnnls'
  spnnls_log_file_2 ='log_spnnlsn'
  mass_stiff_log_file ='log_ms2' 

  SROB_dir ='SROB.orthonormalized' 
  mesh_dir ='Mesh'
  

class StochasticParameters:
  """Class to hold stochastic parameters."""
  s_0 = 1e-3
  beta_0 = 6
  diag_coef = 1.0
  diag_min = 0.0
  extra_diag_min = -20.0
  diag_max = 40.0
  extra_diag_max = 20.0
  lb_s = 1.0e-5
  ub_s = 1.0
  lb_beta = 0.01
  ub_beta = 30.0
  weight_mean = 0.6
  weight_proj = 0.1


class GParams: 
  """Class to hold GParams as defined in Soize and Farhat, IJNME 2017."""
  d = int(3)  # dimension of the computational domain
  m = int(3)  # dimension of the field discretized by FEM
  nu_p = int(20)
  nu = np.array(range(nu_p)) + 1.0
  K_nu = -1 + ((2 * (nu - 0.5)) / nu_p)
  n = 0
  N0 = 0
  N = 0
  def __init__(self):
    self.L = np.zeros(self.d)
    R2sum_nu = np.zeros(self.nu_p)
    print('GParams init',flush=True)
