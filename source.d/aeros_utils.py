################################################################################
# Utilities that generate/modify/read Aero-S input/result files
################################################################################

import numpy as np
import os
from scipy.io import mmread

def readAeroSROB(filename, n_dofs_per_node, IOParameters):
  """
  Read the Aero-S ROB from a file.

  Parameters:
  filename (str): The name of the file to read from.
  n_dofs_per_node (int): The number of degrees of freedom per node.
  IOParameters (object): The input/output parameters.

  Returns:
  tuple: The V, ROB_nodes, and ROB_tag values.
  """
  with open(filename, 'r') as f:
      ROB_size = int(f.readline())
      ROB_n_nodes = int(f.readline())
      ROB_nodes = np.zeros(ROB_n_nodes, dtype=int)
      for i in range(ROB_n_nodes):
          ROB_nodes[i] = int(f.readline()) - 1
      ROB_tag = np.zeros(ROB_size)
      V = np.zeros((ROB_n_nodes * n_dofs_per_node, ROB_size))
      ROB = np.zeros((6, ROB_n_nodes))
      for ii in range(ROB_size):
          ROB_tag[ii] = float(f.readline())
          for i in range(ROB_n_nodes):
              ROB[:,i] = np.array(f.readline().split(), dtype=float)
          V[:,ii] = ROB[0:n_dofs_per_node,:].flatten(order='F')
  return V, ROB_nodes, ROB_tag

def readAeroSInputDisp(filename):
  """
  Extract DISP from Aero-S input file.

  Parameters:
  filename (str): The name of the file to read from.

  Returns:
  array: The v values.
  """
  print('readAeroSInputDisp filename= ', filename)
  os.system('csplit -f yy -s %s /DISP/+1' % filename)
  os.system('csplit -f yyy -s yy01 /^[A-Z]/')
  v = np.loadtxt('yyy00', comments='*')
  print('readAeroSInputDisp v.shape= ', v.shape)
  os.system('rm yy*')
  return v

def referenceValues(IOParameters, problemParameters, observed_point):
  """
  Read reference values.

  Parameters:
  IOParameters (object): The input/output parameters.
  problemParameters (object): The problem parameters.
  observed_point (array-like): The observed point values.

  Returns:
  tuple: The std_ref, mean_ref, mean_rom, tt, and mean_ref_full values.
  """
  c_mean = 0
  c_std = 0
  delta = 1.2
  n_dofs_per_node = problemParameters.n_dofs_per_node
  compositeCostFlag = getattr(problemParameters, "compositeCostFlag", False)

  for ii in range(observed_point.shape[0]):
    file_dis = file_format(observed_point[ii,:], 'dis');
    file_disp = file_format(observed_point[ii,:], 'disp'); 
  
    print(file_format(observed_point[ii,:], 'dis'))
    refsol_, tt = femsol('%s/%s/%s' % (IOParameters.output_ID, 
        IOParameters.ResultsHDM, file_dis), 3, 0)
    refsol = refsol_[:,problemParameters.solStartI:]
    if not problemParameters.hyperFlag:
      romsol_, tt = femsol('%s/%s/%s' % (IOParameters.output_ID,
        IOParameters.ResultsROM, file_disp), 3, 0)
    else:
      print('%s/%s/%s' % (IOParameters.output_ID,
            IOParameters.ResultsHROM, file_disp))
      romsol_, tt = femsol('%s/%s/%s' % (IOParameters.output_ID,
        IOParameters.ResultsHROM, file_disp), 3, 0)
    romsol = romsol_[:,problemParameters.solStartI:]

    if ii == 0:
      mean_ref = np.zeros((len(problemParameters.probed_dofs), refsol.shape[1], observed_point.shape[0]))
      mean_rom = np.zeros((len(problemParameters.probed_dofs), refsol.shape[1], observed_point.shape[0]))
      std_ref = np.zeros((len(problemParameters.probed_dofs), refsol.shape[1], observed_point.shape[0]))
      if compositeCostFlag:
        if problemParameters.hyperFlag:
          sampledHDMnodes = np.loadtxt('%s/eqNodes' % IOParameters.output_ID).astype(int)
          sampledHDMdofs =np.vstack(((sampledHDMnodes-1)*3 + 1, (sampledHDMnodes-1)*3 + 2, (sampledHDMnodes-1)*3 + 3)).T.flatten()
          mean_ref_full = np.zeros((len(sampledHDMdofs),  refsol.shape[1], observed_point.shape[0]))
        else:
          mean_ref_full =  np.zeros((len(problemParameters.probed_dofs), refsol.shape[1], observed_point.shape[0]))
      else:
        mean_ref_full = []

    mean_ref[:,:,ii]  =  refsol[problemParameters.probed_dofs,:]
    mean_rom[:,:,ii] = romsol[problemParameters.probed_dofs,:]

    try:
      from misc_utils import computeFluctuations
      std_ref[:,:,ii], qualityFlag = computeFluctuations(mean_ref[:,:,ii], mean_rom[:,:,ii])
    except ImportError:
      print(
      'Error: computeFluctuations not available - using a legacy computation')
      qualityFlag = -1
      std_ref[:,:,ii] = np.abs(mean_ref[:,:,ii]-mean_rom[:,:,ii]) * delta
      
    if compositeCostFlag:
      if not problemParameters.hyperFlag:
        mean_ref_full[:,:,ii]= refsol
      else:
        mean_ref_full[:,:,ii]= refsol[sampledHDMdofs,:]
    print('qualityFlag: ', qualityFlag)

  return std_ref, mean_ref, mean_rom, \
        tt[problemParameters.solStartI:],  mean_ref_full

def file_format(parameterPoint, type):
  """
  Implement a file format convention for parametric result files.

  Parameters:
  parameterPoint (array-like): The parameter point values.
  type (str): The type of the file.

  Returns:
  str: The formatted file name.
  """
  file = type 
  for i in range(parameterPoint.shape[-1]):
    file = file + '_%.4f' % parameterPoint[i]
  return file

def femsol(inpfile, nc, n):
  """
  Read an xpost solution.

  Parameters:
  inpfile (str): The input file name.
  nc (int): The number of columns.
  n (int): The number of rows.

  Returns:
  tuple: The reshaped array and the first row of the array.
  """
  igz = inpfile.find('.gz')
  if igz == -1:
    f = open(inpfile, 'r')
  else:
    os.system('gzip -d -c ' + inpfile + ' > /tmp/' + inpfile[0:igz])
    f = open('/tmp/' + inpfile[0:igz], 'r')
  
  if n==0:
    f.readline()
    n = int(f.readline())
    print('inpfile = ', inpfile, ' n = ', n)

  array = [x for line in f for x in line.split()]
  v = np.array(array, dtype=float)
  
  f.close()
  if igz != -1:
    os.system('rm /tmp/' +  inpfile[0:igz])
  
  zz = np.reshape(v, (nc * n + 1,-1), order = 'F')
  return zz[1:,:], zz[0,:]

def aeros_rwSampledMesh(lr, pathname, opathname, sz):
  """
  Read (& modify) Aero-S sampled mesh.

  Parameters:
  lr (array-like): The lr values.
  pathname (str): The path name.
  opathname (str): The output path name.
  sz (int): The size.

  Returns:
  array: The nodes.
  """
  if len(lr) == 2: 
    lr =np.array([lr[0], lr[1], lr[1]])

  os.system('csplit -s %s /TOPOLOGY/' % pathname)
  os.system('csplit -f xxx -s xx00 /NODES/+1')
  with open('xxx00','r') as f, open('xx00_','w') as fo:
    for line in f:
      if sz != 0 and 'inorm' in line:
        fo.write('0 inorm "SROB.orthonormalized" %d\n' % sz)
      else:
        fo.write(line)
  nodes = np.loadtxt('xxx01', comments = '*')
  nodes[:,1:4] = lr * nodes[:,1:4]
  with open('xx00_', 'a') as fo:
    for i in range(nodes.shape[0]):
      fo.write('%d  %.16e %.16e %.16e\n' % (int(nodes[i, 0]),
            nodes[i, 1], nodes[i, 2], nodes[i, 3]))
    fo.write('*\n')
  if sz != 0:
    with open('xx02', 'w') as fo:
      print('*\nDIMASS\nMODAL\n"samplmsh.new.reducedmass"', file = fo)
    os.system('cat xx00_ xx01 xx02 > %s' % opathname)
  else :
    os.system('cat xx00_ xx01 > %s' % opathname)
  
  os.system('rm xx*')
  return nodes[:,1:]

def aeros_MassStiff2(parameterPoint, pathSampledMesh, IOParameters):
  """
  Create input Aero-S input files to write stiffness & mass matrices.

  Parameters:
  parameterPoint (array-like): The parameter point values.
  pathSampledMesh (str): The path to the sampled mesh.
  IOParameters (object): An object containing input/output parameters.
  """
  file = file_format(parameterPoint, '')
  pathOutput = f'{IOParameters.output_ID}/{IOParameters.mass_stiff_output}'
  pathInput = f'{IOParameters.input_ID}/{IOParameters.Input}'
  pathMesh = f'{IOParameters.output_ID}/{IOParameters.Mesh}'

  with open(pathOutput,'w') as f:
    f.write(f'DYNAMICS\nsrom\nPRINTMAT "{pathMesh}/{file}"\n*\nNONLINEAR\nlinearelastic\n*\nINCLUDE "{pathSampledMesh}"\n*\nINCLUDE "{pathInput}/COMPOSITE.txt"\n*\nEND\n')

def getPBSInfo():
  """
  Read PBS information to initialize the list of nodes and number of cores.

  Returns:
  tuple: The number of processors per node (ppn) and a list of unique nodes (unodes).
  """
  filename = os.getenv('PBS_NODEFILE')
  with open(filename, 'r') as f:
    nodesn = f.readlines()
  unodesn = list(set(nodesn))
  unodes = [unoden.strip() for unoden in unodesn]
  ppn = int(os.getenv('PBS_NUM_PPN'))
  return ppn, unodes

def readROBPlus(problemParameters, IOParameters, observed_point, eroot):
  """
  Read/generate ROB, coordinates, Dirichlet BC, mass&stiffness from Aero-s files.

  Parameters:
  problemParameters (object): An object containing problem parameters.
  IOParameters (object): An object containing input/output parameters.
  observed_point (array-like): The observed point values.
  eroot (str): The root directory for the Aero-s executable.

  Returns:
  tuple: The reduced order basis (V), ROB nodes, ROB tag, coordinates (x), boundary conditions (BC), stiffness matrices (K), mass matrices (M), and an empty list (R).
  """
  K = []
  M = []
  if problemParameters.hyperFlag and problemParameters.sampledFlag:
    # Read the sampled ROB
    V, ROB_nodes, ROB_tag = readAeroSROB(f'{IOParameters.output_ID}/Mesh/samplmsh.elementmesh.inc.compressed.basis.orthonormalized.out', problemParameters.n_dofs_per_node, IOParameters)
    # Orthogonalize the basis
    [Q,R] = np.linalg.qr(V)
    V = Q
    # Find unconstrained dofs by reading DISPLACEMENT section of the input file
    BC = readAeroSInputDisp(f'{IOParameters.output_ID}/Mesh/samplmsh.elementmesh.inc')
    # Generate the mass matrix that corresponds to the observed parameter
    for ii in range(observed_point.shape[0]):
      # Scale the sampled mesh
      aeros_rwSampledMesh(observed_point[ii,:], f'{IOParameters.output_ID}/Mesh/samplmsh.elementmesh.inc', f'{IOParameters.output_ID}/Mesh/samplmshx.elementmesh.inc', 0)
      # Create Aero-s input file to generate the mass matrix
      aeros_MassStiff2(observed_point[ii,:], f'{IOParameters.output_ID}/Mesh/samplmshx.elementmesh.inc', IOParameters)
      # Run Aero-s
      os.system(f'{eroot}/aeros -v 2 {IOParameters.output_ID}/{IOParameters.mass_stiff_output}  >& {IOParameters.mass_stiff_log_file}')
      file = file_format(observed_point[ii, :], '')
      # Read matrix and convert to full matrices as Python sparse matrix
      M.append(mmread(f'{IOParameters.output_ID}/{IOParameters.Mesh}/{file}.mass').toarray())
      K.append(mmread(f'{IOParameters.output_ID}/{IOParameters.Mesh}/{file}.stiffness').toarray())
      aeros_rwSampledMesh(observed_point[ii,:], f'{IOParameters.output_ID}/Mesh/samplmsh.elementmesh.inc', f'{IOParameters.output_ID}/Mesh/samplmsh.new{ii}.elementmesh.inc', V.shape[1])
    x = aeros_rwSampledMesh([1.0, 1.0, 1.0], f'{IOParameters.output_ID}/Mesh/samplmsh.elementmesh.inc', f'{IOParameters.output_ID}/Mesh/samplmshx.elementmesh.inc', 0)
  else: 
    V, ROB_nodes, ROB_tag = readAeroSROB(f'{IOParameters.output_ID}/{IOParameters.orth_basis_dir}.out', problemParameters.n_dofs_per_node, IOParameters)
    R = []
    x = np.loadtxt(f'{IOParameters.input_ID}/Input/GEOMETRY.txt', usecols = (1, 2, 3), skiprows = 1)
    BC = np.loadtxt(f'{IOParameters.input_ID}/Input/DISPLACEMENTS.txt', skiprows = 1)
  return V, ROB_nodes, ROB_tag, x, BC, K, M, R

def new_read_dydv_files(filename, n, N, uncdof, totaltimestep):
  """
  Read exact gradients.

  Parameters:
  filename (str): The file name.
  n (int): The number of rows.
  N (int): The number of columns.
  uncdof (int): The unconstrained degrees of freedom.
  totaltimestep (int): The total time step.

  Returns:
  array: The reshaped array.
  """
  with open(filename, 'rb') as f:
    f.read(8)
    d = np.fromfile(f, dtype = np.float64)
  dy = np.reshape(d, (n, totaltimestep, N, n), order = 'F')
  return dy[:, :, uncdof, :]