################################################################################
# Utilities that generate/modify/read Aero-S input/result files
################################################################################

import numpy as np
import os
from scipy.io import mmread

################################################################################
# Read the Aero-S ROB
def readAeroSROB(filename, ndofspernode, IOParameters):
# filename of the ASCII ROB file
# ndofspernode - how many dofs per node to return in the ROB
# V - ROB
# ROB_nodes - list of nodes from the ROB file
# ROB_tag - vector of tags from the ROB file
  f = open(filename, 'r')
  ROB_size = int(f.readline())
  ROB_n_nodes = int(f.readline())
  ROB_nodes = np.zeros(ROB_n_nodes, dtype = int)
  for i in range(ROB_n_nodes):
    ROB_nodes[i] = int(f.readline()) - 1
  ROB_tag = np.zeros(ROB_size)
  V = np.zeros((ROB_n_nodes * ndofspernode, ROB_size))
  ROB = np.zeros((6, ROB_n_nodes))
  for ii in range(ROB_size):
    ROB_tag[ii] = float(f.readline())
    for i in range(ROB_n_nodes):
      ROB[:,i]  = np.array(f.readline().split(), dtype = float)
    V[:,ii]  = ROB[0:ndofspernode,:].flatten(order = 'F')
  f.close()
  return V, ROB_nodes, ROB_tag

################################################################################
# Extract DISP from Aero-S input file
def readAeroSInputDisp(filename):
  print('readAeroSInputDisp filename= ', filename)
  os.system('csplit -f yy -s %s /DISP/+1' % filename)
  os.system('csplit -f yyy -s yy01 /^[A-Z]/')
  v = np.loadtxt('yyy00', comments = '*')
  print('readAeroSInputDisp v.shape= ', v.shape)
  os.system('rm yy*')
  return v

################################################################################
# Read reference values
def referenceValues(IOParameters, problemParameters, observed_point):
  c_mean = 0
  c_std = 0
  delta = 1.2
  ndofspernode = problemParameters.ndofspernode
  if hasattr(problemParameters, "compositeCostFlag"):
    compositeCostFlag = problemParameters.compositeCostFlag
  else:
    compositeCostFlag = False

  for ii in range(observed_point.shape[0]):
    file_dis = file_format(observed_point[ii,:], 'dis');
    file_disp = file_format(observed_point[ii,:], 'disp'); 
  
    print(file_format(observed_point[ii,:], 'dis'))
    refsol_, tt = femsol('%s/%s/%s' % (IOParameters.OutputID, 
    	  IOParameters.ResultsHDM, file_dis), 3, 0)
    refsol = refsol_[:,problemParameters.solStartI:]
    if not problemParameters.hyperFlag:
       romsol_, tt = femsol('%s/%s/%s' % (IOParameters.OutputID,
    	  IOParameters.ResultsROM, file_disp), 3, 0)
    else:
      print('%s/%s/%s' % (IOParameters.OutputID,
            IOParameters.ResultsHROM, file_disp))
      romsol_, tt = femsol('%s/%s/%s' % (IOParameters.OutputID,
    	  IOParameters.ResultsHROM, file_disp), 3, 0)
    romsol = romsol_[:,problemParameters.solStartI:]

#    if problemParameters.derivativeflag:
#    #RT - from referenceValues_withderiv2.m
#  mean_ref(:,:,ii)  =  refsol(problemParameters.probed_dofs,1:end);
#  mean_rom(:,:,ii) = romsol(problemParameters.probed_dofs,1:end);
#mean_ref = mean_ref(:, 2:(problemParameters.finalTime)/(25*problemParameters.timestep) + 1, ii) ;
#mean_rom = mean_rom(:, 2:problemParameters.finalTime/(25*problemParameters.timestep)+ 1, ii) ;
#std_ref(:,:,ii) = abs(mean_ref(:,:,ii)-mean_rom(:,:,ii))*delta;
#    else:

    if ii == 0:
      mean_ref = np.zeros((len(problemParameters.probed_dofs), refsol.shape[1], observed_point.shape[0]))
      mean_rom = np.zeros((len(problemParameters.probed_dofs), refsol.shape[1], observed_point.shape[0]))
      std_ref = np.zeros((len(problemParameters.probed_dofs), refsol.shape[1], observed_point.shape[0]))
      if compositeCostFlag:
        if problemParameters.hyperFlag:
          sampledHDMnodes = np.loadtxt('%s/eqNodes' % IOParameters.OutputID).astype(int)
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

    #c_mean = c_mean + np.sum(mean_ref[:, :, ii] ** 2)
    #c_std = c_std + np.sum(std_ref[:, :, ii] ** 2)
    
#  if problemParameters.derivativeflag:
# %% print stiffness matrices
#  aeros_printMatrices(observed_point, problemParameters, IOParameters)
#  eval(sprintf('!%s %s/%s >& %s', IOParameters.programExec, IOParameters.OutputID, IOParameters.OutputMatrices, IOParameters.LogFileMassStiff));
#  file=file_format(observed_point, '');
#  sprintf('%s/%s.stiffness', IOParameters.ResultsMatrices, file)
#  reformatK(sprintf('%s/%s.stiffness', IOParameters.ResultsMatrices, file))
#  stiff_mat=load(sprintf('%s/%s.stiffness.new', IOParameters.ResultsMatrices, file)) ;
#  k=spconvert(stiff_mat);
#  %k=full(k);
#  K=k + k.'-diag(diag(k));
#
#  reformatK(sprintf('%s/%s.mass', IOParameters.ResultsMatrices, file))
#  mass_mat=load(sprintf('%s/%s.mass.new', IOParameters.ResultsMatrices, file)) ;
#  m=spconvert(mass_mat);
#  %m=full(m);
#  M=m + m.'-diag(diag(m));
#
#
  return std_ref, mean_ref, mean_rom, \
         tt[problemParameters.solStartI:],  mean_ref_full

################################################################################
# Implement a file format convention for parametric result files
def file_format(parameterPoint, type):
  file = type 
  for i in range(parameterPoint.shape[-1]):
    file = file + '_%.4f' % parameterPoint[i]
  return  file
 
################################################################################
# Read an xpost solution
def femsol(inpfile, nc, n):

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

  # RT - vanilla Python code - something faster needed
  array = []
  for line in f:
    for x in line.split():
      array.append(x)
  v = np.array(array, dtype=float)
  
  f.close()
  if igz != -1:
   os.system('rm /tmp/' +  inpfile[0:igz])
  
  zz = np.reshape(v, (nc * n + 1,-1), order = 'F')
  return zz[1:,:], zz[0,:]

################################################################################
# Read(&modify) Aero-S sampled mesh
def aeros_rwSampledMesh(lr, pathname, opathname, sz):
# sz == 0, just scale the nodes, do not touch anything else
# sz ~= 0, change basis name, scale nodes, and add DMASS
# Warning - this could be slow if reading a lot of nodes because the nodes
# are not pre-allocated exactly
 
  # RT - correct for now obsolete 2d scaling 
  if len(lr) == 2: 
    lr =np.array([lr[0], lr[1], lr[1]])
  # Split into three sections - before nodes including the inorm, nodes, and
  # topology and the rest, and rewrite the first part, and the modified nodes

  os.system('csplit -s %s /TOPOLOGY/' % pathname)
  os.system('csplit -f xxx -s xx00 /NODES/+1')
  f = open('xxx00','r')
  fo = open('xx00_','w')
  while True:
    line = f.readline()
    if not line:
      break
    if sz != 0 and line.find('inorm') >= 0:
      fo.write('0 inorm "SROB.orthonormalized" %d\n' % sz)
    else:
      fo.write(line)
  f.close()
  nodes = np.loadtxt('xxx01', comments = '*')
  nodes[:,1:4] = lr * nodes[:,1:4]
  for i in range(nodes.shape[0]):
    fo.write('%d  %.16e %.16e %.16e\n' % (int(nodes[i, 0]),
              nodes[i, 1], nodes[i, 2], nodes[i, 3]))
  fo.write('*\n')
  fo.close()
  if sz != 0:
    fo = open('xx02', 'w')
    print('*\nDIMASS\nMODAL\n"samplmsh.new.reducedmass"', file = fo)
    fo.close()
    os.system('cat xx00_ xx01 xx02 > %s' % opathname)
  else :
    os.system('cat xx00_ xx01 > %s' % opathname)
  
  os.system('rm xx*')
  return nodes[:,1:]

################################################################################
# Create input Aero-S input files to write stiffness & mass matrices
def aeros_MassStiff2(parameterPoint, pathSampledMesh, IOParameters):

  
  file = file_format(parameterPoint, '')
  pathOutput = '%s/%s' % (IOParameters.OutputID, IOParameters.OutputMassStiff)
  pathInput = '%s/%s' % (IOParameters.InputID, IOParameters.Input)
  pathMesh = '%s/%s' % (IOParameters.OutputID, IOParameters.Mesh)
  
  f = open(pathOutput,'w')
  
  f.write('\
DYNAMICS\n\
srom\n\
PRINTMAT "%s/%s"\n\
*\n\
NONLINEAR\n\
linearelastic\n\
*\n\
INCLUDE "%s"\n\
*\n\
INCLUDE "%s/COMPOSITE.txt"\n\
*\n\
END\n' % (pathMesh, file, pathSampledMesh, pathInput))
  
  f.close()
  return

################################################################################
# Read PBS information to initialize the list of nodes and number of cores
def getPBSInfo():
  filename = os.getenv('PBS_NODEFILE')
  f = open(filename, 'r')
  nodesn = a=f.readlines()
  unodesn = list(set(nodesn))
  # Strip \n
  unodes = []
  for unoden in unodesn:
    unodes.append(unoden.strip())
  ppn = int(os.getenv('PBS_NUM_PPN'))
  return ppn, unodes

################################################################################
# Read/generate ROB, coordinates, Dirichlet BC, mass&stiffness from Aero-s files
def readROBPlus(problemParameters, IOParameters, observed_point, eroot):

  K = []
  M = []
  if problemParameters.hyperFlag and problemParameters.sampledFlag:
      
    # Read the sampled ROB
    V, ROB_nodes, ROB_tag = readAeroSROB(
     '%s/Mesh/samplmsh.elementmesh.inc.compressed.basis.orthonormalized.out' %
      IOParameters.OutputID, problemParameters.ndofspernode, IOParameters)
    
    # Orthogonalize the basis
    [Q,R] = np.linalg.qr(V)
    V = Q
    
    # Find unconstrained dofs by reading DISPLACEMENT section of the input file
    # Assumes simple dof numbering
    BC = readAeroSInputDisp('%s/Mesh/samplmsh.elementmesh.inc' % IOParameters.OutputID)
    
    # Generate the mass matrix that corresponds to the observed parameter
    for ii in range(observed_point.shape[0]):
      # Scale the sampled mesh
      aeros_rwSampledMesh(observed_point[ii,:],
          '%s/Mesh/samplmsh.elementmesh.inc' % IOParameters.OutputID,'%s/Mesh/samplmshx.elementmesh.inc' % IOParameters.OutputID, 0)
      # Create Aero-s input file to generate the mass matrix
      aeros_MassStiff2(observed_point[ii,:],
       '%s/Mesh/samplmshx.elementmesh.inc' % IOParameters.OutputID, IOParameters)
      # Run Aero-s
      os.system('%s/aeros -v 2 %s/%s  >& %s' % (eroot, IOParameters.OutputID,
         IOParameters.OutputMassStiff, IOParameters.LogFileMassStiff))
      file = file_format(observed_point[ii, :], '')
      # RT - read matrix and convert to full matrices as Python sparse matrix
      #      support seems sparse
      M.append(mmread('%s/%s/%s.mass' % 
         (IOParameters.OutputID, IOParameters.Mesh, file)).toarray())
      K.append(mmread('%s/%s/%s.stiffness' %
         (IOParameters.OutputID, IOParameters.Mesh, file)).toarray())
      aeros_rwSampledMesh(observed_point[ii,:],
       '%s/Mesh/samplmsh.elementmesh.inc' % IOParameters.OutputID,
       '%s/Mesh/samplmsh.new%d.elementmesh.inc' % (IOParameters.OutputID, ii),
       V.shape[1])
    
    x = aeros_rwSampledMesh([1.0, 1.0, 1.0],
        '%s/Mesh/samplmsh.elementmesh.inc' % IOParameters.OutputID,
        '%s/Mesh/samplmshx.elementmesh.inc' % IOParameters.OutputID, 0)
      
  else: 
    V, ROB_nodes, ROB_tag = readAeroSROB(
        '%s/%s.out' % (IOParameters.OutputID, IOParameters.BasisOrth),
         problemParameters.ndofspernode, IOParameters)
    R = []
    
    x = np.loadtxt('%s/Input/GEOMETRY.txt' % IOParameters.InputID, usecols = (1, 2, 3), skiprows = 1)
    BC = np.loadtxt('%s/Input/DISPLACEMENTS.txt' % IOParameters.InputID,
                     skiprows = 1)
  return V, ROB_nodes, ROB_tag, x, BC, K, M, R

################################################################################

# Read exact gradients
def new_read_dydv_files(filename, n, N, uncdof, totaltimestep):
  f = open(filename, 'rb')
  f.read(8)
  d = np.fromfile(f, dtype = np.float64)
  dy = np.reshape(d, (n, totaltimestep, N, n), order = 'F')
  f.close()
  return dy[:, :, uncdof, :]
