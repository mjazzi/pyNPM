################################################################################
# Problem-dependent functions that generate/modify/read Aero-S input_dir files
################################################################################

from aeros_utils import file_format

################################################################################
def solverSetup(modelType,outputFile, pathMFTT, rayleigh_damping, nonLinTol):
  
  f = open(outputFile, 'w')
  
  f.write('STATICS\n')
  
  if modelType == 1:
    f.write('sparse\n')
  elif modelType == 2 or modelType == 3 or modelType == 6:
    f.write('eisgal\n')
  elif modelType == 4:
    f.write('mumps\n')
  f.write('*\n');
  
  f.write('DYNAMICS\n')
  if modelType !=4:
    f.write('newmark\n')
    f.write('mech 0.25 0.5 0.0 0.0\n')
    f.write('time 0.0 1.8e-4 0.036\n')
  f.write('rayleigh_damping %e %e\n' % (rayleigh_damping[0], rayleigh_damping[1]))
  if modelType !=1 and  modelType != 5 and modelType != 6:
     f.write('srom 0\n')
  f.write('*\n')
  f.write('MFTT\n')
  fm = open(pathMFTT, 'r')
  for line in fm:
    f.write('%s' % line)
  fm.close()
  f.write('*\n')
  
  if (modelType != 4 and modelType !=5):
    f.write('NONLINEAR\nunsymmetric\nnltol %e\nrebuild 1\nmaxit 25\n*\n' %
       nonLinTol)
  if (modelType == 1):
    f.write('nltol 1e-8\n*\n')
  
  f.close() 
  return

################################################################################
def includeProperties(input_, output_):

  f = open(output, 'a')
  
  
  f.write('INCLUDE "%s/GEOMETRY.txt"\n' % input_)
  f.write('*\n')
  f.write('INCLUDE "%s/TOPOLOGY.txt"\n' % input_)
  f.write('*\n')
  f.write('INCLUDE "%s/DISPLACEMENTS.txt"\n' % input_)
  f.write('*\n')
  f.write('INCLUDE "%s/PRESSURES.txt"\n' % input_)
  f.write('*\n')
  f.write('INCLUDE "%s/MATERIAL.txt"\n' % input_)
  f.write('*\n')
  f.write('INCLUDE "%s/ATTRIBUTES.txt"\n' % input_)
  f.write('*\n')
  f.write('END\n')
  
  f.close()

################################################################################
def aeros_buildROM(parameterPoint, problemParameters, IOParameters, paths = []):
  print('aeros_buildROM has not been defined for this problem', flush = True)
  return

################################################################################
def aeros_buildHROM(parameterPoint, problemParameters, IOParameters,
                    outputGradFlag, k, paths = []):

  if paths:
    pathOutput = '%s/%s' % (paths[1], IOParameters.HROM_output)
    pathResults = paths[2]
    pathMesh =  paths[2]
    pathInput = '%s/%s/%s' % (paths[0], IOParameters.input_ID,IOParameters.input_dir)
    pathMFTT = '%s/%s/%s' % (paths[0], IOParameters.input_ID, IOParameters.MFTT)
  else:
    pathResults= '%s/%s' % (IOParameters.output_ID, IOParameters.HROM_results_dir)
    pathOutput= '%s/%s'% (IOParameters.output_ID, IOParameters.HROM_output)
    pathMesh= '%s/%s' % (IOParameters.output_ID, IOParameters.Mesh)
    pathMFTT= '%s/%s' % (IOParameters.input_ID, IOParameters.MFTT)
    pathInput = '%s/%s' % (IOParameters.input_ID, IOParameters.input_dir)

  file_dis = file_format(parameterPoint, 'dis')
  file_vel = file_format(parameterPoint, 'vel')
  file_acc = file_format(parameterPoint, 'acc')

  solverSetup(3, pathOutput, pathMFTT, problemParameters.rayleigh_damping, 1e-10)
  f = open(pathOutput, 'a')
  if outputGradFlag:
    f.write('exactgrad\n');
    f.write('*\n')
  f.write('OUTPUT\n')
  f.write('gdisplac 21 15 "%s/%s" 1\n' % (pathResults, file_dis))
  f.write('gvelocit 21 15 "%s/%s" 1\n' % (pathResults, file_vel))
  f.write('gacceler 21 15 "%s/%s" 1\n' % (pathResults, file_acc))
  if outputGradFlag:
    f.write('DispBasisSens 21 15 "%s/%s/%s.%d" 1\n' % (paths[0],IOParameters.output_ID,  IOParameters.exact_grad_output, k))
  f.write('*\n')
  f.write('INCLUDE "%s/samplmsh.new.elementmesh.inc"\n' % pathMesh)
  f.write('*\n')
  f.write('END\n')
  f.close()
  return
