################################################################################
# Problem-dependent functions that generate/modify/read Aero-S input_dir files
################################################################################

from aeros_utils import file_format

def solverSetup(modelType, outputFile, pathMFTT, rayleigh_damping, nonLinTol):
  """
  Set up the solver by writing the necessary configurations to the output file.

  Parameters:
  modelType (int): The type of the model.
  outputFile (str): The path to the output file.
  pathMFTT (str): The path to the MFTT file.
  rayleigh_damping (list): The Rayleigh damping parameters.
  nonLinTol (float): The non-linear tolerance.
  """
  with open(outputFile, 'w') as f:
    f.write('STATICS\n')
    f.write({
      1: 'sparse\n',
      2: 'eisgal\n',
      3: 'eisgal\n',
      4: 'mumps\n',
      6: 'eisgal\n'
    }.get(modelType, ''))
    f.write('*\n')
    f.write('DYNAMICS\n')
    if modelType != 4:
      f.write('newmark\nmech 0.25 0.5 0.0 0.0\ntime 0.0 1.8e-4 0.036\n')
    f.write(f'rayleigh_damping {rayleigh_damping[0]} {rayleigh_damping[1]}\n')
    if modelType not in [1, 5, 6]:
      f.write('srom 0\n')
    f.write('*\nMFTT\n')
    with open(pathMFTT, 'r') as fm:
      for line in fm:
        f.write(line)
    f.write('*\n')
    if modelType not in [4, 5]:
      f.write(f'NONLINEAR\nunsymmetric\nnltol {nonLinTol}\nrebuild 1\nmaxit 25\n*\n')
    if modelType == 1:
      f.write('nltol 1e-8\n*\n')

def includeProperties(input_, output_):
  """
  Include properties from the input file to the output file.

  Parameters:
  input_ (str): The path to the input file.
  output_ (str): The path to the output file.
  """
  with open(output_, 'a') as f:
    for prop in ['GEOMETRY.txt', 'TOPOLOGY.txt', 'DISPLACEMENTS.txt', 'PRESSURES.txt', 'MATERIAL.txt', 'ATTRIBUTES.txt']:
      f.write(f'INCLUDE "{input_}/{prop}"\n*\n')
    f.write('END\n')

def aeros_buildROM(parameterPoint, problemParameters, IOParameters, paths=[]):
  """
  Build the ROM for the Aero-S problem.

  Parameters:
  parameterPoint (array-like): The parameter point values.
  problemParameters (object): An object containing problem parameters.
  IOParameters (object): An object containing input/output parameters.
  paths (list): A list of paths.

  Note:
  This function has not been defined for this problem.
  """
  print('aeros_buildROM has not been defined for this problem', flush=True)

def aeros_buildHROM(parameterPoint, problemParameters, IOParameters, outputGradFlag, k, paths=[]):
  """
  Build the HROM for the Aero-S problem.

  Parameters:
  parameterPoint (array-like): The parameter point values.
  problemParameters (object): An object containing problem parameters.
  IOParameters (object): An object containing input/output parameters.
  outputGradFlag (bool): A flag indicating whether to output gradients.
  k (int): An integer parameter.
  paths (list): A list of paths.
  """
  if paths:
    pathOutput = f'{paths[1]}/{IOParameters.HROM_output}'
    pathResults = paths[2]
    pathMesh = paths[2]
    pathInput = f'{paths[0]}/{IOParameters.input_ID}/{IOParameters.input_dir}'
    pathMFTT = f'{paths[0]}/{IOParameters.input_ID}/{IOParameters.MFTT}'
  else:
    pathResults = f'{IOParameters.output_ID}/{IOParameters.HROM_results_dir}'
    pathOutput = f'{IOParameters.output_ID}/{IOParameters.HROM_output}'
    pathMesh = f'{IOParameters.output_ID}/{IOParameters.Mesh}'
    pathMFTT = f'{IOParameters.input_ID}/{IOParameters.MFTT}'
    pathInput = f'{IOParameters.input_ID}/{IOParameters.input_dir}'

  file_dis = file_format(parameterPoint, 'dis')
  file_vel = file_format(parameterPoint, 'vel')
  file_acc = file_format(parameterPoint, 'acc')

  solverSetup(3, pathOutput, pathMFTT, problemParameters.rayleigh_damping, 1e-10)
  with open(pathOutput, 'a') as f:
    if outputGradFlag:
      f.write('exactgrad\n*\n')
    f.write('OUTPUT\n')
    f.write(f'gdisplac 21 15 "{pathResults}/{file_dis}" 1\n')
    f.write(f'gvelocit 21 15 "{pathResults}/{file_vel}" 1\n')
    f.write(f'gacceler 21 15 "{pathResults}/{file_acc}" 1\n')
    if outputGradFlag:
      f.write(f'DispBasisSens 21 15 "{paths[0]}/{IOParameters.output_ID}/{IOParameters.exact_grad_output}.{k}" 1\n')
    f.write('*\n')
    f.write(f'INCLUDE "{pathMesh}/samplmsh.new.elementmesh.inc"\n*\n')
    f.write('END\n')
from aeros_utils import file_format

