import numpy as np
import time
import os
import multiprocessing
from joblib import Parallel, delayed

################################################################################
class NodeProcessor:
  def __init__(self, _N_sim, _m_mu, _exec_nodes, _ncores_per_node, _oid):
    self.N_sim = _N_sim
    self.m_mu = _m_mu
    self.exec_nodes = _exec_nodes
    self.ncores_per_node = _ncores_per_node
    self.oid = _oid

    # Create unique directory to store files for the SROM realizations
    wid = os.getpid()
    self.proot = os.getcwd()
    t = time.time()
    timestamp = np.fix(1000*t)
    self.dirname = 'shrom%d_%015.0f' % (wid, timestamp)
    self.root = '%s/%s/%s' % \
             (self.proot, self.oid, self.dirname)
    self.croot2 = '.' # relative location of files on a compute node
  
    os.system('mkdir %s' % (self.root) )

  ##############################################################################
  # Return directories where batch jobs will be written
  def getDirs(self):
    return self.root, self.proot, self.croot2

  ##############################################################################
  # Cleanup temporary files on the compute node
  def cleanNode(self, node):
    command = 'ssh -o ForwardX11=no %s "rm -rf /dev/shm/shrom*"' % node
    os.system(command)

  ##############################################################################
  # Transfer and run jobs on the compute nodes
  def runJobs(self, parallelFlag, batch):
    # Cleanup shared memory directories on the compute nodes
    tcleanup = time.time()
    n_compute_nodes = len(self.exec_nodes)
    print(self.exec_nodes,flush = True)
    if parallelFlag:
      iis = [i for i in range(n_compute_nodes)]
      Parallel(n_jobs = int(self.ncores_per_node / 2),
               prefer="threads") (delayed(self.cleanNode)(self.exec_nodes[i]) for i in iis)
    else:
      for i in range(n_compute_nodes):
        self.cleanNode(self.exec_nodes[i])
    print('Cleanup time: ', time.time() - tcleanup, flush = True)

    # Transfer & run the batch files on the compute nodes
    njobs = self.m_mu * self.N_sim
    ncycles = int(np.ceil(njobs / (self.ncores_per_node * n_compute_nodes)))
    job_index = 0
  
    #RT from sgd
    non_analytical_realizations = np.array(list(
                          set(np.arange(self.N_sim, dtype = int)) - set(batch)))
    job_distribution_ = np.copy(batch)
    for i in range(1, self.m_mu):
      job_distribution_ = np.hstack((job_distribution_, int(i * self.N_sim) + batch))
    for i in range(0, self.m_mu):
      job_distribution_ = np.hstack((job_distribution_,
                                  int(i * self.N_sim) + non_analytical_realizations))
    job_distribution = np.reshape(job_distribution_, (-1, self.ncores_per_node),
                                   order = 'C').flatten(order = 'F')
    print('job_distribution =', job_distribution, flush = True)

    for c in range(ncycles):
      print('Entering cycle: ', c + 1, ' of ', ncycles, flush = True)
      tnodes = time.time()
      ncyclejobs = njobs - job_index
      if ncyclejobs > self.ncores_per_node * n_compute_nodes:
        ncyclejobs = self.ncores_per_node * n_compute_nodes
      for ii in range(n_compute_nodes):
        print('Running on node ', ii + 1, ' of ', n_compute_nodes)
        ff = open('%s/f_%d' % (self.root, ii),'w')
        for ji in range(job_index, min(job_index + self.ncores_per_node, njobs)):
          jobi = job_distribution[ji]
          ff.write('%s/%d\n' % (self.dirname, jobi))
        ff.close()
        # Write the master batch file that handles
        f = open('%s/mb%d_%d' % (self.root, c, ii), 'w')
        #   - cleanup on the compute node
        f.write('rm -rf /dev/shm/%s\n' % self.dirname)
        #   - transfer of directories
        f.write('(cd %s ; tar zcf - --files-from=%s/f_%d)|(cd /dev/shm;  tar  zxf -)\n' % (self.oid, self.root, ii))
        f.write('cd /dev/shm/%s\n' % self.dirname)
        #   - launching of batches within the directories
        for ji in range(job_index, min(job_index + self.ncores_per_node, njobs)):
          jobi = job_distribution[ji]
          f.write('cd  %d\n' % jobi)
          f.write('bash ./batch >& ./log_batch &\n')
          f.write('cd  ..\n')
        f.close()
        job_index = job_index + self.ncores_per_node
      print('Master batch writing time: ', time.time() - tnodes, flush = True)
  
      # Launch master batch files
      tlaunch = time.time()
      time.sleep(0.2)
      for ii in range(n_compute_nodes):
        os.system('ssh -o ForwardX11=no %s "cd %s; bash %s/mb%d_%d >& %s/log_mb%d_%d" & ' % (self.exec_nodes[ii], self.proot, self.root, c, ii, self.root, c, ii))
      print('Launch time: ', time.time() - tlaunch, flush = True)
      trun = time.time()
      # Wait for completion
      cnt = 0
      while cnt != ncyclejobs:
        time.sleep(0.2)
        stream = os.popen('ls %s/done* 2>/dev/null ' % self.root)
        done_list = stream.read()
        cnt = done_list.count('done')
      print('Run time: ', time.time() - trun, flush = True)
      os.system('rm %s/done*' % self.root)

################################################################################
