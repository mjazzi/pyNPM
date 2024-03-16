import numpy as np
from SROM import SROMAeroS, initHyperParameters, ObjectiveF
from scipy.optimize import minimize, Bounds
from scipy.io import mmread, loadmat


class SROMAeroS1(SROMAeroS):

  def __init__(self, problemPath):
    super().__init__(problemPath)

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
    #RT - testing p4
    #self.problemParameters.N_sim = 48
    #arand = loadmat('arand_p4.mat')
    #self.ArrayRand1IDENT = arand['ArrayRand1IDENT']
    #self.ArrayRand2IDENT = arand['ArrayRand2IDENT']

    bnds = Bounds(lb, ub)
    if not self.problemParameters.useR:
      R = []

    # Cost function weights and mean_ref_full
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
      if self.problemParameters.compositeCostFlag:
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
    if self.problemParameters.derivativeFlag:
      jac = True
    else:
      jac = '2-point'

#    #opt_params = minimize(of.objective, alpha_0, callback = of.callback_sgd_tr,
    opt_params = minimize(of.objective, alpha_0, callback = of.callback_sgd,
                                     jac = jac, bounds = bnds,
                                     method='SLSQP',
                                     options = { 'maxiter': nit, 'disp': True })
    #                                    method='TNC',
    #                                    method='SLSQP',
    #                                    method='L-BFGS-B',
    #                                    method='trust-constr',

#    opt_params = basinhopping(of.objective, alpha_0, minimizer_kwargs={'method':'SLSQP', 'jac':jac, 'callback':of.callback_sgd, 'bounds':bnds}, disp=True, stepsize=2, T=2)


    # Print and save the optimal values
    print('params=', opt_params.x)
    np.save('opt_params.npy', opt_params.x)
    return opt_params.x
    #return alpha_0

################################################################################
#sROM = SROMAeroS1('../CF_Input4')
#alpha = sROM.optimize(60)
#sol, mean_rom2 = sROM.postprocessing(alpha, 256)
#sROM.plot(sol, sROM.mean_ref, mean_rom2, 0, 0, 0.95, 0)

