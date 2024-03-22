################################################################################
# Core NPM routines
################################################################################

import numpy as np
from scipy.linalg import fractional_matrix_power

def GtoW2(G, sigma, s, V, B, dGdB = []):
    """
    This function computes the SROB W from the random matrix G and the hyperparameters.

    Parameters:
    G (array-like): input random matrix (discretized random fields).
    sigma (array-like): The sigma values.
    s (float): The s value.
    V (array-like): The deterministic basis.
    B (array-like): The constraints matrix.
    dGdB (array-like, optional): The derivative of G with respect to beta. Defaults to an empty list.

    Returns:
    W (array-like): The computed SROB.
    dVds (array-like, optional): The derivative of V with respect to s. Only returned if dGdB is not empty.
    dVdB (array-like, optional): The derivative of V with respect to beta. Only returned if dGdB is not empty.
    dVdSig (array-like, optional): The derivative of V with respect to sigma. Only returned if dGdB is not empty.
    """
    if dGdB != []:
        U = G @ sigma
        Temp = V.T @ U
        D = 0.5 * (Temp + Temp.T)
        Z = U - V @ D

        Norm = np.identity(Z.shape[1]) + s * s * Z.T @ Z
        H = fractional_matrix_power(Norm, -1.0/2.0) 
        W = (V + s * Z) @ H

        dVds, dVdB, dVdSig = deriv_v_a(V, Z, sigma, s, G, dGdB, B)
        return W, dVds, dVdB, dVdSig

    else:
        U = G @ sigma
        A = U - B @ (B.T @ U)
        Temp = V.T @ A 
        D = 0.5 * (Temp + Temp.T)
        Z = A - V @ D

        Norm = np.identity(Z.shape[1]) + s * s * Z.T @ Z
        Eigen, MatPhi = np.linalg.eigh(Norm)
        Eigm12 = np.diag(1.0 / np.sqrt(Eigen))

        H = (MatPhi @ Eigm12) @ MatPhi.T
        return  (V + s * Z) @ H

def generateG(Gparams, beta,  x, Dof, Regular, ArrayRand1IDENTell,ArrayRand2IDENTell, derivFlag = False):
  """
  This function generates the random matrix G using the hyperparameter beta 

  Parameters:
  Gparams (object): An object containing the parameters for the G matrix.
  beta (float): The beta hyperparameter 
  x (array-like): The mesh coordinates 
  Dof (array-like): The degrees of freedom numbering 
  Regular (float): The regularization parameter for the G matrix.
  ArrayRand1IDENTell (array-like): The first array of random values for the G matrix.
  ArrayRand2IDENTell (array-like): The second array of random values for the G matrix.
  derivFlag (bool, optional): A flag indicating whether to compute the derivative of G. Defaults to False.

  Returns:
  G (array-like): The generated G matrix.
  dG (array-like, optional): The derivative of the G matrix with respect to beta. Only returned if derivFlag is True.
  """
  N = Gparams.N
  N0 = Gparams.N0
  n = Gparams.n
  m = Gparams.m
  d = Gparams.d
  nu_p = Gparams.nu_p
  R2sum_nu = Gparams.R2sum_nu
  L = Gparams.L
  K_nu = Gparams.K_nu

  L_g = beta * L
  PL_g = np.pi / L_g
  LK = np.outer(PL_g, K_nu)

  Angle2 = 2.0 * np.pi * ArrayRand1IDENTell
  cAngle2 = np.cos(Angle2)
  sAngle2 = np.sin(Angle2)
  LKx = np.zeros((N0, d, nu_p))
  for nu in range(nu_p):
    for i in range(d):
      LKx[:, i, nu] = LK[i, nu] * x[:, i]

  cLKx = np.cos(LKx)
  sLKx = np.sin(LKx)
  cLKx1 = np.squeeze(cLKx[:,0,:])
  cLKx2 = np.squeeze(cLKx[:,1,:])
  cLKx3 = np.squeeze(cLKx[:,2,:])
  sLKx1 = np.squeeze(sLKx[:,0,:])
  sLKx2 = np.squeeze(sLKx[:,1,:])
  sLKx3 = np.squeeze(sLKx[:,2,:])

  if derivFlag:
    LKx_B= LKx / beta
    LKx_B1 = np.squeeze(LKx_B[:,0,:])
    LKx_B2 = np.squeeze(LKx_B[:,1,:])
    LKx_B3 = np.squeeze(LKx_B[:,2,:])
    cLKx1_m=cLKx1 * LKx_B1
    cLKx2_m=cLKx2 * LKx_B2
    cLKx3_m=cLKx3 * LKx_B3
    sLKx1_m=sLKx1 * LKx_B1
    sLKx2_m=sLKx2 * LKx_B2
    sLKx3_m=sLKx3 * LKx_B3


  # Initial value of G
  G = np.zeros((N, n))
  dG = np.zeros((N, n))

  # Iterations for generating G
  for k in range(n):
    for l in range(m):
      LogPsi = np.sqrt(- np.log(ArrayRand2IDENTell[:,:,l,k]))
      SigLog = np.zeros((d, nu_p))

      for nu in range(nu_p):
        for i in range(d):
          SigLog[i, nu] = R2sum_nu[nu] * LogPsi[i, nu]

      Q = np.zeros((N0, d))
      dQ = np.zeros((N0, d))

      sc = SigLog * cAngle2[:,:,l,k]
      ss = SigLog * sAngle2[:,:,l,k]
      Q[:,0] = cLKx1 @ sc[0,:].T - sLKx1 @ ss[0,:].T
      Q[:,1] = cLKx2 @ sc[1,:].T - sLKx2 @ ss[1,:].T
      Q[:,2] = cLKx3 @ sc[2,:].T - sLKx3 @ ss[2,:].T


      G_Curl = (1.0 / np.sqrt(N)) * Q[:, 0]
      for i in range(1,d):
        G_Curl = G_Curl * Q[:,i]

      G[Dof[:, l], k] = Regular * G_Curl

      if derivFlag:
        dQ[:,0] = cLKx1_m @ ss[0,:].T + sLKx1_m @ sc[0,:].T
        dQ[:,1] = cLKx2_m @ ss[1,:].T + sLKx2_m @ sc[1,:].T
        dQ[:,2] = cLKx3_m @ ss[2,:].T + sLKx3_m @ sc[2,:].T

        sum = np.zeros(N0)
        for i in range(d):
          prod = dQ[:,i]
          for j in range(d):
            if j != i:
              prod = prod * Q[:,j]
          sum = sum + prod

        dG_Curl = (1.0/np.sqrt(N)) * sum
        dG[Dof[:,l], k] = Regular * dG_Curl

  if derivFlag:
    return G, dG
  else:
    return G

def deriv_IB_a(Z, s, dZda_B, dZda_Sig):
  """
  This function computes the derivative of the IB matrix with respect to s, B, and Sigma.

  Parameters:
  Z (array-like): The Z matrix.
  s (float): The s value.
  dZda_B (array-like): The derivative of Z with respect to B.
  dZda_Sig (array-like): The derivative of Z with respect to Sigma.

  Returns:
  dIBds (array-like): The derivative of the IB matrix with respect to s.
  dIBdB (array-like): The derivative of the IB matrix with respect to B.
  dIBdSig (array-like): The derivative of the IB matrix with respect to Sigma.
  """
  B = s * Z

  N, n = B.shape

  # Component-wise derivative 
  dIBds = 2 * s * (Z.T @ Z)
  dZdB = dZda_B
  dIBdB = s**2 * (dZdB.T @ Z + Z.T @ dZdB);
  dZdSig=dZda_Sig
  n_sigma = int((n * (n + 1)) / 2)
  A1 = np.reshape(Z.T @ np.reshape(dZdSig, (N, n * n_sigma), order = 'F'),
          (n, n, n_sigma), order = 'F')
  A1 = np.transpose(A1, (1, 0, 2))
  A2 = np.reshape(Z.T @ np.reshape(dZdSig, (N, n*n_sigma), order = 'F'),
          (n, n, n_sigma), order = 'F')
  dIBdSig = s**2 * (A1 + A2)

  return dIBds, dIBdB, dIBdSig

def deriv_z_a(V, sigma, G, dGdB):
  """
  This function computes the derivative of Z with respect to B and Sigma.

  Parameters:
  V (array-like): The V matrix.
  sigma (array-like): The sigma values.
  G (array-like): The G matrix.
  dGdB (array-like): The derivative of G with respect to B.

  Returns:
  dZda_B (array-like): The derivative of Z with respect to B.
  dZda_Sig (array-like): The derivative of Z with respect to Sigma.
  """
  N, n = G.shape

  # Component-wise derivatives
  dUdB = dGdB @ sigma

  dsigma_da = diffSigma(n)
  n_sigma = int(n * (n+1)) / 2
  dUdSig = np.einsum('ij,jkl->ikl', G, dsigma_da)

  dUtdSig = np.transpose(dUdSig, (1, 0, 2))

  dUtdB= dUdB.T 

  dZdB = dUdB -0.5 * V @ (V.T @ dUdB + dUtdB @ V)

  dUtdSig_times_V = np.einsum('ij,jkl->ikl', V.T, dUdSig)
  dUtdSig_times_V = np.transpose(dUtdSig_times_V, (1, 0, 2))

  temp2 = np.einsum('ij,jkl->ikl', V, dUtdSig_times_V)
  temp0 = np.einsum('ij,jkl->ikl', V.T, dUdSig)
  temp1 = np.einsum('ij,jkl->ikl', V, temp0)

  dZda_Sig = dUdSig - 0.5 * temp1 - 0.5 * temp2

  dZda_B = dZdB

  return dZda_B, dZda_Sig

def deriv_sqinv(s, Z, dZda_B, dZda_Sig):
  """
  This function computes the derivative of the square inverse matrix with respect to s, B, and Sigma.

  Parameters:
  s (float): The s hyperparameter.
  Z (array-like): The Z matrix.
  dZda_B (array-like): The derivative of Z with respect to B.
  dZda_Sig (array-like): The derivative of Z with respect to Sigma.

  Returns:
  dIBsq_ds (array-like): The derivative of the square inverse matrix with respect to s.
  dIBsq_dB (array-like): The derivative of the square inverse matrix with respect to B.
  dIBsq_dSig (array-like): The derivative of the square inverse matrix with respect to Sigma.
  sqA (array-like): The square inverse matrix.
  """
   
  n = Z.shape[1] 
  A = np.identity(Z.shape[1]) + s * s * Z.T @ Z

  sqA = fractional_matrix_power(A, -1.0/2.0)
  dsqAdA = np.linalg.inv(np.kron(np.identity(n), sqA) +
                        np.kron(sqA, np.identity(n)))
  
  invA = np.linalg.inv(A)
  dAinvdA = np.kron(-invA, invA)
  
  D = np.zeros((n, n , n, n))
  E = np.zeros((n, n , n, n))
  
  for i in range(dsqAdA.shape[1]):
    m = i % n
    p = int(i / n) 
    D[:,:,m,p] = np.reshape(dsqAdA[:,i], (n, n), order = 'F')
    E[:,:,m,p] = np.reshape(dAinvdA[:,i], (n, n), order = 'F')
   
  F = np.einsum('ijkl,klmn->ijmn', D, E)
  dIBds, dIBdB, dIBdSig = deriv_IB_a(Z, s, dZda_B, dZda_Sig)
  
  # Componentwise derivative 
  dIBsq_ds = np.einsum('ijmn,mn->ij', F, dIBds)
  dIBsq_dB = np.einsum('ijmn,mn->ij', F, dIBdB)
  dIBsq_dSig = np.einsum('ijkl,klm->ijm', F, dIBdSig)

  return dIBsq_ds, dIBsq_dB, dIBsq_dSig, sqA

def deriv_v_a(V, Z, sigma, s, G, dGdB, B):
    """
    This function computes the derivative of V with respect to s, B, and Sigma.

    Parameters:
    V (array-like): The deterministic basis.
    Z (array-like): The Z matrix.
    sigma (array-like): The sigma values.
    s (float): The s hyperparameter.
    G (array-like): The G matrix.
    dGdB (array-like): The derivative of G with respect to beta.
    B (array-like): The B matrix.

    Returns:
    dVds (array-like): The derivative of V with respect to s.
    dVdB (array-like): The derivative of V with respect to beta.
    dVdSig (array-like): The derivative of V with respect to Sigma.
    """
    N, n = V.shape

    dZda_B, dZda_Sig = deriv_z_a(V,sigma, G, dGdB)
  
    dIBsq_ds, dIBsq_dB, dIBsq_dSig, sqA = deriv_sqinv(s, Z, dZda_B, dZda_Sig)
  
  
    # Component-wise derivatives 
    dvsz_ds = Z
    dvsz_dB = s * dZda_B
    dvsz_dSig = s * dZda_Sig
  
    dVds = dvsz_ds @ sqA + (V + s*Z) @ dIBsq_ds

    dVdB = dvsz_dB @ sqA + (V + s*Z) @ dIBsq_dB; 

    n_sigma = (n * (n+1)) / 2
    dVdSig = np.einsum('ikm,kj->ijm', dvsz_dSig, sqA) + \
        np.einsum('ik,kjm->ijm', V + s * Z, dIBsq_dSig)
  
    return dVds, dVdB, dVdSig

def diffSigma(n):
    """
    This function computes the derivative of Sigma.

    Parameters:
    n (int): The size of the Sigma matrix.

    Returns:
    dsigma_da (array-like): The derivative of Sigma.
    """
    size_alpha = int(n * (n + 1) / 2 + 2)
    dsigma_da = np.zeros((n, n, int(n * (n + 1) / 2)))
    k = 0
    for i in range(n):
        for j in range(i,n):
            dsigma_da[i, j, k] = 1
            k = k + 1
    return dsigma_da
