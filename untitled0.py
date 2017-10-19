# import matplotlib as mpl
import scipy
import numpy as np

R = np.array([[0.9688,-0.0251,0.2457],
              [0.0405,0.9995,-0.0035],
              [-0.2454,0.0113,0.9693]])
T = np.array([[-25.8063],
              [-0.9851],
              [4.2503]])
WL = np.array([[414.1067,0,185.1447],
               [0,409.9127,129.4448],
               [0,0,1]])
Wr = np.array([[404.0648,0,182.3743],
               [0,398.4736,116.8343],
               [0,0,1]])
tx = float(T[0,0])
ty = float(T[1,0])
tz = float(T[2,0])
S = np.array([[0,-tz,ty],
              [tz,0,-tx],
              [-ty,tx,0]])
F = np.dot(np.dot(np.dot(np.linalg.inv(WL.transpose()),S.transpose()),R),np.linalg.inv(Wr))

def null(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

print(null(F))