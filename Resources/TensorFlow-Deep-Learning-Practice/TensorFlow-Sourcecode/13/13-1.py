import struct
import matplotlib.pyplot as plt
import  numpy as np
dateMat = np.ones((7,7))

kernel = np.array([[2,1,1],[3,0,1],[1,1,0]])

def convolve(dateMat,kernel):
    m,n = dateMat.shape
    km,kn = kernel.shape
    newMat = np.ones(((m - km + 1),(n - kn + 1)))
    tempMat = np.ones(((km),(kn)))
    for row in range(m - km + 1):
        for col in range(n - kn + 1):
            for m_k in range(km):
                for n_k in range(kn):
                    tempMat[m_k,n_k] = dateMat[(row + m_k),(col + n_k)] * kernel[m_k,n_k]
            newMat[row,col] = np.sum(tempMat)

    return newMat
