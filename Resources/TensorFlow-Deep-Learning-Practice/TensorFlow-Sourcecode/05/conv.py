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
