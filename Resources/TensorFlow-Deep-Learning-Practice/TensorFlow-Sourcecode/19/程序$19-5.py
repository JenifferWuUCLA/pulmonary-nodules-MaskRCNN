import numpy as np
import matplotlib.pyplot as plt

def produce_batch(batch_size, noise=0.3):
    xs = np.random.random(size=[batch_size, 1]) * 10
    ys = np.cos(xs) + 5 + np.random.normal(size=[batch_size, 1], scale=noise)
    return [xs.astype(np.float32), ys.astype(np.float32)]

if __name__ == "__main__" :
    x_train, y_t rain = produce_batch(200)
    x_test, y_test = produce_batch(200)
    plt.scatter(x_train,y_train,marker="8")
    plt.scatter(x_test, y_test,marker="*")
    plt.show()
