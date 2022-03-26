import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = np.exp(-(x - mean) ** 2 / (2 * std ** 2))# / sigma
    return x_out
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def draw_guass():
    #sin & cos曲线
    x = np.arange(-10, 10, 0.1)
    y1 = 0.1 * x
    y2 = sigmoid(x)
    y3 = GaussProjection(x, x.mean(), x.std())

    plt.plot(x,y1,label="Origin",linestyle = "--")
    plt.plot(x,y3,label="Gauss")
    plt.plot(x,y2,label="Sigmoid",linestyle='dotted')

    plt.ylim((-0.1,1.1))
    # plt.ylim()

    plt.xlabel("Before rectification")
    plt.ylabel("After rectification")
    plt.legend()   #打上标签

    plt.savefig('./drawGauss.svg', format='svg')
    plt.show()

def mse(x, gt):
    return np.sum((x - gt)**2)

def afi(x, gt1, gt2):
    return mse(x, gt1) + max(mse(x, gt1) - mse(x, gt2), 0)

def draw_afi():
    x = np.arange(-10, 30, 1)
    g1 = 10
    g2 = 20
    gt1 = np.full(x.shape, g1)
    gt2 = np.full(x.shape, g2)

    y = [afi(x[i], gt1[i], gt2[i])  for i in range(len(x))]
    y_mse = [mse(x[i], gt1[i])  for i in range(len(x))]

    infinit = np.arange(-1000, 1000, 1)
    g1x = np.full(infinit.shape, g1)
    g2x = np.full(infinit.shape, g2)
    plt.plot(x,y_mse, linestyle=':',marker = '*',label="MSE")
    plt.plot(x,y, label="AFI")
    plt.plot(g1x, infinit, label="GT1",linestyle = "--", color='g')
    plt.plot(g2x, infinit, label="GT2",linestyle = "--", color='r')

    plt.ylim((-10, 420))

    plt.xlabel("Predicted Value")
    plt.ylabel("Loss")
    plt.legend()   #打上标签

    plt.savefig('./drawAFI.png', format='png')
    plt.show()

if __name__ == '__main__':
    draw_afi()
    # x = 10
    # gt1 = 10
    # gt2 = 20
    # a = mse(x, gt1) - mse(x, gt2)
    # print(max(a, 0))