import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib import ticker
import json

def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = np.exp(-(x - mean) ** 2 / (2 * std ** 2))# / sigma
    return x_out
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def draw_guass():
    #sin & cos曲线
    x = np.arange(1, 10, 0.1)
    # y1 = 0.1 * x
    y1 = x
    y2 = sigmoid(x)
    y3 = 10 * GaussProjection(x, x.mean(), x.std())

    plt.plot(x,y1,label="Origin",linestyle = "--")
    plt.plot(x,y3,label="Gauss")
    # plt.plot(x,y2,label="Sigmoid",linestyle='dotted')

    # plt.ylim((-0.1,1.1))
    # plt.ylim()

    plt.xlabel("Before rectification")
    plt.ylabel("After rectification")
    plt.legend()   #打上标签

    plt.savefig('./OriginAndGauss.svg', format='svg')
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

def draw_person_num():
    COCO = [98.7, 1.2, 0.1]
    CrowdPose = [92.2, 7.7, 0.1]
    OCHuman = [18.8, 80.3, 0.9]
    x = [1, 2, 3]
    width = 1.0 / 3.77
    x1 = [i - width for i in x]
    x2 = [i + width for i in x]
    total_width, n = 3.5, 13
    width = total_width / n

    bar1 = plt.bar(x1, COCO, width=width, label='COCO')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    bar2 = plt.bar(x, CrowdPose, width=width, label='CrowdPose')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    bar3 = plt.bar(x2, OCHuman, width=width, label='OCHuman')

    plt.xlabel("Number of Persons in BB(IoU > 0.5)")
    plt.ylabel("Number of BBs(%)")
    plt.legend()

    for bar in [bar1, bar2, bar3]:
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height+0.6, str(height), ha="center", va="bottom")

    plt.savefig('./IouNumPerson.svg', format='svg')
    plt.show()

def draw_subplot():
    fig = plt.figure()
    # -----------------
    coco = fig.add_subplot(2, 2, 1)
    x = np.arange(0, 1, 1.0 / 14)
    y = [68, 6.7, 5.8, 4.5, 4, 4.5, 3.5, 1, 1, 0.5, 0.5, 0.5, 0.4, 0.1]
    plt.bar(x, y, width=0.064)
    coco.yaxis.set_major_formatter(ticker.PercentFormatter())
    coco.set_title('MS COCO')

    # -----------------
    mpii = fig.add_subplot(2, 2, 2)
    x = np.arange(0, 1, 1.0 / 14)
    y = [87, 3, 2, 2, 1, 1, 1, 0.5, 0.5, 0.4, 0.4,0.4,0.4,0.4]
    plt.bar(x, y, width=0.066)
    mpii.yaxis.set_major_formatter(ticker.PercentFormatter())
    mpii.set_title('MPII')

    # -----------------
    ai = fig.add_subplot(2, 2, 3)
    x = np.arange(0, 1, 1.0 / 14)
    y = [62, 3, 6, 4, 2.5, 5.5, 3.5, 2.5, 3, 2.5, 1.9, 1.4, 1.1, 1.1]
    plt.bar(x, y, width=0.066)
    ai.yaxis.set_major_formatter(ticker.PercentFormatter())
    ai.set_title('AI Challenger')

    # -----------------
    crowdpose = fig.add_subplot(2, 2, 4)
    x = np.arange(0, 1, 0.1)
    y = [10, 10.2, 10.3, 9.7, 10, 10.3, 9.8, 9.7, 9.5, 10.5]
    plt.bar(x, y, width=0.066)
    crowdpose.yaxis.set_major_formatter(ticker.PercentFormatter())
    plt.xlim((-0.075, 1.01))
    crowdpose.set_title('CrowdPose')

    fig.tight_layout()

    plt.savefig('./CrowdIndex.svg', format='svg')
    plt.show()


def count_afi():
    afi_json = 'G:\Wei\grad\ccc\Rccc\exp\Train_two_2_32_AFILoss/run-log-tag-interference_point_count.json'
    x = []
    y = []
    with open(afi_json, 'r') as f:
        x_ys = json.load(f)
        s = set()
        for xy in x_ys:
            if xy[1] in s:
                continue
            s.add(xy[1])
            x.append(xy[1])
            y.append(xy[2])

    # infinit = np.arange(-1000, 1000, 1)
    plt.plot(x, y, label="interference keypoint num")

    # plt.ylim((-10, 420))

    plt.xlabel("epoch")
    # plt.ylabel("interference keypoint num")
    plt.legend()   #打上标签
    # plt.title("num")
    plt.tight_layout()

    plt.savefig('./count_AFI.svg', format='svg')
    plt.show()

if __name__ == '__main__':
    count_afi()
    # draw_guass()
    # draw_subplot()
    # draw_person_num()
    # x = 10
    # gt1 = 10
    # gt2 = 20
    # a = mse(x, gt1) - mse(x, gt2)
    # print(max(a, 0))