import time
from typing import List

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random


# Definition of Complex landscape
def ComplexLandscape(x, y):
    return 4 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 15 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - (1. / 3) * np.exp(-(x + 1) ** 2 - y ** 2) - 1 * (
                   2 * (x - 3) ** 7 - 0.3 * (y - 4) ** 5 + (y - 3) ** 9) * np.exp(-(x - 3) ** 2 - (y - 3) ** 2)


# Definition of gradient of Complex landscape
def ComplexLandscapeGrad(x, y):
    g = np.zeros(2)
    g[0] = -8 * np.exp(-(x ** 2) - (y + 1) ** 2) * ((1 - x) + x * (1 - x) ** 2) - 15 * np.exp(-x ** 2 - y ** 2) * (
            (0.2 - 3 * x ** 2) - 2 * x * (x / 5 - x ** 3 - y ** 5)) + (2. / 3) * (x + 1) * np.exp(
        -(x + 1) ** 2 - y ** 2) - 1 * np.exp(-(x - 3) ** 2 - (y - 3) ** 2) * (
                   14 * (x - 3) ** 6 - 2 * (x - 3) * (2 * (x - 3) ** 7 - 0.3 * (y - 4) ** 5 + (y - 3) ** 9))
    g[1] = -8 * (y + 1) * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 15 * np.exp(-x ** 2 - y ** 2) * (
            -5 * y ** 4 - 2 * y * (x / 5 - x ** 3 - y ** 5)) + (2. / 3) * y * np.exp(
        -(x + 1) ** 2 - y ** 2) - 1 * np.exp(-(x - 3) ** 2 - (y - 3) ** 2) * (
                   (-1.5 * (y - 4) ** 4 + 9 * (y - 3) ** 8) - 2 * (y - 3) * (
                   2 * (x - 3) ** 7 - 0.3 * (y - 4) ** 5 + (y - 3) ** 9))
    return g


# Function implementing gradient ascent
def GradAscent(StartPt, NumSteps, LRate, top):
    ArriveFlag = False
    for i in range(NumSteps):
        # height = SimpleLandscape(StartPt[0], StartPt[1])
        # gradient = SimpleLandscapeGrad(StartPt[0], StartPt[1])
        # StartPt = np.maximum(StartPt, [-2, -2])
        # StartPt = np.minimum(StartPt, [2, 2])

        height = ComplexLandscape(StartPt[0], StartPt[1])
        gradient = ComplexLandscapeGrad(StartPt[0], StartPt[1])
        StartPt = np.maximum(StartPt, [-3, -3])
        StartPt = np.minimum(StartPt, [7, 7])

        StartPt[0] += LRate * gradient[0]
        StartPt[1] += LRate * gradient[1]

        if height >= top:
            ArriveFlag = True
            arrive.append(True)
            num.append(i)
            break

    if not ArriveFlag:
        # print(StartPt)
        arrive.append(ArriveFlag)
    return ArriveFlag


# Returns a mutated point given the old point and the range of mutation
def Mutate(OldPt, MaxMutate):
    MutatedPt = OldPt
    MutDist = random.uniform(-MaxMutate, MaxMutate)
    # TO DO: Randomly choose which element of OldPt to mutate and mutate by MutDist
    if random.randint(0, 1) == 0:
        MutatedPt[0] += MutDist
    else:
        MutatedPt[1] += MutDist
    return MutatedPt


# Function implementing hill climbing
def HillClimb(StartPt, NumSteps, MaxMutate, top):
    # PauseFlag = 1
    ArriveFlag = False
    for i in range(NumSteps):
        height = ComplexLandscape(StartPt[0], StartPt[1])
        # height = SimpleLandscape(StartPt[0], StartPt[1])

        # print(StartPt[0], StartPt[1], height)
        NewPt = Mutate(np.copy(StartPt),
                       MaxMutate)
        # NewPt = np.maximum(NewPt, [-2, -2])
        # NewPt = np.minimum(NewPt, [2, 2])
        NewPt = np.maximum(NewPt, [-3, -3])
        NewPt = np.minimum(NewPt, [7, 7])

        NewHeight = ComplexLandscape(StartPt[0], StartPt[1])
        # NewHeight = SimpleLandscape(NewPt[0], NewPt[1])

        if NewHeight > height:
            StartPt = NewPt
        if height >= top:
            # print(NewHeight)
            ArriveFlag = True
            arrive.append(ArriveFlag)
            num.append(i)
            break
    if not ArriveFlag:
        arrive.append(ArriveFlag)


# Randomly generate the initial point of a complex function
def ComplexStartPt():
    ComStartPt = []
    ComplexStart = -3
    ComplexEND = 7
    ComStartPt.append(random.uniform(ComplexStart, ComplexEND))
    ComStartPt.append(random.uniform(ComplexStart, ComplexEND))
    return ComStartPt


# 遍历所有点并绘制复杂函数的图像
# Iterate through all points and draw a graph of complex functions
def draw_Complex(LRate, MaxMutate):
    plist = np.linspace(start=-3, stop=7, num=501, endpoint=True)
    # print(plist)
    xx, yy = np.meshgrid(plist, plist)
    # print(xx)
    fig, ax = plt.subplots()
    temp = 0
    for x in plist:
        for y in plist:
            # GradAscent([x, y], NumSteps, LRate, top)
            HillClimb([x, y], NumSteps, MaxMutate, top)
        # print(str(temp / 5) + '%', end='\n')
        temp += 1
    arrive_color = np.array(arrive)
    # title = 'GradAscent LRate=' + str(LRate) + 'NumSteps=' + str(NumSteps)
    # print('GradAscent_avg', sum(num) / len(num))
    # print('GradAscent_Rate of arrive', arrive.count(True) / len(arrive))

    title = 'HillClimb Mutate=' + str(MaxMutate) + 'NumSteps=' + str(NumSteps)
    print('HillClimb_avg', sum(num) / len(num))
    print('HillClimb_Rate of arrive', arrive.count(True) / len(arrive))

    # ax.scatter(xx, yy, c=arrive)
    # plt.show()

    plt.title(title)
    plt.pcolormesh(xx, yy, arrive_color.reshape(xx.shape), shading='auto')
    plt.savefig(title + '.png')
    plt.close()


NumSteps = 100
LRate = 0.14
# MaxMutate = 1
top = 12

global num, arrive
# Record the number of times required to reach the optimal solution
num = list()
# Record whether the optimal solution is reached
arrive = list()


def main():
    # Complex_test()
    # draw_Simple()
    plist = np.linspace(start=1, stop=2.4, num=8, endpoint=True)
    print(plist)
    for i in plist:
        start = time.time()
        draw_Complex(LRate, MaxMutate=i)
        end = time.time()
        print('MaxMutate = %.2f,spend time %.2f s' % i, (end - start))


if __name__ == '__main__':
    main()
