# MCMCS Coursework 1
# Luc Berthouze 2020-11-01

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


# Definition of Simple landscape
def SimpleLandscape(x, y):
    return np.where(1 - np.abs(2 * x) > 0, 1 - np.abs(2 * x) + x + y, x + y)


# Definition of gradient of Simple landscape
def SimpleLandscapeGrad(x, y):
    g = np.zeros(2)
    if 1 - np.abs(2 * x) > 0:
        if x < 0:
            g[0] = 3
        elif x == 0:
            g[0] = 0
        else:
            g[0] = -1
    else:
        g[0] = 1
    g[1] = 1
    return g


# Function to draw a surface (equivalent to ezmesh in Matlab)
# See argument cmap of plot_surface instruction to adjust color map (if so desired)
def DrawSurface(fig, varxrange, varyrange, function):
    """Function to draw a surface given x,y ranges and a function."""
    ax = fig.gca(projection='3d')
    # ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    xx, yy = np.meshgrid(varxrange, varyrange,  sparse=False)
    z = function(xx, yy)
    # ax.plot_surface(xx, yy, z, cmap='RdBu')  # color map can be adjusted, or removed!
    ax.contour(xx, yy, z, zdir='y', offset=8, cmap='rainbow')
    ax.contour(xx, yy, z, zdir='z', offset=-14, cmap='rainbow')
    fig.canvas.draw()
    return ax


# Function implementing gradient ascent
def GradAscent(StartPt, NumSteps, LRate):
    PauseFlag = 1
    for i in range(NumSteps):
        # TO DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape

        height = SimpleLandscape(StartPt[0], StartPt[1])
        print(StartPt[0], StartPt[1], height)

        # TO DO: Plot point on the landscape 
        # Use a markersize that you can see well enough (e.g., * in size 10)
        plt.plot(StartPt[0], StartPt[1], height, marker="o", markersize=10)

        # TO DO: Calculate the gradient at StartPt using SimpleLandscapeGrad or ComplexLandscapeGrad

        gradient = SimpleLandscapeGrad(StartPt[0], StartPt[1])

        # TO DO: Calculate the new point and update StartPt
        StartPt[0] += LRate * gradient[0]
        StartPt[1] += LRate * gradient[1]

        # Ensure StartPt is within the specified bounds (un/comment relevant lines)
        StartPt = np.maximum(StartPt, [-2, -2])
        StartPt = np.minimum(StartPt, [2, 2])
        # StartPt = np.maximum(Start,[-3,-3])
        # StartPt = np.minimum(StartPt,[7,7])

        # Pause to view output
        # if PauseFlag:
        #     x = plt.waitforbuttonpress()


# TO DO: Mutation function
# Returns a mutated point given the old point and the range of mutation
def Mutate(OldPt, MaxMutate):
    # TO DO: Select a random distance MutDist to mutate in the range (-MaxMutate,MaxMutate)
    MutatedPt = OldPt
    MutDist = random.uniform(-MaxMutate, MaxMutate)
    # TO DO: Randomly choose which element of OldPt to mutate and mutate by MutDist
    if random.randint(0, 1) == 0:
        MutatedPt[0] += MutDist
    else:
        MutatedPt[1] += MutDist
    return MutatedPt


# Function implementing hill climbing
def HillClimb(StartPt, NumSteps, MaxMutate):
    PauseFlag = 1
    for i in range(NumSteps):
        # TO DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape
        height = ComplexLandscape(StartPt[0], StartPt[1])
        print(StartPt[0], StartPt[1], height)

        # TO DO: Plot point on the landscape 
        # Use a markersize that you can see well enough (e.g., * in size 10)
        plt.plot(StartPt[0], StartPt[1], height, marker="*", markersize=10)

        # Mutate StartPt into NewPt
        NewPt = Mutate(np.copy(StartPt),
                       MaxMutate)  # Use copy because Python passes variables by references (see Mutate function)

        # Ensure NewPt is within the specified bounds (un/comment relevant lines)
        # NewPt = np.maximum(NewPt, [-2, -2])
        # NewPt = np.minimum(NewPt, [2, 2])
        NewPt = np.maximum(NewPt, [-3, -3])
        NewPt = np.minimum(NewPt, [7, 7])

        # TO DO: Calculate the height of the new point
        NewHeight = ComplexLandscape(StartPt[0], StartPt[1])
        # TO DO: Decide whether to update StartPt or not
        if NewHeight > height:
            StartPt = NewPt
        # Pause to view output
        # if PauseFlag:
        #     x = plt.waitforbuttonpress()
        #     break


# Template
# Plot the landscape (un/comment relevant line)
plt.ion()
fig = plt.figure()
# ax = DrawSurface(fig, np.arange(-2, 2.025, 0.025), np.arange(-2, 2.025, 0.025), SimpleLandscape)
ax = DrawSurface(fig, np.arange(-3, 7.025, 0.025), np.arange(-3, 7.025, 0.025), ComplexLandscape)

# Enter maximum number of iterations of the algorithm, learning rate and mutation range
NumSteps = 50
LRate = 0.6
MaxMutate = 1

# TO DO: choose a random starting point with x and y in the range (-2, 2) for simple landscape, (-3,7) for complex landscape
StartPt = []

SimpleStart = -2
SimpleEnd = 2
StartPt.append(random.uniform(SimpleStart, SimpleEnd))
StartPt.append(random.uniform(SimpleStart, SimpleEnd))

# ComplexStart = -3
# ComplexEND = 7
# StartPt.append(random.uniform(ComplexStart, ComplexEND))
# StartPt.append(random.uniform(ComplexStart, ComplexEND))

# Find maximum (un/comment relevant line)
# StartPt.clear()
# StartPt.append(0.05)
# StartPt.append(0.05)
# GradAscent(StartPt, NumSteps, LRate)
# HillClimb(StartPt, NumSteps, MaxMutate)
plt.ioff()
plt.show()
