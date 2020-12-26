import numpy as np


def ComplexLandscape(x, y):
    return 4 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 15 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - (1. / 3) * np.exp(-(x + 1) ** 2 - y ** 2) - 1 * (
                   2 * (x - 3) ** 7 - 0.3 * (y - 4) ** 5 + (y - 3) ** 9) * np.exp(-(x - 3) ** 2 - (y - 3) ** 2)


def SimpleLandscape(x, y):
    return np.where(1 - np.abs(2 * x) > 0, 1 - np.abs(2 * x) + x + y, x + y)


plist = np.linspace(start=-2, stop=2, num=1000, endpoint=True)
maxz = -1000000
for x in plist:
    for y in plist:
        maxz = max(maxz, ComplexLandscape(x, y))
        # maxz = max(maxz, SimpleLandscape(x, y))
print(maxz)
