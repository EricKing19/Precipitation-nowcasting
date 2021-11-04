import numpy as np
import math
import matplotlib.pyplot as plt
x = np.arange(0, 1, 0.01)
y_0 = []
y_1 = []
y_2 = []
y_3 = []
y_4 = []
epsilon = 0.5
for t in x:
    temp = t**2/(t**2 + epsilon)
    y_0.append(temp)
for t in x:
    temp = (1-t)**2/(t**2 + 1 + epsilon) + (1-t)**2/(t**2 + epsilon)
    y_1.append(temp)
# for t in x:
#     temp = (1-t)**3/(t**2 + 1 + epsilon)
#     y_4.append(temp)
for t in x:
    temp = -np.log10(t)
    y_2.append(temp)
for t in x:
    temp = -(1-t)**2 * np.log10(t)
    y_3.append(temp)
# plt.plot(x, y_0, label="Dice")
plt.plot(x, y_1, label="Dice")
plt.plot(x, y_2, label="CE")
plt.plot(x, y_3, label="Focal")
# plt.plot(x, y_4, label="DFocal")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(0, 3)
plt.legend()
plt.show()