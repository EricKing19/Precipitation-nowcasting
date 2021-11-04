import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fun1(x, y):
    return np.power((x-y), 2)


def fun2(x, y):
    return (np.power((x-y), 2))/(np.power(x, 2)+np.power(y, 2)+0.1)


def fun3(x, y):
    return 2*(x-y)*(x*y+y**2+1)/np.power(np.power(x, 2)+np.power(y, 2)+1, 2)


def fun4(x, y):
    return 1 - (2*x*y+0.5)/(np.power(x, 2)+np.power(y, 2)+0.5)


def fun5(x, y):
    return -1*x*np.log10(y) - (1-x)*np.log(1-y)


fig1 = plt.figure()
ax = Axes3D(fig1)
X = np.arange(0.1, 1, 0.01)
Y = np.arange(0.1, 1, 0.01)
X, Y = np.meshgrid(X, Y)
Z = fun4(X, Y)
plt.title("This is main title")
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
ax.set_xlabel('x label', color='r')
ax.set_ylabel('y label', color='g')
ax.set_zlabel('z label', color='b')
plt.show()
