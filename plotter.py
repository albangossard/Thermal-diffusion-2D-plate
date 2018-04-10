from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plotter(x, title='', Case=None):
    N=x.shape[0]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    X = np.linspace(0.,1.,x.shape[1])
    Y = np.linspace(0.,1.,x.shape[0])
    X, Y = np.meshgrid(X, Y)
    Z = X*0+x
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(0.99*np.min(x), 1.01*np.max(x))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title(title)
    plt.show()

    if Case is not None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Make data.
        Z = Case.sol
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(0.99*np.min(Z), 1.01*np.max(Z))
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.title("Theoretical solution")
        plt.show()


        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Make data.
        Z = Case.sol-x
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(0.99*np.min(Z), 1.01*np.max(Z))
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.title("Error")
        plt.show()


def section(x,list_xSection,axis):
    N=int(np.sqrt(x.shape[0]))
    dx=1./N
    xAxis=np.linspace(0.,1.,N)
    if axis=='x':
        for xSection in list_xSection:
            data=[]
            idX=int(xSection/dx)
            for j in range(N):
                data.append(x[idX+N*j])
            plt.plot(xAxis,data,label='x='+str(xSection))
        plt.xlabel('y')
        plt.ylabel('Z')
    else:
        for xSection in list_xSection:
            data=[]
            idY=int(xSection/dx)
            for i in range(N):
                data.append(x[i+N*idY])
            plt.plot(xAxis,data,label='y='+str(xSection))
        plt.xlabel('x')
        plt.ylabel('Z')
    plt.legend()
    plt.show()