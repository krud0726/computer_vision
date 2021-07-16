#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.datasets import make_circles
from mpl_toolkits import mplot3d


# In[65]:


from sklearn.datasets import make_blobs

# case 1 : random_state = 20
# returns the samples created in x.
# returns the labels of samples created in y.
x_20, y_20 = make_blobs(n_samples=100, n_features=2, centers = 2, cluster_std = 1.2, random_state=20)

# case 2 : random_state = 30
x_30, y_30 = make_blobs(n_samples=100, n_features=2, centers = 2, cluster_std = 1.2, random_state=50)

# case 3 : random_state = 40
x_40, y_40 = make_blobs(n_samples=100, n_features=2, centers = 2, cluster_std = 1.2, random_state=40)


# In[66]:


# displaying case1 data in a scatter plot
plt.scatter(x_20[:, 0], x_20[:, 1], marker='o', c=y_20)

plt.xlabel("Example 1. random_state = 20", labelpad = 10, fontsize = 15)
plt.show()

# displaying case2 data in a scatter plot
plt.scatter(x_30[:, 0], x_30[:, 1], marker='o', c=y_30)

plt.xlabel("Example 2. random_state = 30", labelpad = 10, fontsize = 15)
plt.show()

# displaying case3 data in a scatter plot
plt.scatter(x_40[:, 0], x_40[:, 1], marker='o', c=y_40)

plt.xlabel("Example 3. random_state = 40", labelpad = 10, fontsize = 15)
plt.show()


# In[67]:


# Create a classifier and train the data.
# C 값이 클 수록 오류 허용 안함
# C 값이 작을수록 오류 허용
def draw_decision_boundary(x,y):
    # Declare and configure the model - linear
    clf = svm.SVC(kernel='linear', C=10)
    clf2 = svm.SVC(kernel='linear', C=1)
    clf3 = svm.SVC(kernel='linear', C=0.1)
    
    # train model by data
    clf.fit(x, y)
    clf2.fit(x, y)
    clf3.fit(x, y)
    
    # set plot size
    plt.figure(figsize=(20,5))

    # subplot 1 out of 3 in row 1
    ax1 = plt.subplot(1,3,1)
    ax1.scatter(x[:,0], x[:,1], marker='o',c=y)
    ax1.set_title("C =10.0")
    
    # subplot 2 out of 3 in row 1
    ax2 = plt.subplot(1,3,2)
    ax2.scatter(x[:,0], x[:,1], marker='o',c=y)
    ax2.set_title("C =1.0")
    
    # subplot 3 out of 3 in row 1
    ax3 = plt.subplot(1,3,3)
    ax3.scatter(x[:,0], x[:,1], marker='o',c=y)
    ax3.set_title("C =0.1")
    
    # get range of x and y axis after referencing current axis of plot
    ax = plt.gca()
    x_axis = ax.get_xlim()
    y_axis = ax.get_ylim()

    # stores a range of values on the x and y axes for creating a meshgrid
    xx = np.linspace(x_axis[0], x_axis[1])
    yy = np.linspace(y_axis[0], y_axis[1])

    # Calculate square matrices for xx, yy using the meshgrid function.
    # The square matrix created by the meshgrid function is a square matrix in units of rows and columns, respectively.
    # YY ==> Generate by row
    # XX ==> Generate by column
    YY, XX = np.meshgrid(yy,xx)

    # The XX.ravel() function changes the array to one dimension,
    # and the x-coordinate and y-coordinate arrays are combined into rows through vstack.
    # That is, all Y values according to each X point are displayed as coordinates.
    # Then transpose it using Transpose. ==> Convert to x, y form
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # After finding z, we map it to the coordinates we previously calculated with the meshgrid function.
    # and the isoline is drawn through the contour function. 
    # in this process, the isoline is drawn by interpolation between each cluster.
    # that is, if there are only two (x,y) coordinates, the isoline can be obtained.
  
    # returns the uncertainty value and labeling value of that input by decision_function
    Z = clf.decision_function(xy).reshape(XX.shape)
    Z2 = clf2.decision_function(xy).reshape(XX.shape)
    Z3 = clf3.decision_function(xy).reshape(XX.shape)

    # case 1 (c = 10): draw decision boundary and margins lines
    ax1.contour(XX, YY, Z ,colors='k', levels = [-1,0,1], alpha= 0.5,
              linestyles=['--','-','--'])
    
    # print the supporter vector in different colors.
    ax1.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100,
              linewidth=1)

    # case 2 (c = 1): draw decision boundary and margins lines 
    ax2.contour(XX, YY, Z2 ,colors='k', levels = [-1,0,1], alpha= 0.5,
              linestyles=['--','-','--'])
    
    # print the supporter vector in different colors.
    ax2.scatter(clf2.support_vectors_[:,0], clf2.support_vectors_[:,1], s=100,
              linewidth=1)

    # case 3 (c = 0.1): draw decision boundary and margins lines 
    ax3.contour(XX, YY, Z3 ,colors='k', levels = [-1,0,1], alpha= 0.5,
              linestyles=['--','-','--'])

    # print the supporter vector in different colors.
    ax3.scatter(clf3.support_vectors_[:,0], clf3.support_vectors_[:,1], s=100,
              linewidth=1)

    plt.show()


# In[68]:


# print outputs the result through the function.
draw_decision_boundary(x_20,y_20)


# In[69]:


# print outputs the result through the function.
draw_decision_boundary(x_30,y_30)


# In[70]:


# print outputs the result through the function.
draw_decision_boundary(x_40,y_40)


# In[71]:


# Non linear
# Create a classifier and train the data.
# C 값이 클 수록 오류 허용 안함
# C 값이 작을수록 오류 허용
def draw_decision_boundary_nonlinear(x,y):
    # Declare and configure the model - non linear
    clf = svm.SVC(kernel='rbf',  C=10)
    clf2 = svm.SVC(kernel='rbf',C=1)
    clf3 = svm.SVC(kernel='rbf', C=0.1)
    
    # train model by data
    clf.fit(x, y)
    clf2.fit(x, y)
    clf3.fit(x, y)

    # set plot size
    plt.figure(figsize=(20,5))

    # subplot 1 out of 3 in row 1
    ax1 = plt.subplot(1,3,1)
    ax1.scatter(x[:,0], x[:,1], marker='o',c=y, cmap='autumn')
    ax1.set_title("C =10.0")
    
    # subplot 2 out of 3 in row 1
    ax2 = plt.subplot(1,3,2)
    ax2.scatter(x[:,0], x[:,1], marker='o',c=y, cmap='autumn')
    ax2.set_title("C =1.0")
    
    # subplot 3 out of 3 in row 1
    ax3 = plt.subplot(1,3,3)
    ax3.scatter(x[:,0], x[:,1], marker='o',c=y, cmap='autumn')
    ax3.set_title("C =0.1")

    # get range of x and y axis after referencing current axis of plot
    ax = plt.gca()
    x_axis = ax.get_xlim()
    y_axis = ax.get_ylim()

    # stores a range of values on the x and y axes for creating a meshgrid
    xx = np.linspace(x_axis[0], x_axis[1])
    yy = np.linspace(y_axis[0], y_axis[1])

    # Calculate square matrices for xx, yy using the meshgrid function.
    # The square matrix created by the meshgrid function is a square matrix in units of rows and columns, respectively.
    # YY ==> Generate by row
    # XX ==> Generate by column
    YY, XX = np.meshgrid(yy,xx)

    # The XX.ravel() function changes the array to one dimension,
    # and the x-coordinate and y-coordinate arrays are combined into rows through vstack.
    # That is, all Y values according to each X point are displayed as coordinates.
    # Then transpose it using Transpose. ==> Convert to x, y form
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # After finding z, we map it to the coordinates we previously calculated with the meshgrid function.
    # and the isoline is drawn through the contour function. 
    # in this process, the isoline is drawn by interpolation between each cluster.
    # that is, if there are only two (x,y) coordinates, the isoline can be obtained.
    
    # returns the uncertainty value and labeling value of that input by decision_function
    Z = clf.decision_function(xy).reshape(XX.shape)
    Z2 = clf2.decision_function(xy).reshape(XX.shape)
    Z3 = clf3.decision_function(xy).reshape(XX.shape)

    # case 1 (c = 10): draw decision boundary and margins lines
    ax1.contour(XX, YY, Z ,colors='k', levels = [-1,0,1], alpha= 0.5,
              linestyles=['--','-','--'])
    
    # print the supporter vector in different colors.
    ax1.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100,
              linewidth=1)

    # case 2 (c = 1): draw decision boundary and margins lines 
    ax2.contour(XX, YY, Z2 ,colors='k', levels = [-1,0,1], alpha= 0.5,
              linestyles=['--','-','--'])
    
    # print the supporter vector in different colors.
    ax2.scatter(clf2.support_vectors_[:,0], clf2.support_vectors_[:,1], s=100,
              linewidth=1)

    # case 3 (c = 0.1): draw decision boundary and margins lines 
    ax3.contour(XX, YY, Z3 ,colors='k', levels = [-1,0,1], alpha= 0.5,
              linestyles=['--','-','--'])
    
    # print the supporter vector in different colors.
    ax3.scatter(clf3.support_vectors_[:,0], clf3.support_vectors_[:,1], s=100,
              linewidth=1)

    plt.show()


# In[72]:


# factor ==> Scale factor between inner and outer circle in the range (0, 1)
# noise ==> Standard deviation of Gaussian noise added to the data.
X, y = make_circles(n_samples=100, factor=0.2, noise=0.1)

# displaying data in a scatter plot
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=50, cmap = 'autumn')
plt.xlabel("Example 4. factor = 0.1, noise = 0.1", labelpad = 10, fontsize = 15)
plt.show()


# In[73]:


# To use a Gaussian RBF kernel based on euclidean distance, 
# square and add the distance from the zero coordinate of each axis.
r = np.exp(-(X** 2).sum(1))

# Creating an Axes3D object using the projection='3d' keyword
ax = plt.subplot(projection='3d')

# create a scatter plot in 3D space.
ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')

# set the axis name.
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r')
ax.set_title('< Gaussian RBF kernel >')

# pring result 
plt.show()


# In[74]:


# print outputs the result through the function.
draw_decision_boundary_nonlinear(X,y)


# In[ ]:




