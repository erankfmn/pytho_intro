import matplotlib.pyplot as plt
import numpy as np
from cvxopt import *
from cvxpy import *
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

from MultiSVM import *
from mesh import *





N=50
s = np.random.normal(0, 0.5,(N,2))

X1=s+[0,0]
X2=s+[0,6]
X3=s+[6,6]
X4=s+[6,0]


y=np.concatenate([0*np.ones(N),1*np.ones(N),2*np.ones(N),3*np.ones(N)])
y=y.astype(int)
X=np.concatenate([X1,X2,X3,X4])
weight_sample = np.concatenate([1*np.ones(2*N),10*np.ones(2*N)])
theta=np.array([1,1,10,10],dtype = np.int)
# y=np.concatenate([10*np.ones(N),100*np.ones(N),200*np.ones(N),300*np.ones(N),50*np.ones(N)])
# X=np.concatenate([X1,X2,X3,X4,X5])

cls=MultiSVM(theta)
cls.fit(X,y)

# fit svm classifier
#alphas = fit5(X, y)
#alphas2 = fit2(X, y)
#alphas4 = fit(X, y)



# get weights
# w = np.sum(alphas4 * X, axis = 0)
#
# w2= alphas  * X
# # get bias
# cond = ((alphas4 > 1e-4)).reshape(-1)
# b = y[cond] - np.dot(X[cond], w)
# bias = b[0]

# print("quadric",w,bias)
# print("mine:",w2)
# print("my result",cls.w2.value,cls.b2.value)
# print("alphas mine",alphas[(alphas > 1e-4 )])
# print("negative alphas mine",alphas[(-alphas > 1e-4 )])
# print("alphas mine",alphas)
# print("alphas2",alphas2[alphas2 > 1e-4])
# print("alpha quad ",alphas4[alphas4 > 1e-4])

# a = -w[0] / w[1]
# xx = np.linspace(-2,10)
# yy = a * xx - (bias) / w[1]
# yy_down = a * xx - (bias+1) / w[1]
# yy_up = a * xx - (bias-1) / w[1]

#
# plt.figure()
# plt.clf()
# plt.plot(X1[:,0],X1[:,1],'*r',X2[:,0],X2[:,1],'^b',X3[:,0],X3[:,1],'+y',X4[:,0],X4[:,1],'og',X5[:,0],X5[:,1],'''''')
#
# plt.plot(xx, yy, 'k-')
# plt.plot(xx, yy_down, 'b--')
# plt.plot(xx, yy_up, 'r--')

#print("sum",np.sum(alphas,axis=1))#alphas*np.ones([ X.shape[0],1]))
plt.figure()
plt.clf()
plt.plot(X1[:,0],X1[:,1],'*r',X2[:,0],X2[:,1],'^b',X3[:,0],X3[:,1],'+y',X4[:,0],X4[:,1],'og')

x = np.arange(-2, 10)
uni=cls.y
y1 = -(cls.w2.value[0, 0] * x + cls.b2.value[0] -uni[0] ) / cls.w2.value[0, 1]
plt.plot(x, y1.T, color='red')

y2 = -(cls.w2.value[1, 0] * x + cls.b2.value[1] -uni[1]) / cls.w2.value[1, 1]
plt.plot(x, y2.T, color='blue')

y3 = -(cls.w2.value[2, 0] * x + cls.b2.value[2]-uni[2]) / cls.w2.value[2, 1]
plt.plot(x, y3.T, color='yellow')

y4 = -(cls.w2.value[3, 0] * x + cls.b2.value[3]-uni[3]) / cls.w2.value[3, 1]
plt.plot(x, y4.T, color='green')

# y5 = -(cls.w2.value[4, 0] * x + cls.b2.value[4]-uni[4]) / cls.w2.value[4, 1]
# plt.plot(x, y5.T)


clf = svm.SVC(kernel='linear',decision_function_shape='ovo',class_weight={2:10,3:10})
#clf2 = svm.SVC(kernel='linear',decision_function_shape='ovr')
clf2 = svm.LinearSVC(multi_class='ovr',class_weight={2:10,3:10})
clf3 = svm.LinearSVC(multi_class='crammer_singer',class_weight={2:10,3:10})




# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
#C = 1.0  # SVM regularization parameter
# models = (svm.SVC(kernel='linear', C=C),
#           svm.LinearSVC(C=C),
#           svm.SVC(kernel='rbf', gamma=0.7, C=C),
#           svm.SVC(kernel='poly', degree=3, C=C))

model = (clf,clf2,clf3,cls)


models = (clf.fit(X, y) for clf in model)

# title for the plots
titles = ('SVC 1vs 1',
          'SVC 1vs rest',
           'SVC 1vs crammer',
          'prioritized SVC')

# Set-up 2x2 grid for plotting.
#plt.figure()
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

xx, yy = make_meshgrid(X[:, 0], X[:, 1])

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # ax.set_xlabel('x label')
    # ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()



