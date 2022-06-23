# -*- coding:utf-8 -*-
# https://blog.csdn.net/xc_zhou/article/details/81535033
import numpy as np
from scipy.spatial.distance import pdist


def minkowski(x,y):
    X=np.vstack([x,y])
    p1=1;p2=2;p3=1000;
    d1=pdist(X,'minkowski',p=p1)[0]
    d2=pdist(X,'minkowski',p=p2)[0]
    d3=pdist(X,'minkowski',p=p3)[0]
    # p=1 Manhattan Distance
    # p=2 Euclidean Distance
    # p->∞ Chebyshev Distance
    print('p='+str(p1)+', Minkowski(Manhattan) d1:',d1)
    print('p='+str(p2)+', Minkowski(Euclidean) d2:',d2)
    print('p='+str(p3)+', Minkowski(close to Chebyshev) d3:',d3)
    return d1, d2, d3

def euclidean(x,y):
    X=np.vstack([x,y])
    d2=pdist(X,'seuclidean')[0]
    print('Standardized Euclidean Distance:',d2)
    return d2

def seuclidean(x,y):
    X=np.vstack([x,y])
    d2=pdist(X)[0]
    print('Euclidean Distance:',d2)
    return d2

def manhattan(x,y):
    X=np.vstack([x,y])
    d2=pdist(X,'cityblock')[0]
    print('Manhattan Distance:',d2)
    return d2

def chebyshev(x,y):
    X=np.vstack([x,y])
    d2=pdist(X,'chebyshev')[0]
    print('Chebyshev Distance:',d2)
    return d2

def cosine(x,y):
    # same: 0, diff: x.xx
    X=np.vstack([x,y])
    d2=pdist(X,'cosine')[0]
    # print('Cosine Distance:',d2)
    return d2

def hamming(x,y):
    X=np.vstack([x,y])
    d2=pdist(X,'hamming')[0]
    print('Hamming Distance:',d2)
    return d2

def jaccard(x,y):
    X=np.vstack([x,y])
    d2=pdist(X,'jaccard')[0]
    print('Jaccard Distance:',d2)
    return d2



def mahalanobis(x,y):
    X = np.vstack([x, y])
    # sample number should be greater than dimension
    XT = X.T
    d2 = pdist(XT, 'mahalanobis')
    print('Mahalanobis Distance:',d2)
    return d2

def pearson(x,y):
    X=np.vstack([x,y])
    d2=np.corrcoef(X)[0][1]
    print('Pearson Distance:',d2)
    return d2

def braycurtis(x,y):
    X=np.vstack([x,y])
    d2=pdist(X,'braycurtis')[0]
    print('Bray Curtis Distance:',d2)
    return d2

if __name__ == "__main__":
    ll = [1,2,3]
    ll = np.array(ll)
    x = y = ll
    print(type(ll))
    # x=np.random.random(10)
    # y=np.random.random(10)
    print("x:\n",x,'\ny:\n',y)
    minkowski(x,y)
    # seuclidean(x,y)
    cosine(x,y)
    hamming(x,y)
    jaccard(x,y)
    pearson(x,y)
    braycurtis(x,y)

