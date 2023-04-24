def plot_svc_old(svc, X, y):
    """
    plots the results from the sklearn SVM fit

    Parameters
    ----------
    svc: instance of sklearn.svm._classes.SVC
    X: ndarray of size (M, 2) : data, with M the number of samples and 2 features
    y: ndarray of size (M,) :  labels of the datapoints (either 1 or other)
    
    Returns
    -------
    None
    """
        
    # build mesh of gridpoints
    x1_min, x1_max = X[:,0].min(), X[:,0].max()
    x2_min, x2_max = X[:,1].min(), X[:,1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    # compute hypothesis on the meshgrid 
    X_mesh = np.column_stack((xx1.ravel(), xx2.ravel()))
    h = svc.predict(X_mesh).reshape(xx1.shape)
    
    # plot the contour level(s)
    plt.contourf(xx1, xx2, h, colors=['g','r'],alpha=0.1)
    
    # plot the data and decisionboundary
    plotdata(X, y)

    # plot the supportvectors
    plt.scatter(svc.support_vectors_[:,0], svc.support_vectors_[:,1], c='k', marker='|', s=100, linewidths=1)
    plt.title(f'C={svc.C}; Number of support vectors: {svc.support_.size}')    
   
    plt.xlabel('x1')
    plt.ylabel('x2')