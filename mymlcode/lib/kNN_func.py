import numpy as np

def kNN_func(k, X_train, y_train, x, labels = None):

    assert 1 <=k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"
    
    # get vector distances, below both ok
    #distances = np.sqrt(np.sum((X_train-x)**2))
    distances = np.linalg.norm(X_train-x, axis=1)

    # arg sort distances
    nearest = np.argsort(distances)

    # get top k label
    topK_y = [y_train[i] for i in nearest[:k]]

    # cal result weights
    weights = np.bincount(np.array(topK_y))

    print_weights(weights, labels)

    # get expect result
    return np.argmax(weights)

# print result rate
def print_weights(weights, labels = None):

    total = sum(weights)

    if labels is not None:
        assert len(weights) == len(labels), "the size of weights must equal to the size of labels"

        for i, val in enumerate(weights):
            print("[%s]概率:[%s]" % (labels[i], val/total))

    else:
        for i, val in enumerate(weights):
            print("[%s]概率:[%s]" % (i, val/total))
        


