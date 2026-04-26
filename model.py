from sklearn import svm

def train_model(X, y):
    model = svm.SVC()
    model.fit(X, y)
    return model