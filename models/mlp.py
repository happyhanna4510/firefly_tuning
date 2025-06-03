from sklearn.neural_network import MLPRegressor

def create_model(n_features):
    model = MLPRegressor(hidden_layer_sizes=(5,), activation='relu', solver='lbfgs', max_iter=1)
    model.fit([[0]*n_features], [0])
    return model

def apply_weights(model, weights):
    idx = 0
    for i in range(len(model.coefs_)):
        shape = model.coefs_[i].shape
        size = shape[0] * shape[1]
        model.coefs_[i] = weights[idx:idx+size].reshape(shape)
        idx += size
    for i in range(len(model.intercepts_)):
        size = model.intercepts_[i].shape[0]
        model.intercepts_[i] = weights[idx:idx+size]
        idx += size
    return model
