import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.model import create_model

def compute_linear_CKA(args, local_params, global_params, input_data):
    global_model = create_model(args)
    global_model.load_state_dict(global_params)
    baseline_output = global_model(input_data)
    # print(f"Global output shape: {baseline_output.shape}")

    similarities = []
    for param in local_params:
        local_model = create_model(args)
        local_model.load_state_dict(param)
        local_output = local_model(input_data)
        # print(f"Local output shape: {local_output.shape}")

        similarity = linear_CKA(baseline_output, local_output)
        # print(f'similarity: {similarity}')

        similarities.append(similarity)

    return np.array(similarities).reshape(-1, 1)

def centering(K):
    # print(f'K: ｛K｝, K.shape:｛K.shape｝')
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH

def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX

def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

def linear_HSIC(X, Y):
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    # print(f'X: ｛X｝, X.shape:｛X.shape｝')
    # print(f'Y: ｛Y｝, Y.shape:｛Y.shape｝')
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    # print(f'L_X: ｛L_X｝, L_X.shape:｛L_X.shape｝')
    # print(f'L_Y: ｛L_Y｝, L_Y.shape:｛L_Y.shape｝')
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    # print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)