from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from skimage import io
import numpy as np

#from statistics import mode
from tqdm import tqdm
from glob import glob
import cv2
import json
import random
import argparse
from natsort import natsorted

def get_mlp(n_ft, bs=16):
    mlp = MLPClassifier(
        hidden_layer_sizes=(n_ft),
        activation='relu',

        solver='adam',
        #solver='sgd',
        #learning_rate='adaptive',
        #momentum=0.9,

        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.2,
        max_iter=500,
        shuffle=True,

        batch_size=bs,
        warm_start=True,
        verbose=True
    )

    return mlp

def get_svm(svm_cfg):
    svm = SVC(**svm_cfg)
    return svm

def extract_lbp(im, r, method):
    lbp_image = local_binary_pattern(im, P=8*r, R=r, method=method)
    n_bins = 256
    hist, _ = np.histogram(lbp_image, density=True, bins=n_bins, range=(0, n_bins))
    return hist

def gen_train_test(data_root, random_state, test_size, radius, method):
    X = []
    y = []
    fs = natsorted(list(glob(f"{data_root}/*/*")))

    for f in tqdm(fs):
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        c = f.split("/")[-2]
        hist = extract_lbp(im, radius, method)
        
        X.append(hist)
        y.append(int(c))
        
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = test_size,
        shuffle=True,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def main(
    radius,       # LBP radius
    method,       # LBP method
    random_state, # 32-bit integer for reproducibility
    data_root,    # Directory containing all images (LPRD syntax)
    test_size,    # float between 0 and 1 indicating test size
    classifier,   # SVM or MLP
    n_features,   # MLP arg
    kernel        # SVM kernel
):
    
    args = {
        "lbp_radius": radius,
        "lbp_method": method,
        "svm_method": "SVC",
        "svm_kernel": kernel,
        "random_state": random_state,
        "test_size": test_size
    }
    
    X_train, X_test, y_train, y_test = gen_train_test(
        data_root,
        random_state,
        test_size,
        radius,
        method
    )

    if classifier == 'svm':
        model = get_svm({'kernel':kernel})
    elif classifier == 'mlp':
        model = get_mlp(n_features)
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"LBP + {classifier.upper()} with parameters:\n", json.dumps(args, indent=2))
    print(f"Test Accuracy: ", accuracy_score(y_test, predictions))
    print(f"Test F1-score: ", f1_score(y_test, predictions, average='micro'))

    predictions = model.predict(X_train)

    print(f"Train Accuracy: ", accuracy_score(y_train, predictions))
    print(f"Train F1-score: ", f1_score(y_train, predictions, average='micro'))


if __name__ == "__main__":
    """
    LBP methods: default, ror, uniform, nri_uniform, var
    SVM kernels: linear, poly, rbf, sigmoid
    MLP n_features: 16, 24, 32, 48, 64
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--radius', default=3, type=int)
    parser.add_argument('-m', '--method', default='uniform', type=str)
    parser.add_argument('-c', '--classifier', default='mlp', type=str)

    # Model specific arguments
    parser.add_argument('-k', '--kernel', default='rbf', type=str)
    parser.add_argument('-nf', '--n_features', default=64, type=int)

    # Data arguments
    parser.add_argument('-s', '--random_state', default=123, type=int)
    parser.add_argument('-t', '--test_size', default=0.2, type=float)
    parser.add_argument('-d', '--data_root', default="LPRD_Dataset/lps/", type=str)

    args = vars(parser.parse_args())

    if args['random_state'] == 0:
        args['random_state'] = random.randint(1, (2**32) - 1)

    main(**args)
