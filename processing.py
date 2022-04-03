from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold

import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import pickle as pk

from os.path import exists


# useful methods
# gets the mnist data
def get_mnist_dataset(format = 'center'):
    # performs transformation
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform = transforms.Compose([transforms.ToTensor()])
    
    # downloads data
    train_dataset = datasets.MNIST('./MNIST/', train=True, transform=transform, download=True)
    test_dataset  = datasets.MNIST('./MNIST/', train=False, transform=transform, download=True)
    
    # loads data
    print('\tloading...')
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=len(test_dataset))
    
    # into numpy array
    print('\tconverting...')
    for _, (input, target) in enumerate(train_loader):
        train_dataset_array_x = input
        train_dataset_array_y = target
    train_dataset_array_x = train_dataset_array_x.numpy()
    train_dataset_array_y = np.array(train_dataset_array_y)
    
    for _, (input, target) in enumerate(test_loader):
        test_dataset_array_x = input
        test_dataset_array_y = target
    test_dataset_array_x = test_dataset_array_x.numpy()
    test_dataset_array_y = np.array(test_dataset_array_y)
    
    if format == 'center':
        #centers
        print('\tcentering...')
        train_dataset_array_x = np.array([np.append(train_dataset_array_x[i].sum(axis=1), train_dataset_array_x[i].sum(axis=2)) for i in range(len(train_dataset_array_x))])
        test_dataset_array_x  = np.array([np.append(test_dataset_array_x[i].sum(axis=1), test_dataset_array_x[i].sum(axis=2)) for i in range(len(test_dataset_array_x))])
    else:
        # flattens data
        print('\tflattening...')
        train_dataset_array_x = np.array([train_dataset_array_x[i].flatten() for i in range(len(train_dataset_array_x))])
        test_dataset_array_x = np.array([test_dataset_array_x[i].flatten() for i in range(len(test_dataset_array_x))])
    
    return train_dataset_array_x, test_dataset_array_x, train_dataset_array_y, test_dataset_array_y





# trains 3 models w/ hparam opt
def train_models(x_train, x_test, y_train, y_test):
    # storing model results
    d = {'svm':{}, 'knn':{}, 'xgboost':{}}
    # radial svm
    params = {'C': [1, 1.5]}
    model  = SVC()
    grid   = GridSearchCV(model, params, n_jobs=12, cv=StratifiedKFold(shuffle=True), verbose=2, refit=True)
    grid.fit(x_train, y_train)
    y_pred_test  = grid.predict(x_test)
    d['svm']['true'] = y_test
    d['svm']['pred'] = y_pred_test
    d['svm']['model'] = grid
    
    # knn
    params = {'n_neighbors': [3, 6]}
    model  = KNeighborsClassifier()
    grid   = GridSearchCV(model, params, n_jobs=12, cv=StratifiedKFold(shuffle=True), verbose=2, refit=True)
    grid.fit(x_train, y_train)
    y_pred_test  = grid.predict(x_test)
    d['knn']['true'] = y_test
    d['knn']['pred'] = y_pred_test
    d['knn']['model'] = grid
    
    # xgboost
    xgb_params = {'max_depth': [5, 7], 'min_child_weight': [6,9]}
    xgb_model = xgb.XGBClassifier(objective = 'multi:softmax', num_class = 10, seed=1337)
    grid_xgb = GridSearchCV(xgb_model, xgb_params, n_jobs=12, cv=StratifiedKFold(shuffle=True), verbose=2, refit=True)
    grid_xgb.fit(x_train, y_train)
    y_pred_test  = grid_xgb.predict(x_test)
    d['xgboost']['true'] = y_test
    d['xgboost']['pred'] = y_pred_test
    d['xgboost']['model'] = grid_xgb
    
    return d





# creates contact matrix for missclassification
def generate_contact_matrix(d):
    d_ = d['true'].groupby([d['pred'], d['true']]).size().unstack().reindex(columns=(range(0, d['true'].max() + 1))).fillna(0).astype(int)
    return d_




# generates the data for tsne projections
def generate_tsne_projections(d, seed=3):
    # tsne
    tsne = TSNE(n_components=3, random_state=seed)
    # creates projections
    tsne_projections = tsne.fit_transform(x)
    return tsne_projections






# runit
if __name__ == '__main__':
    
    # loading the MNIST data
    print('processing MNIST data...')
    data_mnist = get_mnist_dataset(format='center')
    x_train, x_test, y_train, y_test = data_mnist
    
    # generating tsne projection data
    print('generating tsne projections...')
    x, y = x_train[:1000], y_train[:1000]
    x, y = x[np.argsort(y)], np.sort(y)
    tsne_projections = generate_tsne_projections(x, seed=6)
    tsne_projections_labels = (tsne_projections, y)
    with open('data/tsne_projections_labels.pk', 'wb') as _:
        pk.dump(tsne_projections_labels, file = _)
        
    # model training (checks if there is already a model)
    print('training models...')
    if exists('data/model_results.pk'):
    # if True:
        with open('data/model_results.pk', 'rb') as _:
            model_results = pk.load(_)
    else:
        model_results = train_models(x_train, x_test, y_train, y_test)
        with open('data/model_results.pk', 'wb') as _:
            pk.dump(model_results, file = _)
    
    # generating the input for heatmaps
    print('generating performance...')
    model_performance = {}
    for i in ['svm', 'knn', 'xgboost']:
        d = pd.DataFrame({'true': model_results[i]['true'],
                          'pred': model_results[i]['pred']})
        model_performance[i] = np.array(generate_contact_matrix(d))
    with open('data/model_performance.pk', 'wb') as _:
        pk.dump(model_performance, file = _)






#
