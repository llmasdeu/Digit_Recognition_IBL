#!/opt/local/bin/python2.7
# -*- coding: utf-8 -*-

#
# Mineria de Dades
# Pràctica 2: Aprendre a categoritzar imatges de dígits per IBL
# Curs 2019-2020
#
# Autor: Lluís Masdeu
# Login: lluis.masdeu
#

import numpy as np
import scipy
import sklearn
import sklearn.datasets
import sklearn.model_selection
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.neighbors
import sklearn.metrics as metrics
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.image as mpimg
import pandas as pd
import time

#
# Funció encarregada de carregar i analitzar el dataset de les imatges amb els números.
# Retorna les dades i la seva descripció.
#
def get_and_analyze_dataset():
    # Obtenim el dataset de les imatges amb els números
    # URL del dataset: http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    digits = sklearn.datasets.load_digits()

    # Mostrem els diferents camps de dades que té el dataset carregat
    # print digits.keys()

    # Separem les dades obtingudes en dades i la seva descripció
    X = digits.data
    Y = digits.target

    # Mostrem les dimensions de les dades obtingudes
    # print X.shape, Y.shape

    # Mostrem les dades obtingudes
    # print X, Y

    # Mostrem la descripció de les dades obtingudes
    # print digits.DESCR

    # Obtenim i mostrem les estadístiques bàsiques de les dades carregades
    print 'Classe X: La mitjana aritmètica de les dades és ', np.mean(X)
    print 'Classe Y: La mitjana aritmètica de les dades és ', np.mean(Y)
    print 'Classe X: La desviació típica de les dades és ', np.std(X)
    print 'Classe Y: La desviació típica de les dades és ', np.std(Y)
    print 'Classe X: Tenim ', X.shape[0], ' imatges de 8x8 (representats en tuples de 64)'
    print 'Classe Y: Tenim ', Y.shape[0], ' dades'

    # Reconstruïm la primera imatge del dataset, i la mostrem
    # image = np.reshape(X[0], (8, 8))
    # plt.imshow(image, interpolation='nearest')
    # plt.show()

    # Reconstruïm les primeres 9 imatges del dataset, i les mostrem en una graella
    # f = plt.figure()
    #
    # for x in range (9):
    #     f.add_subplot(3, 3, x + 1)
    #     image = np.reshape(X[x], (8, 8))
    #     plt.imshow(image, interpolation='nearest')
    #
    # plt.show()

    # Reconstruïm les primeres 10 imatges del dataset, i les mostrem en una graella
    f = plt.figure()

    for x in range(10):
        f.add_subplot(2, 5, x + 1)
        image = np.reshape(X[x], (8, 8))
        plt.imshow(image, interpolation='nearest')

    plt.show()

    return X, Y

#
# Funció encarregada de dividir les dades en els conjunts d'entrenament i de test, i de normalitzar-los.
# Retorna els conjunts totals, d'entrenament i de test sense normalitzar, normalitzats, i el percentatge de divisió utilitzat.
#
def separate_and_standardize_data(X, Y):
    # Definim la mida dels conjunts d'entrenament i de test de les dades
    train_size = 0.7
    test_size = 0.3

    # Dividim les dades en els conjunt d'entrenament i de test
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=test_size, train_size=train_size)

    # Mostrem les dimensions dels conjunts d'entrenament i de test de les dades
    # print X_train.shape, Y_train.shape
    # print X_test.shape, Y_test.shape

    # Mostrem les dades dels conjunts d'entrenament i de test de les dades
    # print X_train, Y_train
    # print X_test, Y_test

    # Normalitzem les dades per tal que estiguin centrades a 0 amb desviació típica 1
    # X_std = sklearn.preprocessing.scale(X)
    # X_train_std = sklearn.preprocessing.scale(X_train)
    # X_test_std = sklearn.preprocessing.scale(X_test)
    X_std = sklearn.preprocessing.StandardScaler().fit_transform(X)
    X_train_std = sklearn.preprocessing.StandardScaler().fit_transform(X_train)
    X_test_std = sklearn.preprocessing.StandardScaler().fit_transform(X_test)

    # Mostrem les dimensions de les dades d'entrenament i de test normalitzades
    # print X_train_std.shape, X_test_std.shape

    # Mostrem les dades d'entrenament i de test normalitzades
    # print X_train_std, X_test_std

    return X_std, X_train, Y_train, X_test, Y_test, X_train_std, X_test_std, train_size, test_size

#
# Funció encarregada de gestionar les projeccions en components principals.
#
def data_projection(X, X_std, Y, X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, train_size, test_size):
    # Generem la projecció en PCA de les dades bàsiques
    # pca_projection(X, Y, '[X, Y] PCA - First two PCs using digits data')

    # Generem la projecció en PCA de les dades bàsiques normalitzades
    # pca_projection(X_std, Y, '[X_std, Y] PCA - First two PCs using digits data')

    # Generem la projecció en PCA de les dades d'entrenament
    pca_projection(X_train, Y_train, '[X_train, Y_train] PCA - First two PCs using digits data')

    # Generem la projecció en PCA de les dades d'entrenament normalitzades
    pca_projection(X_train_std, Y_train, '[X_train_std, Y_train] PCA - First two PCs using digits data')

    # Generem la projecció en PCA de les dades de test
    # pca_projection(X_test, Y_test, '[X_test, Y_test] PCA - First two PCs using digits data')

    # Generem la projecció en PCA de les dades de test normalitzades
    # pca_projection(X_test_std, Y_test, '[X_test_std, Y_test] PCA - First two PCs using digits data')

    # Generem la projecció en SVD-Truncated de les dades bàsiques
    # svd_truncated_projection(X, Y, '[X, Y] SVD-T - First two PCs using digits data')

    # Generem la projecció en SVD-Truncated de les dades bàsiques normalitzades
    # svd_truncated_projection(X_std, Y, '[X_std, Y] SVD-T - First two PCs using digits data')

    # Generem la projecció en SVD-Truncated de les dades d'entrenament
    svd_truncated_projection(X_train, Y_train, '[X_train, Y_train] SVD-T - First two PCs using digits data')

    # Generem la projecció en SVD-Truncated de les dades d'entrenament normalitzades
    svd_truncated_projection(X_train_std, Y_train, '[X_train_std, Y_train] SVD-T - First two PCs using digits data')

    # Generem la projecció en SVD-Truncated de les dades de test
    # svd_truncated_projection(X_test, Y_test, '[X_test, Y_test] SVD-T - First two PCs using digits data')

    # Generem la projecció en SVD-Truncated de les dades de test normalitzades
    # svd_truncated_projection(X_test_std, Y_test, '[X_test_std, Y_test] SVD-T - First two PCs using digits data')

    # Generem la projecció en LDA de les dades bàsiques
    # lda_projection(X, Y, '[X, Y] LDA - First two PCs using digits data')

    # Generem la projecció en LDA de les dades bàsiques normalitzades
    # lda_projection(X_std, Y, '[X_std, Y] LDA - First two PCs using digits data')

    # Generem la projecció en LDA de les dades d'entrenament
    lda_projection(X_train, Y_train, '[X_train, Y_train] LDA - First two PCs using digits data')

    # Generem la projecció en LDA de les dades d'entrenament normalitzades
    lda_projection(X_train_std, Y_train, '[X_train_std, Y_train] LDA - First two PCs using digits data')

    # Generem la projecció en LDA de les dades de test
    # lda_projection(X_test, Y_test, '[X_test, Y_test] LDA - First two PCs using digits data')

    # Generem la projecció en LDA de les dades de test normalitzades
    # lda_projection(X_test_std, Y_test, '[X_test_std, Y_test] LDA - First two PCs using digits data')

#
# Funció encarregada de dur a terme la descomposició de les dades en components principals mitjançant la tècnica PCA.
#
def pca_projection(X, Y, title):
    # Descomposem les dades en components principals mitjançant la tècnica PCA (Principal Component Analysis)
    # pca = sklearn.decomposition.PCA(svd_solver='auto')
    # pca.fit(X)
    # pca = sklearn.decomposition.PCA(n_components=2)
    # pca.fit(X, Y)
    # pca_projected = pca.transform(X)
    pca = sklearn.decomposition.PCA(n_components=2)
    pca_projected = pca.fit_transform(X)

    # Mostrem les dimensions de les dades en components principals mitjançant la tècnica PCA
    # print pca_projected.shape

    # Mostrem les dades en components principals mitjançant la tècnica PCA
    # print pca_projected

    # Mostrem la variança i els valors singulars
    print '[Projecció PCA (2 components)] Variança: ', pca.explained_variance_ratio_, ' - Valors singulars: ', pca.singular_values_

    # Dibuixem el diagrama de dispersió amb les dades
    draw_scatter_plot(pca_projected, Y, title)

#
# Funció encarregada de dur a terme la descomposició en components principals mitjançant la tècnica SVD-Truncated.
#
def svd_truncated_projection(X, Y, title):
    # Descomposem les dades en components principals mitjançant la tècnica SVD-Truncated (Single Valu Decomposition-Truncated)
    tsvd = sklearn.decomposition.TruncatedSVD(n_components=2)
    tsvd_projected = tsvd.fit_transform(X)

    # Mostrem les dimensions de les dades en components principals mitjançant la tècnica SVD-Truncated
    # print tsvd_projected.shape

    # Mosytem les dades en components principals mitjançant la tècnica SVD-Truncated
    # print tsvd_projected

    # Mostrem la variança
    print '[Projecció SVD-T (2 components)] Variança: ', tsvd.explained_variance_ratio_

    # Dibuixem el diagrama de dispersió amb les dades
    draw_scatter_plot(tsvd_projected, Y, title)

#
# Funció encarregada de dur a terme la descomposició en components principals mitjançant la tècnica LDA.
#
def lda_projection(X, Y, title):
    # Descomposem les dades en components principals mitjançant la tècnica LDA (Linear Discriminant Analysis)
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
    lda_projected = lda.fit(X, Y).transform(X)

    # Mostrem la variança
    print '[Projecció LDA (2 components)] Variança: ', lda.explained_variance_ratio_

    # Dibuixem el diagrama de dispersió amb les dades
    draw_scatter_plot(lda_projected, Y, title)

#
# Funció encarregada de mostrar un diagrama de dispersió amb les dades indicades.
#
def draw_scatter_plot(projection, color, title):
    # Mostrem els possibles valors del colormap
    # print plt.cm.cmap_d.keys()

    # Generem el diagrama de dispersió
    plt.scatter(projection[:, 0], projection[:, 1], c=color, edgecolor='none', alpha=0.5,cmap=plt.cm.get_cmap('Spectral', 10))

    # Definim els títols del diagrama de dispersió
    plt.title(title, fontsize=12)
    plt.xlabel('PC1', fontsize=10)
    plt.ylabel('PC2', fontsize=10)

    # Preparem la barra de colors
    plt.colorbar()

    # Mostrem el diagrama de dispersió generat
    plt.show()

#
# Funció encarregada de dur a terme els anàlisis a les dades de test.
#
def analyze_test_data(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test):
    # Duem a terme la validació creuada amb les dades sense normalitzar mitjançant la tècnica PCA
    pca_cross_validation(X_train, Y_train, X_test, Y_test, 'PCA - Cross validation')

    # Duem a terme la validació creuada amb les dades normalitzades mitjançant la tècnica PCA
    pca_cross_validation(X_train_std, Y_train, X_test_std, Y_test, 'PCA (Std) - Cross validation')

    # Duem a terme la validació creuada amb les dades sense normalitzar mitjançant la tècnica SVD-Truncated
    svd_truncated_cross_validation(X_train, Y_train, X_test, Y_test, 'SVD-T - Cross validation')

    # Duem a terme la validació creuada amb les dades normalitzades mitjançant la tècnica SVD-Truncated
    svd_truncated_cross_validation(X_train_std, Y_train, X_test_std, Y_test, 'SVD-T (Std) - Cross validation')

#
# Funció encarregada de dur a terme la validació creuada mitjançant la tècnica PCA.
#
def pca_cross_validation(X_train, Y_train, X_test, Y_test, title):
    # Definim el paràmetre de n_neighbors
    parameters = get_n_neighbors_parameter()

    # Definim el classificador K-Nearest Neighbors
    knearest = sklearn.neighbors.KNeighborsClassifier()

    # Definim la estructura que ens servirà per desar els resultats
    gridsearch = sklearn.model_selection.GridSearchCV(knearest, parameters, cv=10, iid=True)

    # Definim els arrays buits que ens ajudaran a desar els resultats
    accuracy = []
    params = []
    means = []

    # Obtenim l'array amb les dimensions
    dimensions = get_dimensions_array()

    # Per cada n dimensions...
    for d in dimensions:
        # Fem la descomposició en d components
        pca = sklearn.decomposition.PCA(n_components=d)

        if d < 64:
            X_fit = pca.fit_transform(X_train)
            X_fit_atest = pca.transform(X_test)
        else:
            X_nl = X_train
            X_nl1 = X_test

        # Calculem i desem els resultats
        gridsearch.fit(X_fit, Y_train)
        result = compute_test(X_fit_atest, Y_test, gridsearch, 10)
        accuracy.append(result)
        means.append(np.mean(result))
        params.append(gridsearch.best_params_['n_neighbors'])

    # Mostrem els valors obtinguts
    # print accuracy, means, params
    print '[Validació creuada PCA] Precisió obtinguda: ', accuracy
    print '[Validació creuada PCA] Mitjanes obtingudes: ', means
    print '[Validació creuada PCA] Veïns més propers: ', params

    # Generem el gràfic amb els resultats
    draw_accuracy(means, dimensions, title)

#
# Funció encarregada de dur a terme la validació creuada mitjançant la tècnica SVD-Truncated.
#
def svd_truncated_cross_validation(X_train, Y_train, X_test, Y_test, title):
    # Definim el paràmetre de n_neighbors
    parameters = get_n_neighbors_parameter()

    # Definim el classificador K-Nearest Neighbors
    knearest = sklearn.neighbors.KNeighborsClassifier()

    # Definim la estructura que ens servirà per desar els resultats
    gridsearch = sklearn.model_selection.GridSearchCV(knearest, parameters, cv=10, iid=True)

    # Definim els arrays buits que ens ajudaran a desar els resultats
    accuracy = []
    params = []
    means = []

    # Obtenim l'array amb les dimensions
    dimensions = get_dimensions_array()

    # Per cada d dimensions...
    for d in dimensions:
        # Fem la descomposició en d components
        svd = sklearn.decomposition.TruncatedSVD(n_components=d)

        if d < 64:
            X_fit = svd.fit_transform(X_train)
            X_fit_atest = svd.transform(X_test)
        else:
            X_nl = X_train
            X_nl1 = X_test

        # Calculem i desem els resultats
        gridsearch.fit(X_fit, Y_train)
        result = compute_test(X_fit_atest, Y_test, gridsearch, 10)
        accuracy.append(result)
        means.append(np.mean(result))
        params.append(gridsearch.best_params_['n_neighbors'])

    # Mostrem els valors obtinguts
    # print accuracy, means, params
    print '[Validació creuada SVD-T] Precisió obtinguda: ', accuracy
    print '[Validació creuada SVD-T] Mitjanes obtingudes: ', means
    print '[Validació creuada SVD-T] Veïns més propers: ', params

    # Generem el gràfic amb els resultats
    draw_accuracy(means, dimensions, title)

#
# Funció encarregada de generar els paràmetres que emprarem per a calcular la validació creuada.
#
def get_n_neighbors_parameter():
    k = np.arange(20) + 1
    parameters = {'n_neighbors': k}

    return parameters

#
# Funció encarregada de generar l'array amb les dimensions que emprarem per dur a terme la validació creuada.
#
def get_dimensions_array():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#
# Funció encarregada de calcular les puntuacions d'encert.
#
def compute_test(X_test, Y_test, clf, cv):
    KFolds = sklearn.model_selection.KFold(n_splits=cv)
    scores = []

    for i, j in KFolds.split(X_test, Y_test):
        test_set = X_test[j]
        test_labels = Y_test[j]
        scores.append(metrics.accuracy_score(test_labels, clf.predict(test_set)))

    return scores

#
# Funció encarregada de mostrar en un diagrama de barres els resultats obtinguts amb la validació creuada.
#
def draw_accuracy(means, dimensions, title):
    # Preparem les dades
    means_def = np.asarray(means)
    pos = np.arange(len(dimensions))

    # Generem el gràfic amb les dades
    plt.bar(pos, np.array(means_def))
    plt.xticks(pos, dimensions)
    plt.title(title)
    plt.xlabel('Dimensions')
    plt.ylabel('Accuracy')
    plt.show()

#
# Funció principal del programa.
#
if __name__ == '__main__':
    # Obtenim i analizem les dades del dataset
    X, Y = get_and_analyze_dataset()

    # Obtenim els conjunts d'entrenament i de test sense normalitzar, normalitzarsm i el percentatge de divisió utilitzats
    X_std, X_train, Y_train, X_test, Y_test, X_train_std, X_test_std, train_size, test_size = separate_and_standardize_data(X, Y)

    # Fem la projecció en diferents components principals
    data_projection(X, X_std, Y, X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, train_size, test_size)

    # Calculem la precisió de les dades
    analyze_test_data(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test)

    exit(0)
