import pandas as pd
import numpy as np


def data_preprocessing():
    """
    Imports and processes csv files into X and y data, label encodes gait classes, and performs train/test splits
    :return: train and test data splits for 2-class and 3-class data in respective lists dataset_1 and dataset_2
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn import model_selection
    
    from sklearn import tree

    # Import csv data and split into independent (X) and dependent (y) for normal / abnormal classification
    data_2c = pd.read_csv('dataset_1.csv')
    X_2c = data_2c.iloc[:, 0:6].values
    y_2c = data_2c.iloc[:, -1].values
    labelencoder_2c = LabelEncoder()
    y_2c = labelencoder_2c.fit_transform(y_2c)

    # Import csv data and split into independent (X) and dependent (y) for normal / DH / Sp classification
    data_3c = pd.read_csv('dataset_2.csv')
    X_3c = data_3c.iloc[:, 0:6].values
    y_3c = data_3c.iloc[:, -1].values
    labelencoder_3c = LabelEncoder()
    y_3c = labelencoder_3c.fit_transform(y_3c)
    print()

    # Split data into training and testing sets
    X_2c_train, X_2c_test, y_2c_train, y_2c_test = model_selection.train_test_split(X_2c, y_2c, test_size=0.2,
                                                                                    random_state=42)
    train_test_2c = [X_2c_train, X_2c_test, y_2c_train, y_2c_test]

    X_3c_train, X_3c_test, y_3c_train, y_3c_test = model_selection.train_test_split(X_3c, y_3c, test_size=0.2,
                                                                                    random_state=42)
    train_test_3c = [X_3c_train, X_3c_test, y_3c_train, y_3c_test]

    return train_test_2c, train_test_3c


def model_train_evaluation(train_test_2c, train_test_3c, classifier_method):
    """
    Creates classifier object, fits classifier to training data, predicts test data, calculates confusion matrix and
    normalized accuracy.
    :param train_test_2c: list - [X_2c_train, X_2c_test, y_2c_train, y_2c_test]
    :param train_test_3c:  list - [X_3c_train, X_3c_test, y_3c_train, y_3c_test]
    :param classifier_method: string to select classifier from 'knn', 'svm', 'nb', and 'rf'
    :return: a list of confusion matrices and normalized accuracies for 2c and 3c data
    """
    if classifier_method == 'knn':
        # Selects and fits the k-nearest neighbors classifier
        from sklearn.neighbors import KNeighborsClassifier
        classifer_2c = KNeighborsClassifier(n_neighbors=5)
        classifer_2c.fit(train_test_2c[0], train_test_2c[2])

        classifer_3c = KNeighborsClassifier(n_neighbors=5)
        classifer_3c.fit(train_test_3c[0], train_test_3c[2])

    elif classifier_method == 'svm':
        # Selects and fits the SVM neighbors classifier (must be feature scaled)
        from sklearn.preprocessing import StandardScaler
        sc_2c = StandardScaler()
        train_test_2c[0] = sc_2c.fit_transform(train_test_2c[0])  # 2c train
        train_test_2c[1] = sc_2c.transform(train_test_2c[1])  # 2c test

        sc_3c = StandardScaler()
        train_test_3c[0] = sc_3c.fit_transform(train_test_3c[0])  # 3c train
        train_test_3c[1] = sc_3c.transform(train_test_3c[1])  # 3c test

        from sklearn.svm import SVC
        classifer_2c = SVC(random_state=42)
        classifer_2c.fit(train_test_2c[0], train_test_2c[2])

        classifer_3c = SVC(random_state=42)
        classifer_3c.fit(train_test_3c[0], train_test_3c[2])

    elif classifier_method == 'nb':
        # Selects and fits the naive Bayes neighbors classifier
        from sklearn.naive_bayes import GaussianNB
        classifer_2c = GaussianNB()
        classifer_2c.fit(train_test_2c[0], train_test_2c[2])

        classifer_3c = GaussianNB()
        classifer_3c.fit(train_test_3c[0], train_test_3c[2])


    elif classifier_method == 'dt':
        # Selects and fits the decesion tree classifier
        from sklearn. tree import DecisionTreeClassifier
        classifer_2c = DecisionTreeClassifier(criterion='entropy', random_state=42)
        classifer_2c.fit(train_test_2c[0], train_test_2c[2])

        classifer_3c = DecisionTreeClassifier(criterion='entropy', random_state=42)
        classifer_3c.fit(train_test_3c[0], train_test_3c[2])    

    elif classifier_method == 'rf':
        # Selects and fits the random forest classifier
        from sklearn.ensemble import RandomForestClassifier
        classifer_2c = RandomForestClassifier(criterion='entropy', random_state=42)
        classifer_2c.fit(train_test_2c[0], train_test_2c[2])

        classifer_3c = RandomForestClassifier(criterion='entropy', random_state=42)
        classifer_3c.fit(train_test_3c[0], train_test_3c[2])

    # Predict test data
    y_pred_2c = classifer_2c.predict(train_test_2c[1])
    y_pred_3c = classifer_3c.predict(train_test_3c[1])

    # Calculate confusion matrix and normalized accuracy
    from sklearn import metrics
    cm_2c = metrics.confusion_matrix(train_test_2c[3], y_pred_2c)
    acc_2c = metrics.accuracy_score(train_test_2c[3], y_pred_2c, normalize=True)

    cm_3c = metrics.confusion_matrix(train_test_3c[3], y_pred_3c)
    acc_3c = metrics.accuracy_score(train_test_3c[3], y_pred_3c, normalize=True)

    # Perform k-fold cross validation for improved accuracy score
    from sklearn.model_selection import cross_val_score
    accuracies_2c = cross_val_score(classifer_2c, train_test_2c[0], train_test_2c[2], cv=10, n_jobs=-1)
    kfold_acc_2c = accuracies_2c.mean()
    kfold_std_2c = accuracies_2c.std()

    accuracies_3c = cross_val_score(classifer_3c, train_test_3c[0], train_test_3c[2], cv=10, n_jobs=-1)
    kfold_acc_3c = accuracies_3c.mean()
    kfold_std_3c = accuracies_3c.std()

    evaluation_results = [cm_2c, acc_2c, cm_3c, acc_3c, kfold_acc_2c, kfold_std_2c, kfold_acc_3c, kfold_std_3c]

    return evaluation_results


def results_comparison(knn_results, svm_results, nb_results, dt_results, rf_results, cm_output):
    """
    Prints a string comparing the accuracy of each classifier in percentage
    :param knn_results: results list from model_train_evaluation for KNN classifier
    :param svm_results: results list from model_train_evaluation for SVM classifier
    :param nb_results: results list from model_train_evaluation for naive Bayes classifier
    :param rf_results: results list from model_train_evaluation for random forest classifier
    :param cm_output: boolean to print confusion matrices
    :return: console output
    """

    print(f'The accuracy rates for normal / abnormal are: KNN: {knn_results[1]*100:.2f}%, '
          f'SVM: {svm_results[1]*100:.2f}%, Naive Bayes: {nb_results[1]*100:.2f}%, '
          f'Desision Tree: {rf_results[1]*100:.2f}%, '
          f'Random Forest: {rf_results[1]*100:.2f}%')
    print()
    print(f'The accuracy rates for normal / disk hernia / spondylolisthesis are: KNN: {knn_results[3]*100:.2f}%, '
          f'SVM: {svm_results[3]*100:.2f}%, Naive Bayes: {nb_results[3]*100:.2f}%, '
          f'Desision Tree: {rf_results[3]*100:.2f}%, '
          f'Random Forest: {rf_results[3]*100:.2f}%')

    # Prints confusion matrices for classifiers
    if cm_output:
        print()
        print('Confusion matrices for KNN:')
        print(knn_results[0], '\n\n', knn_results[2])
        print()
        print('Confusion matrices for SVM:')
        print(svm_results[0], '\n\n', svm_results[2])
        print()
        print('Confusion matrices for Naive Bayes:')
        print(nb_results[0], '\n\n', nb_results[2])
        print()
        print('Confusion matrices for Desision Tree:')
        print(dt_results[0], '\n\n', dt_results[2])
        print()
        print('Confusion matrices for Random Forest:')
        print(rf_results[0], '\n\n', rf_results[2])
        
    # Prints k-fold results
    print()
    print(f'The k-fold accuracy means for normal / abnormal are: KNN: {knn_results[4]*100:.2f}%, '
          f'SVM: {svm_results[4]*100:.2f}%, Naive Bayes: {nb_results[4]*100:.2f}%, '
          f'Desision Tree: {dt_results[4]*100:.2f}%, '
          f'Random Forest: {rf_results[4]*100:.2f}%')
    print(f'The k-fold accuracy std for normal / abnormal are: KNN: {knn_results[5]*100:.2f}%, '
          f'SVM: {svm_results[5]*100:.2f}%, Naive Bayes: {nb_results[5]*100:.2f}%, '
          f'Desision Tree: {dt_results[5]*100:.2f}%, '
          f'Random Forest: {rf_results[5]*100:.2f}%')
    print()
    print(f'The k-fold accuracy means for normal / disk hernia / spondylolisthesis are: KNN: {knn_results[6]*100:.2f}%,'
          f' SVM: {svm_results[6]*100:.2f}%, Naive Bayes: {nb_results[6]*100:.2f}%, '
          f'Desision Tree: {dt_results[6]*100:.2f}%, '
          f'Random Forest: {rf_results[6]*100:.2f}%')
    print(f'The k-fold accuracy std for normal / disk hernia / spondylolisthesis are: KNN: {knn_results[7]*100:.2f}%, '
          f'SVM: {svm_results[7]*100:.2f}%, Naive Bayes: {nb_results[7]*100:.2f}%, '
          f'Desision Tree: {dt_results[7]*100:.2f}%, '
          f'Random Forest: {rf_results[7]*100:.2f}%')


def main():
    train_test_2c, train_test_3c = data_preprocessing()
    knn_results = model_train_evaluation(train_test_2c, train_test_3c, 'knn')  # k-nearest neighbors
    svm_results = model_train_evaluation(train_test_2c, train_test_3c, 'svm')  # support vector machine
    nb_results = model_train_evaluation(train_test_2c, train_test_3c, 'nb')  # naive Bayes
    dt_results = model_train_evaluation(train_test_2c, train_test_3c, 'dt')  # desicion tree
    rf_results = model_train_evaluation(train_test_2c, train_test_3c, 'rf')  # random forest
    results_comparison(knn_results, svm_results, nb_results, dt_results, rf_results, True)


if __name__ == '__main__':
    main()