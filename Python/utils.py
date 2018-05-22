from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from keras.layers import Input, Embedding, LSTM, Dense, merge, TimeDistributed
from keras.models import Sequential, Model

def classifiers_cv_score(X,Y):
    models = [
        KNeighborsClassifier(3),
        SVC(kernel="rbf", C=0.01),
        SVC(gamma=2, C=1),
        RandomForestClassifier(max_depth=100, n_estimators=1000, max_features='auto'),
        MLPClassifier(alpha=1),
        GaussianNB()]


    mod_names = ["KNN", "SVM1", "SVM2", "RF", "NN", "GNB"]
    best_score = 0
    best_model = ""
    for j in range(len(models)):
        classifier = models[j]

        score = cross_val_score(classifier, X, Y).mean()

        if score > best_score:
            best_score = score
            best_model = models[j]
            best_mname = mod_names[j]

        print ("[INFO] " + mod_names[j] + " CV_Score: " + str(score))

    print ("[INFO] Best CV_Score Achieved: " + str(best_score) + " by " + best_mname)
    return best_model


def classifiers_score(X,Y,split=0.2,rand_state=5):
    models = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        RandomForestClassifier(max_depth=100, n_estimators=1000, max_features=1),
        MLPClassifier(alpha=1),
        GaussianNB()]


    mod_names = ["KNN", "SVM1", "SVM2", "RF", "NN", "GNB"]
    best_score = 0
    best_model = ""

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=split,
                                                        random_state=rand_state)

    for j in range(len(models)):
        classifier = models[j]

        classifier.fit(X_train, Y_train)

        score = classifier.score(X_test, Y_test)

        if score > best_score:
            best_score = score
            best_model = models[j]
            best_mname = mod_names[j]

        print ("[INFO] " + mod_names[j] + " score: " + str(score))

    print ("[INFO] Best Score Achieved: " + str(best_score) + " by " + best_mname)
    return best_model


def simple_ann(input_size, output_size):

    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_size, activation='sigmoid'))

    return model

