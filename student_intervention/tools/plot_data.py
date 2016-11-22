From matplotlib import pyplot as plt

#define how many training steps we will evaluate.
Step_size = 10

Def pl_plot (clf, features, target):
    Y_pred = clf.predict (features)
    Return f1_score (target.values, y_pred, pos_label = 'yes')

Def tc_plot (clf, X_train, and_train):
    Clf.fit (X_train, y_train)

# Method that is responsible for retrieving the F1 index of each classifier for different sample sizes
Def get_F1 (clf, training_set_size):
        X_train_sample = X_train [: training_set_size]
        Y_train_sample = y_train [: training_set_size]
        Tc_plot (clf, X_train_sample, y_train_sample)
        Return pl_plot (clf, X_test, y_test)

#Here we have created a list with all F1 results for each of the classifiers
F1_list_all = []
For clf in [clf_A, clf_B, clf_C]:
    F1_list = []
    For training_set_size in range (50,301, step_size):
        F1_list.append (get_F1 (clf, training_set_size))
    F1_list_all.append (f1_list)

# Here we generate a plot that will reveal the F1 evolution of each classifier
Plt.plot (range (50,301, step_size), f1_list_all [0], label = 'Naive Baeys')
Plt.plot (range (50,301, step_size), f1_list_all [1], label = 'Decision tree')
Plt.plot (range (50,301, step_size), f1_list_all [2], label = 'SVM')

#Configuring axes
Plt.xlabel ( 'Training set size')
Plt.ylabel ( 'F1 score')
Plt.legend (loc = 4)

#generating graph
Plt.show ()

Print 'F1 average for:'
Print 'Naive Bayes: {: .4f}'. Format (np.mean (f1_list_all [0]))
Print 'Decision tree: {: .4f}'. Format (np.mean (f1_list_all [1]))
Print 'SVM: {: .4f}'. Format (np.mean (f1_list_all [2]))