# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import svm
from data import DataSet # to import training and testing data 
data_type = 'features'
seq_length = 60
class_limit =  6
image_shape = None
data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        ) 
# loading the footstep dataset
X_tr, y_tr = data.get_all_sequences_in_memory('train', data_type)
X_train= X_tr.reshape(780,1080)
y_train = np.zeros(780)
from sklearn.utils import shuffle
# convert labels one hot vector into labels
j = 0
for i in y_tr:
    y_train[j] = np.argmax(i)
    j +=1
X_te, y_te = data.get_all_sequences_in_memory('test', data_type)
X_test = X_te.reshape(192,1080)
y_test = np.zeros(192)
j = 0
for i in y_te:
    #print(np.argmax(i))
    y_test[j] = np.argmax(i)
    j +=1
# shuffle the dataset
X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_test,y_test = shuffle(X_test,y_test,random_state=0)
# training a linear SVM classifier
def svc_param_selection(X, y, nfolds):
    Cs = [1, 5, 10, 100]
    #Cs = [x for x in np.arange(1,10,1)]
    gammas = [0.001, 0.1, 1,5,10]
    #gammas = [x for x in np.arange(1,10,1)]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    #grid_search.best_params_
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    return grid_search.best_params_

# Function to grid search hyper parameter for SVM
grid_search_best_params_ = svc_param_selection(X_train,y_train,5)
C = grid_search_best_params_['C']
gamma= grid_search_best_params_['gamma']
# SVM is fit using best C and Gamma values
clf = svm.SVC(kernel='rbf', C=C,gamma = gamma)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
svm_model_linear = svm.SVC(C=C ,gamma= gamma,kernel='rbf').fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
# model accuracy for X_test  
#accuracy_ = svm_model_linear.score(X_train,y_train)
accuracy = svm_model_linear.score(X_test, y_test)
#print(accuracy_)
print(accuracy)
""" # creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
class_names = [1,2,3,4,5,6]
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cm, classes=class_names,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()   """