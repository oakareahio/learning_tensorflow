import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_conf_matrix(cm,
                     classes,
                     normalize=False,
                     title='Confusion Matrix',
                     cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i,j]>thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig('confusion_matrix_mnist.png',
                    bbox_inches='tight',
                    format='png',
                    dpi=300,
                    pad_inches=0,
                    transparent=True)

DATA_DIR = './data/mnist'
data = input_data.read_data_sets(DATA_DIR)
x_data, y_data = data.train.images, data.train.labels.astype(np.int32)
x_test, y_test = data.test.images, data.test.labels.astype(np.int32)

NUM_STEPS = 2000
MINIBATCH_SIZE = 128

feature_columns = learn.infer_real_valued_columns_from_input(x_data)

dnn = learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[200],
    n_classes=10,
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.2
    )
)

dnn.fit(x=x_data,
        y=y_data,
        steps=NUM_STEPS,
        batch_size=MINIBATCH_SIZE
        )

test_accuracy = dnn.evaluate(x=x_test,
                             y=y_test,
                             steps=1)['accuracy']

print(f'test accuracy : {test_accuracy}')


y_pred = dnn.predict(x=x_test, as_iterable=False)
class_name = [str(i) for i in range(10)]
cnf_matrix = confusion_matrix(y_test, y_pred)

from IPython import embed;embed()

plot_conf_matrix(cnf_matrix, class_name)