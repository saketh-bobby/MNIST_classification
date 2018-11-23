import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import os

# Initializing the weights to Normal Distribution
def init_weights(shape,seed=42):
    np.random.seed(seed)
    return tf.Variable(tf.Variable(tf.random_normal(shape,stddev=0.01)), dtype=tf.float32)

def create_hidden_layer(inputTensor, input_hidden_weights, activation_function):    
    return tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))

# validation and testing accuracies
def get_accuracy(y_pred, actual_y):    
    wrong   = 0
    right   = 0
    for i, j in zip(actual_y, y_pred):  
        if i == j:
            right = right + 1
        else:
            wrong = wrong + 1

    return right, wrong, (right/(right+wrong)*100)


# In[11]:
def deep_neural_network(data, learning_rate=0.05, num_epochs=10, num_neurons = [100, 100, 100]):        
    np.random.seed(42)
    X_train, y_train, y_train_onehot, X_val, y_val, X_test, y_test, X_USPS, y_USPS = data
    NUM_HIDDEN_NEURONS_LAYER_1, NUM_HIDDEN_NEURONS_LAYER_2, NUM_HIDDEN_NEURONS_LAYER_3 = num_neurons

    inputTensor  =  tf.placeholder(tf.float32, [None, X_train.shape[1]])
    outputTensor = tf.placeholder(tf.float32, [None, 10])
    lr = tf.placeholder(tf.float32, name="learning_rate")

    # # optimal model
    batch_size = 128
    lambd = 0
    # CREATE MODEL FUNCTION
    # Initializing the input to hidden layer weights
    input_hidden_weights_1  = init_weights([int(X_train.shape[1]), NUM_HIDDEN_NEURONS_LAYER_1])

    # relu layer
    hidden_layer_1 = create_hidden_layer(inputTensor, input_hidden_weights_1, activation_function="relu")
    
    input_hidden_weights_2  = init_weights([int(hidden_layer_1.shape[1]), NUM_HIDDEN_NEURONS_LAYER_2])

    # relu layer
    hidden_layer_2 = create_hidden_layer(hidden_layer_1, input_hidden_weights_2, activation_function="relu")
        
    hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_3, 10])
    
    output_layer = tf.matmul(hidden_layer_2, hidden_output_weights)
        
    # Defining Error Function
    error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor)) + ( lambd * tf.reduce_sum(tf.square(tf.nn.l2_loss(hidden_output_weights))))

    # Defining Learning Algorithm and Training Parameters
    training = None

    training = tf.train.AdamOptimizer(learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
        ).minimize(error_function)
    
    # Prediction Function
    prediction = tf.argmax(output_layer, 1)


    # RUN MODEL NOW

    training_accuracies = []
    y_pred_val = []
    y_pred_test = []
    y_pred_USPS = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in tqdm_notebook(range(num_epochs)):
            print(epoch)
            #Shuffle the Training Dataset at each epoch
            p = np.random.permutation(range(X_train.shape[0]))
            X_train  = X_train[p]
            y_train_onehot = y_train_onehot[p]

            # Start batch training
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                sess.run(training, feed_dict={inputTensor: X_train[start:end], 
                                            outputTensor: y_train_onehot[start:end],
                                            lr: learning_rate
                                            })
                # Training accuracy for an epoch
            training_accuracies.append(np.mean(np.argmax(y_train_onehot, axis=1) ==
                sess.run(prediction, feed_dict={inputTensor: X_train,
                                                    outputTensor: y_train_onehot})))
        # Validation
        y_pred_val.append(sess.run(prediction, feed_dict={inputTensor: X_val}))
        # Testing
        y_pred_test.append(sess.run(prediction, feed_dict={inputTensor: X_test}))
        # USPS Numerals
        y_pred_USPS.append(sess.run(prediction, feed_dict={inputTensor: X_USPS}))

    print("Training Accuracy for learning rate %s is: %s" % (learning_rate, training_accuracies[-1] * 100 ))

    print(np.array(y_pred_val).shape, y_val.shape)
    right, wrong, accuracy = get_accuracy(np.array(y_pred_val).T, y_val)
    print("Errors: " + str(wrong), " Correct :" + str(right))
    print("Validation Accuracy for learning rate %s is: %s" % ( learning_rate, accuracy ) )

    right, wrong, test_acc_dnn = get_accuracy(np.array(y_pred_test).T, y_test)
    print("Errors: " + str(wrong), " Correct :" + str(right))
    print("Testing Accuracy for learning rate %s is: %s" % ( learning_rate, test_acc_dnn ) )

    right, wrong, USPS_acc_dnn = get_accuracy(np.array(y_pred_USPS).T, y_USPS)
    print("Errors: " + str(wrong), " Correct :" + str(right))
    print("USPS Accuracy for learning rate %s is: %s" % ( learning_rate, USPS_acc_dnn ) )

    # cm = confusion_matrix(np.array(y_pred_test).reshape(-1, 1), y_test)
    # print(cm)

    # cm = confusion_matrix(np.array(y_pred_USPS).reshape(-1, 1), y_USPS)
    # print(cm)


    return y_pred_test, y_pred_USPS, test_acc_dnn, USPS_acc_dnn