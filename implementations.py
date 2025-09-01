import numpy as np

from helpers import *



###########################################################
####### LIST OF THE 6 ASKED FUNCTIONS TO IMPLEMENT ########
###########################################################


#***** LOSS *****************************************************************************

def compute_loss_mse(y, tx, w):
    """Calculate the loss using either MSE or MAE.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.
    Returns:
        loss: the value of the loss (a scalar), corresponding to the input parameters w.
    """
    #loss by MSE
    N=len(y)
    loss=1/(2*N) * np.sum((y - tx.dot(w))**2)
    return loss




#***** GRADIENT DESCENT (MSE) ***********************************************************

def compute_gradient_mse(y, tx, w):
    """Computes the gradient at w.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.
    Returns:
        gard: shape=(D, ) (same shape as w), containing the gradient of the loss at w.
    """
    N=len(y)
    e=(y - tx.dot(w))
    grad= - (1/N) * tx.T.dot(e) #gradient for mean square loss (MSE)
    return grad

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        w: shape=(D, ), last weight w after max_iters iteration of SGD
        loss: scalar, the last loss after max_iters iteration of SGD
    """
    w = initial_w
    loss = compute_loss_mse(y,tx,w)
    for n_iter in range(max_iters):

        grad=compute_gradient_mse(y,tx,w) #MSE
        w=w-gamma*grad
        loss=compute_loss_mse(y,tx,w)     #MSE


        if (n_iter==max_iters-1):
            print(
                "GD iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )

    return w, loss




#***** STOCHASTIC GRADIENT DESCENT (MSE) ************************************************

def compute_stoch_gradient_mse(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
    Args:
        y: shape=(B, )
        tx: shape=(B,D)
        w: shape=(D, ). The vector of model parameters.
    Returns:
        grad: An array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    B=len(y)
    e=(y - tx.dot(w))
    grad= - (1/B) * tx.T.dot(e) #gradient for mse
    return grad

def mean_squared_error_sgd(y, tx, initial_w, max_iters=500, gamma=0.1):
    """The Stochastic Gradient Descent algorithm (SGD).
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: scalar, the last loss after max_iters of iteration of SGD 
        w: shape=(D, ), last weight w after max_iters of iteration of SGD
    """
    w = initial_w
    loss = compute_loss_mse(y,tx,w)

    for n_iter in range(max_iters):
     
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1): 
            grad=compute_stoch_gradient_mse(minibatch_y, minibatch_tx, w) #grad for mse
            w=w-gamma*grad
            loss=compute_loss_mse(minibatch_y,minibatch_tx,w)         


        if (n_iter==max_iters-1):
            print(
                "SGD iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )
    return w, loss



#***** LEAST SQUARE *********************************************************************

def least_squares(y, tx):
    """Compute w for least square regression using normal equations
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
    Returns:
        w: shape=(D, ), weight w
        loss: scalar, the mean squared error
    """
    #N = len(y)
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss_mse(y,tx,w)         #1/(2*N) * np.sum((y - tx.dot(w))**2)
    return w, loss



#***** RIDGE REGRESSION *****************************************************************

def ridge_regression(y, tx, lambda_):
    """Compute w for least square regression using normal equations, with a penalty term
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        lambda_: scalar (penalty term)
    Returns:
        w: shape=(D, ), weight w
        loss: scalar, the mean squared error
    """
    N=len(y)
    w = np.linalg.solve(tx.T @ tx + 2*N*lambda_ * np.eye(tx.shape[1]), tx.T @ y)
    loss = compute_loss_mse(y,tx,w)   #NO ADDED LOSS TERM  
    return w, loss



#*****  LOGISTIC REGRESSION *************************************************************

def sigmoid(t):
    """apply sigmoid function on t.
    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array
    """
    return 1/(1+np.exp(-t))

def compute_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    Returns:
        loss: a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    if np.all((y == -1) | (y == 1)):
        y = (y + 1)/2 #ATTENTION, modifie +1 -1 en 1 0

    N=len(y)
    sigm=sigmoid(tx.dot(w))
    loss= -(1/N) * np.sum( y * np.log(sigm) + (1 - y) * np.log(1 - sigm)) 
    return loss

def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    Returns:
        grad: a vector of shape (D, 1)
    """
    if np.all((y == -1) | (y == 1)):
        y = (y + 1)/2 #ATTENTION, modifie +1 -1 en 1 0

    N=len(y)
    sigm=sigmoid(tx.dot(w))
    grad= (1/N) * tx.T @ (sigm - y)
    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """The logistic regression using gradient Descent (GD) algorithm.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        w: shape=(D, ), last weight w after max_iters iteration of GD
        loss: scalar, the last loss after max_iters iteration of GD
    """
    #CAREFUL -> we work with +1/-1 labels, so the y need to be changed, and are changed in the loss and grad calculations already
    #y = (y + 1)/2      
  
    w = initial_w
    loss = compute_logistic_loss(y,tx,w)

    for n_iter in range(max_iters):

        grad=compute_logistic_gradient(y,tx,w) #LOGISTIC
        w=w-gamma*grad
        loss=compute_logistic_loss(y,tx,w)     #LOGISTIC

        if  (n_iter==max_iters-1):
            print(
                "logistic iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )

    return w, loss



#***** REGULARIZED LOGISTIC REGRESSION **************************************************

def compute_reg_logistic_gradient(y, tx, w, lambda_):
    """compute the gradient of loss.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    Returns:
        grad: a vector of shape (D, 1)
    """
    grad=compute_logistic_gradient(y,tx,w) + 2*lambda_*w
    return grad

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """The logistic regression using gradient Descent (GD) algorithm.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        w: shape=(D, ), last weight w after max_iters iteration of GD
        loss: scalar, the last loss after max_iters iteration of GD
    """
    #CAREFUL -> we work with +1/-1 labels, but this function need 1/0 labels
    #y = (y + 1)/2   changed in the loss and grad fct  

    w = initial_w
    loss = compute_logistic_loss(y,tx,w)

    for n_iter in range(max_iters):

        grad=compute_reg_logistic_gradient(y,tx,w,lambda_)
        w=w-gamma*grad
        loss=compute_logistic_loss(y,tx,w)

        if (n_iter==max_iters-1):
            print(
                "reg logistic iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )

    return w, loss








#################################################################################
####### FONCTIONS EN PLUS POUR LE PROJET, PAS DEMANDEES DANS LA CONSIGNE ########
#################################################################################


#***** SPLIT DATA TRAIN SET /VALIDATION SET *********************************************

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """
    # set seed
    np.random.seed(seed)

    indices=np.random.permutation(len(y))
    split=int(np.floor(len(y)*ratio))

    train_indices=indices[:split]
    test_indices=indices[split:]

    x_train=x[train_indices]
    x_test=x[test_indices]
    
    y_train=y[train_indices]
    y_test=y[test_indices]

    return x_train, x_test, y_train, y_test

#***** RANDOM SAMPLE WITH GIVEN PROPORTION  *********************************************

def sample_with_proportion(y, proportion_neg):
    neg_indices = np.where(y == -1)[0]
    pos_indices = np.where(y == 1)[0]
    
    num_neg = int(len(y) * proportion_neg)
    num_pos = len(y) - num_neg
    
    sampled_neg_indices = np.random.choice(neg_indices, num_neg, replace=False)
    sampled_pos_indices = np.random.choice(pos_indices, num_pos, replace=False)
    
    sampled_indices = np.concatenate([sampled_neg_indices, sampled_pos_indices])
    np.random.shuffle(sampled_indices)
    sampled_values = y[sampled_indices]
    
    return sampled_values, sampled_indices

#***** FOR THE K FOLD *******************************************************************

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


#***** STANDARDIZE THE DATASETS *********************************************************

def standardize(x):
    """Stadartize the input data x
    Args:
        x: numpy array of shape=(num_samples, num_features)
    Returns:
        standartized data, shape=(num_samples, num_features)
    """
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0) #attention division par 0



#***** LOGISTIC L1 ***********************************************************************

def L1_logistic_gradient(y, tx, w, lambda_):
    """Compute the gradient of the custom logistic loss.
    Args:
        y: shape=(N, 1) - PyTorch tensor
        tx: shape=(N, D) - PyTorch tensor
        w: shape=(D, 1) - PyTorch tensor
    Returns:
        grad: a vector of shape (D, 1)
    """
    # Compute the standard logistic gradient
    grad = compute_logistic_gradient(y, tx, w) + lambda_ * np.sign(w)
        
    return grad 


def regL1_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression with custom regularization using gradient descent.
    Args:
        y: shape=(N, ) - PyTorch tensor
        tx: shape=(N, D) - PyTorch tensor
        initial_w: shape=(D,) - PyTorch tensor
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        w: shape=(D,) - PyTorch tensor, last weight w after max_iters iterations
        loss: scalar, the last loss after max_iters iterations
    """
    w = initial_w
    loss = 0

    for n_iter in range(max_iters):

        grad = L1_logistic_gradient(y, tx, w, lambda_)
        
        #w = w - gamma/(n_iter/100 + 1) * grad
        w = w - gamma * grad

        if (n_iter == max_iters - 1):
            loss = compute_logistic_loss(y, tx, w)
            print(f"Custom reg logistic iter. {n_iter}/{max_iters - 1}: loss={loss}")

    return w, loss


#***** SVM LOSS *********************************************************************************

def calculate_primal_objective(y, X, w, lambda_):
    """compute the full cost (the primal objective, equation (1) in the exercise pdf),
        that is loss plus regularizer.

    Args:
        X: the full dataset matrix, shape = (num_examples, num_features)
        y: the corresponding +1 or -1 labels, shape = (num_examples)
        w: shape = (num_features)

    Returns:
        scalar, non-negative

    """

    N=len(y)
    l = 1/N * np.sum(np.maximum(0, 1 - y * (X @ w))) # + (lambda_/2) * np.sum(w ** 2) NO REG TERM IN THE LOSS
    loss=np.min(l)
    return loss

def calculate_stochastic_gradient(y, X, w, lambda_, n):
    """compute the stochastic gradient of loss plus regularizer.

    Args:
        X: the dataset matrix, shape = (num_examples, num_features)
        y: the corresponding +1 or -1 labels, shape = (num_examples)
        w: shape = (num_features)
        lambda_: positive scalar number
        n: the index of the (one) datapoint we have sampled
        num_examples: N

    Returns:
        numpy array, shape = (num_features)

    """
    #FULL GRAD
    grad = (1/len(y)) * (-X.T @ (y * np.where(1 - y * (X @ w) > 0, 1.0, 0.0))) + lambda_ * w

    #STOCHASTIC: -> n for only one point
    #X[n] shape (k,)
    #grad = -y[n] * X[n] * np.where(1-y[n]*X[n].dot(w)>0, 1, 0) + lambda_ * w    #je peux mettre aussi juste (1-y[n]*X[n].dot(w)>0) à la place du np where

    return grad

def sgd_for_svm_demo(y, X, lambda_, initial_w, max_iter, gamma):
   
    num_examples = X.shape[0]
    w = initial_w
    loss = 0
 
    for it in range(max_iter):
        # n = sample one data point uniformly at random data from x
        n = np.random.randint(0, num_examples - 1)
    
        grad = calculate_stochastic_gradient(y, X, w, lambda_,n)
        #w -= gamma * grad
        w -= gamma / (it/5 + 1) * grad

        if (it == max_iter - 1):
            loss = calculate_primal_objective(y, X, w, lambda_)
            print(
                "SVM iter. {bi}/{ti}: loss={l}".format(
                    bi=it, ti=max_iter - 1, l=loss
                )
            )

    return w, loss


#***** TO EVALUATE OUR PREDICTIONS, and get our f1 score *********************************************

def classification_metrics2(y_pred, y_val, returnF1=False):
    # Accuracy globale
    accuracy = np.sum(y_val == y_pred) / len(y_val)
    
    # Calcul for class 1 (crise cardiaque)
    crise_card_pred = (y_pred == 1)
    crise_card_val = (y_val == 1)
    crise_card_pred_correct = crise_card_pred & crise_card_val
    
    precision_crise = np.sum(crise_card_pred_correct) / np.sum(crise_card_pred) if np.sum(crise_card_pred) != 0 else 0
    recall_crise = np.sum(crise_card_pred_correct) / np.sum(crise_card_val) if np.sum(crise_card_val) != 0 else 0
    f1_score_crise = (2 * precision_crise * recall_crise) / (precision_crise + recall_crise) if (precision_crise + recall_crise) != 0 else 0
    support_crise = np.sum(crise_card_val)
    
    # Calcul for class -1 (not crise cardiaque)
    non_crise_card_pred = (y_pred == -1)
    non_crise_card_val = (y_val == -1)
    non_crise_card_pred_correct = non_crise_card_pred & non_crise_card_val
    
    precision_non_crise = np.sum(non_crise_card_pred_correct) / np.sum(non_crise_card_pred) if np.sum(non_crise_card_pred) != 0 else 0
    recall_non_crise = np.sum(non_crise_card_pred_correct) / np.sum(non_crise_card_val) if np.sum(non_crise_card_val) != 0 else 0
    f1_score_non_crise = (2 * precision_non_crise * recall_non_crise) / (precision_non_crise + recall_non_crise) if (precision_non_crise + recall_non_crise) != 0 else 0
    support_non_crise = np.sum(non_crise_card_val)

    # Résumé des métriques pour chaque classe
    if not returnF1:
        print(f"Accuracy: {accuracy:.5f}")
        print(f"Non crise:    precision={precision_non_crise:.5f}     recall={recall_non_crise:.5f}     f1-score={f1_score_non_crise:.5f}     support={support_non_crise}")
        print(f"Crise card:   precision={precision_crise:.5f}     recall={recall_crise:.5f}    cf1-score={f1_score_crise:.5f}     support={support_crise}")
    if returnF1:
        return f1_score_crise, f1_score_non_crise
    


#***** DATA PREPROCESSING ***********************************************************************

def convert_weight_to_pounds(weight_column):
    # Initialize an array to store the converted weights
    converted_weights = np.zeros(len(weight_column))

    for i, weight in enumerate(weight_column):
        # Convert weights in pounds (50–999)
        if 50 <= weight <= 999:
            converted_weights[i] = weight

        # Convert weights in kilograms (9000–9998) to pounds
        elif 9000 <= weight <= 9998:
            kg_value = weight - 9000  # Remove leading '9'
            pounds_value = kg_value * 2.20462  # Convert to pounds
            converted_weights[i] = pounds_value
    return converted_weights

def convert_height_to_inches(height_column):
    # Initialize an array to store the converted heights
    converted_heights = np.zeros(len(height_column))

    for i, height in enumerate(height_column):
        # Convert heights in feet and inches (200–711)
        if 200 <= height <= 711:
            feet = height // 100           # Extract the 'feet' part
            inches = height % 100          # Extract the 'inches' part
            total_inches = (feet * 12) + inches  # Convert to inches
            converted_heights[i] = total_inches

        # Convert heights in meters and centimeters (9000–9998)
        elif 9000 <= height <= 9998:
            cm_value = height - 9000         # Remove leading '9'
            inches_value = cm_value / 2.54   # Convert centimeters to inches
            converted_heights[i] = inches_value
    return converted_heights

def ProcessFullData(dataset,feature_names, normalisation):
    index_rows=np.array(range(dataset.shape[0]))
    #remove IDATE, SEQNO, _PSU
    data=np.delete(dataset, [2,7,8], axis=1)
    feature_names=np.delete(feature_names, [2,7,8])

    # #process weight abd height
    INDEX=np.array(range(data.shape[1]))
    idx_weight=INDEX[feature_names=='WEIGHT2']
    idx_height=INDEX[feature_names=='HEIGHT3']

    weight_pounds=convert_weight_to_pounds(data[:,idx_weight])
    data[:,idx_weight]=np.expand_dims(weight_pounds,axis=1)

    height_inshes=convert_height_to_inches(data[:,idx_height])
    data[:,idx_height]=np.expand_dims(height_inshes, axis=1)

    INDEX=np.array(range(data.shape[1]))

    #9-----------------------------
    index9=np.array(range(data.shape[1]))

    indexA9=index9[np.any(data>9, axis=0)]
    Above9=data[:,indexA9]
    print(Above9.shape)

    indexB=index9[~np.any(data>9, axis=0)]
    Below9=data[:,indexB]

    #Fill NAN values with 9
    Below9[np.isnan(Below9)]=9

    #99-----------------------------
    index99=np.array(range(Above9.shape[1]))

    indexA99=index99[np.any(Above9>99, axis=0)]
    Above99=Above9[:,indexA99]

    indexB99=index99[~np.any(Above9>99, axis=0)]
    Below99=Above9[:,indexB99]

    Below99[np.isnan(Below99)]=99
    #999---------------------------
    index999=np.array(range(Above99.shape[1]))

    indexA999=index999[np.any(Above99>999, axis=0)]
    Above999=Above99[:,indexA999]

    indexB999=index999[~np.any(Above99>999, axis=0)]
    Below999=Above99[:,indexB999]

    Below999[np.isnan(Below999)]=999
    #9999---------------------------
    index9999=np.array(range(Above999.shape[1]))

    indexA9999=index9999[np.any(Above999>9999, axis=0)]
    Above9999=Above999[:,indexA9999]

    indexB9999=index9999[~np.any(Above999>9999, axis=0)]
    Below9999=Above999[:,indexB9999]

    Below9999[np.isnan(Below9999)]=9999
    #99999---------------------------
    index99999=np.array(range(Above9999.shape[1]))

    indexA99999=index99999[np.any(Above9999>99999, axis=0)]
    Above99999=Above9999[:,indexA99999]

    indexB99999=index99999[~np.any(Above9999>99999, axis=0)]
    Below99999=Above9999[:,indexB99999]

    Below99999[np.isnan(Below99999)]=99999
    #999999---------------------------
    index999999=np.array(range(Above99999.shape[1]))

    indexA999999=index999999[np.any(Above99999>999999, axis=0)]
    Above999999=Above99999[:,indexA999999]

    indexB999999=index999999[~np.any(Above99999>999999, axis=0)]
    Below999999=Above99999[:,indexB999999]

    Below999999[np.isnan(Below999999)]=999999
    #put all together and shuffle
    dataset_filled=np.concatenate((Below9, Below99, Below999, Below9999, Below99999, Below999999), axis=1)
    shuffled_indices = np.random.permutation(dataset_filled.shape[1])
    dataset_filled=dataset_filled[:,shuffled_indices]
    print(dataset_filled.shape)

    #normaliser et detrend
    if normalisation==True:
        means=np.mean(dataset_filled, axis=0)
        std=np.std(dataset_filled, axis=0)
        dataset_filled_norm=(dataset_filled-means)/std
        dataset_filled_norm  
        return dataset_filled_norm, shuffled_indices, feature_names
    else:
        return dataset_filled, shuffled_indices, feature_names
    

    
def normalize_and_fill(dataset, feature_names):
    index_rows=np.array(range(dataset.shape[0]))
    #remove IDATE, SEQNO, _PSU
    data=np.delete(dataset, [2,7,8], axis=1)
    feature_names=np.delete(feature_names, [2,7,8])

    # #process weight abd height
    INDEX=np.array(range(data.shape[1]))
    idx_weight=INDEX[feature_names=='WEIGHT2']
    idx_height=INDEX[feature_names=='HEIGHT3']

    weight_pounds=convert_weight_to_pounds(data[:,idx_weight])
    data[:,idx_weight]=np.expand_dims(weight_pounds,axis=1)

    height_inshes=convert_height_to_inches(data[:,idx_height])
    data[:,idx_height]=np.expand_dims(height_inshes, axis=1)

    INDEX=np.array(range(data.shape[1]))
    X_normalized = np.copy(data)
    
    # Normalize each feature by ignoring NaN values
    for i in range(data.shape[1]):
        # Get the current column, ignoring NaNs for mean and std calculation
        column = data[:, i]
        mean = np.nanmean(column)  # Mean, ignoring NaNs
        std = np.nanstd(column)    # Std deviation, ignoring NaNs
        
        # Normalize: (X - mean) / std, only where values are not NaN
        X_normalized[:, i] = np.where(~np.isnan(column), (column - mean) / std, column)
    
    # Replace all remaining NaNs with 0
    X_normalized = np.nan_to_num(X_normalized, nan=0)
    
    return X_normalized



    
def downsample_data(x_train, y_train, n):
    """
    Downsample the dataset to have an equal number of samples for each class in y_train.
    
    Parameters:
        X_train (numpy.ndarray): The input features with shape (n_samples, n_features).
        y_train (numpy.ndarray): The target labels with shape (n_samples,).
        
    Returns:
        X_train_downsampled (numpy.ndarray): Downsampled input features.
        y_train_downsampled (numpy.ndarray): Downsampled target labels.
    """
    # Separate the samples for each class
    class_1_indices = np.where(y_train == 1)[0]
    class_neg1_indices = np.where(y_train == -1)[0]
    
    # Determine the minority class count
    n_samples = min(len(class_1_indices), len(class_neg1_indices))
    
    # Randomly downsample both classes to the size of the minority class
    downsampled_class_1_indices = np.random.choice(class_1_indices, n_samples, replace=False)
    downsampled_class_neg1_indices = np.random.choice(class_neg1_indices, n_samples * n, replace=False)
    
    # Combine downsampled indices and shuffle
    downsampled_indices = np.concatenate([downsampled_class_1_indices, downsampled_class_neg1_indices])
    np.random.shuffle(downsampled_indices)
    
    # Select downsampled X and y
    X_train_downsampled = x_train[downsampled_indices]
    y_train_downsampled = y_train[downsampled_indices]
    
    return X_train_downsampled, y_train_downsampled



def perform_pca(X, n_components):
    """
    Perform PCA on the dataset X using NumPy.
    
    Parameters:
        X (numpy.ndarray): The input dataset with shape (n_samples, n_features).
        n_components (int): Number of principal components to keep.
        
    Returns:
        X_reduced (numpy.ndarray): Transformed dataset with reduced dimensions (n_samples, n_components).
        explained_variance (numpy.ndarray): The amount of variance explained by each of the selected components.
    """
    
    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)
    
    # Step 3: Calculate eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Step 4: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 5: Select the top n_components eigenvectors (principal components)
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]
    
    # Step 6: Transform the dataset
    X_reduced = np.dot(X, selected_eigenvectors)
    
    # Calculate the explained variance for each selected component
    explained_variance = sorted_eigenvalues[:n_components] / np.sum(sorted_eigenvalues)
    
    return X_reduced, explained_variance, selected_eigenvectors
