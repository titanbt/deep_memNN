from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T
import lasagne.updates
import pickle
import os
import codecs
from sklearn.metrics import f1_score
# Adapted from Lasagne

def adagrad(loss_or_grads, params, learning_rate=1.0, epsilon=1e-6):
    """Adagrad updates
    Scale learning rates by dividing with the square root of accumulated
    squared gradients. See [1]_ for further description.
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    epsilon : float or symbolic scalar
        Small value added for numerical stability
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    Using step size eta Adagrad calculates the learning rate for feature i at
    time step t as:
    .. math:: \\eta_{t,i} = \\frac{\\eta}
       {\\sqrt{\\sum^t_{t^\\prime} g^2_{t^\\prime,i}+\\epsilon}} g_{t,i}
    as such the learning rate is monotonically decreasing.
    Epsilon is not included in the typical formula, see [2]_.
    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
           Adaptive subgradient methods for online learning and stochastic
           optimization. JMLR, 12:2121-2159.
    .. [2] Chris Dyer:
           Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
    """

    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    accus = []

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        accus.append((accu,value.shape))
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates,accus

def reset_grads(accus):
    for accu in accus:
        accu[0].set_value(np.zeros(accu[1], dtype=accu[0].dtype))

def read_model_data(model, model_path, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./weights/'+model_path+'/', '%s.%s' % (filename, 'params'))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./weights/', filename)
    filename = '%s.%s' % (filename, 'params')
    with open(filename, 'w+') as f:
        pickle.dump(data, f)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            yield shuffled_data[start_index:end_index]


def iterate_minibatches(inputs, targets, masks=None, char_inputs=None, lexicons=None, batch_size=10, shuffle=False):
    assert len(inputs) == len(targets)
    if masks is not None:
        assert len(inputs) == len(masks)
    if char_inputs is not None:
        assert len(inputs) == len(char_inputs)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(len(inputs) // batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        targets = np.asarray(targets)
        yield inputs[excerpt], targets[excerpt], (None if masks is None else masks[excerpt]), \
              (None if char_inputs is None else char_inputs[excerpt]), lexicons[excerpt]


def get_lex_file_list(lex_file_path):
    lex_file_list = []
    with open(lex_file_path, 'rt') as handle:
        for line in handle.readlines():
            path = line.strip()

            if os.path.isfile(path):
                lex_file_list.append(path)
            else:
                print 'wrong file name(s) in the lex_config.txt\n%s' % path
                return None

    return lex_file_list

def minibatches_iter(inputs, aspects, targets, masks=None, char_inputs=None, batch_size=10, shuffle=False):
    assert len(inputs) == len(targets)
    assert len(inputs) == len(aspects)
    if masks is not None:
        assert len(inputs) == len(masks)
    if char_inputs is not None:
        assert len(inputs) == len(char_inputs)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs), batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        targets = np.asarray(targets)
        aspects = np.asarray(aspects)
        yield inputs[excerpt], aspects[excerpt], targets[excerpt], (None if masks is None else masks[excerpt]), \
              (None if char_inputs is None else char_inputs[excerpt])

def compute_f1_score(y_true, y_preds, labels):
    if len(labels.instances) > 2:
        predictions = []
        for y in y_preds:
            for x in y:
                predictions.append(int(x))
        confusion_matrix = {"positive_positive": 0, "positive_neutral": 0, "positive_negative": 0,
                            "neutral_positive": 0, "neutral_neutral": 0, "neutral_negative": 0,
                            "negative_positive": 0, "negative_neutral": 0, "negative_negative": 0}
        for i, pred in enumerate(predictions):
            confusion_matrix[labels.instances[pred]+ "_" + labels.instances[y_true[i]]] += 1

        pi_p = confusion_matrix["positive_positive"] / float(confusion_matrix["positive_positive"] +
                                                             confusion_matrix["positive_neutral"] +
                                                             confusion_matrix["positive_neutral"])

        p_p = confusion_matrix["positive_positive"] / float(confusion_matrix["positive_positive"] +
                                                            confusion_matrix["neutral_positive"] +
                                                            confusion_matrix["negative_positive"])

        pi_n = confusion_matrix["negative_negative"] / float(confusion_matrix["negative_negative"] +
                                                             confusion_matrix["negative_neutral"] +
                                                             confusion_matrix["negative_positive"])

        p_n = confusion_matrix["negative_negative"] / float(confusion_matrix["negative_negative"] +
                                                            confusion_matrix["neutral_negative"] +
                                                            confusion_matrix["positive_negative"])
        f1_p = (2 * pi_p * p_p) / float(pi_p + p_p)
        f1_n = (2 * pi_n * p_n) / float(pi_n + p_n)
        f1_pn = (f1_p + f1_n) / float(2)

        return f1_pn
    return 0

def print_predicted_abels(accuracy_test, filename):
    f = codecs.open('./results/'+filename, 'w+', 'utf-8')
    for i in accuracy_test:
        for j in i:
            f.write(str(j) + '\n')
    f.close()


