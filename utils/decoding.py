import numpy as np

from utils import aa_letters


def to_string(seqmat):
    a = [''.join([aa_letters[np.argmax(aa)] for aa in seq]) for seq in seqmat]
    return a


def greedy_decode_1d(arr1d):
    a = np.zeros(arr1d.shape)
    i = np.argmax(arr1d)
    a[i] = 1
    return a


def greedy_decode(pred_mat):
    return np.apply_along_axis(greedy_decode_1d, -1, pred_mat)


# b = np.array([[1,2,3], [4,5,6], [7,8,9]])
# np.apply_along_axis(greedy_decode_1d, -1, b)
# array([[[1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1]],
#        [[1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1]],
#        [[1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1]]])


def _decode_nonar(generator, z):
    xp = generator.predict(z)
    x = greedy_decode(xp)
    return to_string(x)


def batch_temp_sample(preds, temperature=1.0):
    batch_sampled_aas = []
    for s in preds:
        batch_sampled_aas.append(temp_sample_outputs(s, temperature=temperature))
    out = np.array(batch_sampled_aas)
    return out


def temp_sample_outputs(preds, temperature=1.0):
    # https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
    # helper function to sample an index from a probability array N.B. this works on single prob array (i.e. array for one sequence at one timestep)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
