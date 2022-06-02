"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
"""
import numpy as np
import cupy as cp
from random import uniform
import sys


def sigmoid(x):
    return 1 / (1 + cp.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def dtanh(x):
    return 1 - x * x


# The numerically stable softmax implementation
def softmax(x):
    # assuming x shape is [feature_size, batch_size]
    e_x = cp.exp(x - cp.max(x, axis=0))
    return e_x / e_x.sum(axis=0)


# data I/O
#data = open('data/input.txt', 'r').read()  # should be simple plain text file
data = open('data/THE_CRITIQUE_OF_PRACTICAL_REASON.txt', 'r').read()  # should be simple plain text file
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
std = 0.1

option = sys.argv[1]

# hyperparameters
emb_size = 16 #16
hidden_size = 2048 #256  # size of hidden layer of neurons
seq_length = 512#128  # number of steps to unroll the RNN for
learning_rate = 5e-2 #5e-2
max_updates = 500000#500000
batch_size = 32#32

concat_size = emb_size + hidden_size

# model parameters
# char embedding parameters
Wex = cp.random.randn(emb_size, vocab_size) * std  # embedding layer

# LSTM parameters
Wf = cp.random.randn(hidden_size, concat_size) * std  # forget gate
Wi = cp.random.randn(hidden_size, concat_size) * std  # input gate
Wo = cp.random.randn(hidden_size, concat_size) * std  # output gate
Wc = cp.random.randn(hidden_size, concat_size) * std  # c term

bf = cp.zeros((hidden_size, 1))  # forget bias
bi = cp.zeros((hidden_size, 1))  # input bias
bo = cp.zeros((hidden_size, 1))  # output bias
bc = cp.zeros((hidden_size, 1))  # memory bias

# Output layer parameters
Why = cp.random.randn(vocab_size, hidden_size) * std  # hidden to output
by = cp.random.randn(vocab_size, 1) * std  # output bias

data_stream = cp.asarray([char_to_ix[char] for char in data])
print(data_stream.shape)

bound = (data_stream.shape[0] // (seq_length * batch_size)) * (seq_length * batch_size)
cut_stream = data_stream[:bound]
cut_stream = cp.reshape(cut_stream, (batch_size, -1))


def forward(inputs, targets, memory):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    hprev, cprev = memory
    xs, wes, hs, ys, ps, cs, zs, ins, c_s, ls = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    os, fs = {}, {}
    hs[-1] = cp.copy(hprev)
    cs[-1] = cp.copy(cprev)

    loss = 0
    input_length = inputs.shape[0]

    # forward pass
    for t in range(input_length):
        #xs = cp.asnumpy(xs)
        inputs = cp.asnumpy(inputs)
        xs[t] = np.zeros((vocab_size, batch_size))  # encode in 1-of-k representation
        for b in range(batch_size):
            xs[t][inputs[t][b]][b] = 1
        xs[t] = cp.asarray(xs[t])
        inputs = cp.asarray(inputs)
        # convert word indices to word embeddings
        wes[t] = Wex @ xs[t]

        # LSTM cell operation
        # first concatenate the input and h to get z
        zs[t] = cp.vstack((hs[t - 1], wes[t]))


        # compute the forget gate
        # f = sigmoid(Wf * z + bf)
        fs[t] = sigmoid(Wf @ zs[t] + bf)
        # compute the input gate
        # i = sigmoid(Wi * z + bi)
        ins[t] = sigmoid(Wi @ zs[t] + bi)
        # compute the candidate memory
        # c_ = tanh(Wc * z + bc)
        c_s[t] = cp.tanh(Wc @ zs[t] + bc)

        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        # c_t = f * c_(t-1) + i * c_
        cs[t] = cp.multiply(fs[t],cs[t-1]) + cp.multiply(ins[t],c_s[t])

        # output gate
        #o = sigmoid(Wo * z + bo)
        os[t] = sigmoid(Wo @ zs[t] + bo)
        hs[t] = cp.multiply(os[t],cp.tanh(cs[t]))
        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars
        # softmax for probabilities for next chars
        ys[t] = Why @ hs[t] + by
        ps[t] = softmax(ys[t])
        # label
        ls[t] = np.zeros((vocab_size, batch_size))
        targets = cp.asnumpy(targets)
        #  ls[t] = np.asarray(ls[t])
        for b in range(batch_size):
            ls[t][targets[t][b]][b] = 1
        targets = cp.asarray(targets)
        ls[t] = cp.asarray(ls[t])
        # cross-entropy loss
        loss_t = cp.sum(-cp.log(ps[t]) * ls[t])
        loss += loss_t
        # loss += -np.log(ps[t][targets[t],0])

    # activations = ()
    activations = (xs, wes, hs, ys, ps, cs, zs, ins, c_s, ls, os, fs)
    memory = (hs[input_length - 1], cs[input_length -1])#(hs[-1], cs[-1])

    return loss, activations, memory


def backward(activations, clipping=True, scale=True):
    xs, wes, hs, ys, ps, cs, zs, ins, c_s, ls, os, fs = activations

    # backward pass: compute gradients going backwards
    # Here we allocate memory for the gradients
    dWex, dWhy = cp.zeros_like(Wex), cp.zeros_like(Why)
    dby = cp.zeros_like(by)
    dWf, dWi, dWc, dWo = cp.zeros_like(Wf), cp.zeros_like(Wi), cp.zeros_like(Wc), cp.zeros_like(Wo)
    dbf, dbi, dbc, dbo = cp.zeros_like(bf), cp.zeros_like(bi), cp.zeros_like(bc), cp.zeros_like(bo)

    dhnext = cp.zeros_like(hs[0])
    dcnext = cp.zeros_like(cs[0])

    bsz = dhnext.shape[-1]

    input_length = len(xs)

    # back propagation through time starts here
    for t in reversed(range(input_length)):
        # computing the gradients here
        # first, we need to compute the gradients of the variable closest to the loss function,
        # which is the softmax output p
        # but here I skip it directly to the gradients of the unnormalized scores o because
        # basically dL / do = p - y
        dy = ps[t] - ls[t]

        #if scale:
           # dy = dy / bsz

        # the gradients w.r.t to the weights and the bias that were used to create o[t]
        dWhy += dy @ hs[t].T
        dby += cp.sum(dy, axis=-1, keepdims=True)

        # because h is connected to both o and the next h, we sum the gradients up
        dh = Why.T @ dy + dhnext

        #gradient for os[t] in hs[t] = os[t] * tanh(cs[t])
        dho = cp.tanh(cs[t]) * dh
        dho = dsigmoid(os[t]) * dho

        #gradient for cs in hs[t] = os[t] * tanh(cs[t])
        dc = dsigmoid(cp.tanh(cs[t])) * os[t] * dh#os[t]*(1 - cs[t] * cs[t]) * dh
        dc = dc + dcnext

        #gradient for f in c_t = f * c_(t-1) + i * c_
        dhf = cs[t-1] * dc
        dhf = dsigmoid(fs[t]) * dhf

        #gradient for i in c_t = f * c_(t-1) + i * c_
        dhi = c_s[t] * dc
        dhi = dsigmoid(ins[t]) * dhi

        #gradient for c_ in c_t = f * c_(t-1) + i * c_
        dhc = ins[t] * dc
        dhc = dtanh(c_s[t]) * dhc

        dWf += dhf @ zs[t].T
        dbf += cp.sum(dhf, axis=-1, keepdims=True)
        dxf = Wf.T @ dhf

        dWi += dhi @ zs[t].T
        dbi += cp.sum(dhi, axis=-1, keepdims=True)
        dxi = Wi.T @ dhi

        dWo += dho @ zs[t].T
        dbo += cp.sum(dho, axis=-1, keepdims=True)
        dxo = Wo.T @ dho

        dWc += dhc @ zs[t].T
        dbc += cp.sum(dhc, axis=-1, keepdims=True)
        dxc = Wc.T @ dhc

        dx = dxo + dxc + dxi + dxf
        dWex += dx[hidden_size:,:] @ xs[t].T

        dhnext = dx[:hidden_size,:]#hs[t-1]
        dcnext = fs[t] * dc


    # clip to mitigate exploding gradients
    if clipping:
        for dparam in [dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby]:
            cp.clip(dparam, -5, 5, out=dparam)

    gradients = (dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby)

    return gradients


def sample(memory, seed_ix, n):
    """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
    h, c = memory
    x = cp.zeros((vocab_size, 1))
    #seed_ix = cp.asnumpy(seed_ix)
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        
        wes = Wex @ x

        z = cp.vstack((h, wes))

        f = sigmoid(Wf @ z + bf)

        ins = sigmoid(Wi @ z + bi)

        c_ = cp.tanh(Wc @ z + bc)

        cs = cp.multiply(f,c) + cp.multiply(ins,c_)

        o = sigmoid(Wo @ z + bo)
        h = cp.multiply(o,cp.tanh(cs))

        y = Why @ h + by
        p = softmax(y)
        # forward pass again, but we do not have to store the activations now
        #loss, activations, memory = forward(inputs, targets, (hprev, nprev))
        p = cp.exp(y) / cp.sum(cp.exp(y))
        #p = cp.asnumpy(p)
        ix = cp.random.choice(a=range(vocab_size),size=1, p=p.ravel()).item()

        index = ix
        x = cp.zeros((vocab_size, 1))
        x[index] = 1
        ixes.append(index)
    return ixes


if option == 'train':

    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = cp.zeros_like(Wex), cp.zeros_like(Why)
    mby = cp.zeros_like(by)

    mWf, mWi, mWo, mWc = cp.zeros_like(Wf), cp.zeros_like(Wi), cp.zeros_like(Wo), cp.zeros_like(Wc)
    mbf, mbi, mbo, mbc = cp.zeros_like(bf), cp.zeros_like(bi), cp.zeros_like(bo), cp.zeros_like(bc)

    smooth_loss = -cp.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

    data_length = cut_stream.shape[1]

    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + seq_length + 1 >= data_length or n == 0:
            hprev = cp.zeros((hidden_size, batch_size))  # reset RNN memory
            cprev = cp.zeros((hidden_size, batch_size))
            p = 0  # go from start of data

        inputs = cut_stream[:, p:p + seq_length].T
        targets = cut_stream[:, p + 1:p + 1 + seq_length].T

        # sample from the model now and then
        if n % 200 == 0:
            h_zero = cp.zeros((hidden_size, 1))  # reset RNN memory
            c_zero = cp.zeros((hidden_size, 1))
            sample_ix = sample((h_zero, c_zero), inputs[0][0], 2000)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('----\n %s \n----' % (txt,))

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = forward(inputs, targets, (hprev, cprev))
        hprev, cprev = memory
        gradients = backward(activations)

        dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients
        smooth_loss = smooth_loss * 0.999 + loss/batch_size * 0.001
        if n % 20 == 0:
            print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                      [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                      [mWf, mWi, mWo, mWc, mbf, mbi, mbo, mbc, mWex, mWhy, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / cp.sqrt(mem + 1e-8)  # adagrad update

        p += seq_length  # move data pointer
        n += 1  # iteration counter
        n_updates += 1
        if n_updates >= max_updates:
            break

elif option == 'gradcheck':

    data_length = cut_stream.shape[1]

    p = 0
    # inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    # targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]
    inputs = cut_stream[:, p:p + seq_length].T
    targets = cut_stream[:, p + 1:p + 1 + seq_length].T

    delta = 0.0001

    hprev = cp.zeros((hidden_size, batch_size))
    cprev = cp.zeros((hidden_size, batch_size))

    memory = (hprev, cprev)

    loss, activations, hprev = forward(inputs, targets, memory)
    gradients = backward(activations, clipping=False)
    dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients

    for weight, grad, name in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                  [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                  ['Wf', 'Wi', 'Wo', 'Wc', 'bf', 'bi', 'bo', 'bc', 'Wex', 'Why', 'by']):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert (weight.shape == grad.shape), str_

        print(name)
        countidx = 0
        gradnumsum = 0
        gradanasum = 0
        relerrorsum = 0
        erroridx = []

        for i in range(weight.size):

            # evaluate cost at [x + delta] and [x - delta]
            #weight = cp.asnumpy(weight)
            #wight_copy = weight
            w = weight.flatten().take(indices=i).item()
            weight.put(indices=i,values = w + delta)
            loss_positive, _, _ = forward(inputs, targets, memory)
            weight.put(indices=i,values = w - delta)
            loss_negative, _, _ = forward(inputs, targets, memory)
            weight.put(indices=i,values = w)  # reset old value for this parameter
            #weight = cp.asarray(weight)
            # fetch both numerical and analytic gradient
            #grad = cp.asnumpy(grad)
            grad_analytic = grad.flatten().take(indices=i).item()#grad.flat[i]
            #grad = cp.asarray(grad)
            grad_numerical = (loss_positive - loss_negative) / (2 * delta)
            gradnumsum += grad_numerical
            gradanasum += grad_analytic
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            if rel_error is None:
                rel_error = 0.
            relerrorsum += rel_error

            if rel_error > 0.001:
                print ('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
                countidx += 1
                erroridx.append(i)

        print('For %s found %i bad gradients; with %i total parameters in the vector/matrix!' % (
            name, countidx, weight.size))
        print(' Average numerical grad: %0.9f \n Average analytical grad: %0.9f \n Average relative grad: %0.9f' % (
            gradnumsum / float(weight.size), gradanasum / float(weight.size), relerrorsum / float(weight.size)))
        print(' Indizes at which analytical gradient does not match numerical:', erroridx)
