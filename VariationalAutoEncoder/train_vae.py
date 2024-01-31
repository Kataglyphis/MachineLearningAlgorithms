import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import sys
import pickle
import scipy.io


debug = False
img_rows = 28
img_cols = 20
ff = scipy.io.loadmat('data/frey_rawface.mat')
ff = ff["ff"].T.reshape((-1, 1, img_rows, img_cols))
ff = ff.astype('float32') / 255.
print(ff.shape)
ff = cp.asarray(ff)

n_samples = ff.shape[0]


# Number of parameters
input_size = 560
hidden_size = 128
latent_size = 16
std = 0.02
learning_rate = 0.02
loss_function = 'bce'  # mse or bce
beta1=0.9
beta2=0.999


def get_minibatch(batch_size, idx=0, indices=None):
    start_idx = batch_size * idx
    end_idx = min(start_idx + batch_size, n_samples)

    if indices is None:
        sample_b = ff[start_idx:end_idx]
    else:
        idx = indices[start_idx:end_idx]
        sample_b = ff[idx]

    #sample_b = np.resize(sample_b, (batch_size, 560))
    sample_b = cp.asnumpy(sample_b)
    sample_b = np.resize(sample_b, (batch_size, 560))
    sample_b = cp.asarray(sample_b)

    sample_b = cp.transpose(sample_b, (1, 0))

    return sample_b


def sigmoid(x):
    return 1 / (1 + cp.exp(-x))


# The derivative of the sigmoid function
def dsigmoid(y, x=None):
    return y * (1 - y)

# The tanh function
def tanh(x):
    return cp.tanh(x)


# The derivative of the tanh function
def dtanh(y, x=None):
    return 1 - y * y


def softplus(x):
    return cp.log(1 + cp.exp(x))


def dsoftplus(y, x=None):
    assert x is not None
    return sigmoid(x)


def sample_unit_gaussian(latent_size):
    return cp.random.standard_normal(size=(latent_size))


# (Inplace) relu function
def relu(x):
    x[x < 0] = 0

    return x


# Gradient of Relu
def drelu(y, x=None):
    return 1. * (y > 0)


# Initialization was done not exactly according to Kingma et al. 2014 (he used Gaussian).
# input to hidden weight
Wi = cp.random.uniform(-std, std, size=(hidden_size, input_size))
Bi = cp.random.uniform(-std, std, size=(hidden_size, 1))

# encoding weight (hidden to code mean)
Wm = cp.random.uniform(-std, std, size=(latent_size, hidden_size))  # hidden to mean
Bm = cp.random.uniform(-std, std, size=(latent_size, 1))  # hidden to mean\

Wv = cp.random.uniform(-std, std, size=(latent_size, hidden_size))  # hidden to logvar
Bv = cp.random.uniform(-std, std, size=(latent_size, 1))  # hidden to logvar

# weight mapping code to hidden
Wd = cp.random.uniform(-std, std, size=(hidden_size, latent_size))
Bd = cp.random.uniform(-std, std, size=(hidden_size, 1))

# decoded hidden to output
Wo = cp.random.uniform(-std, std, size=(input_size, hidden_size))
Bo = cp.random.uniform(-std, std, size=(input_size, 1))


def forward(input, epsilon, gradcheck=False):
    if debug:
        print("input shape:", input.shape)
    # YOUR FORWARD PASS FROM HERE

    if input.ndim == 1:
        input = cp.expand_dims(input, axis=1)

    batch_size = input.shape[-1]
    # (1) linear
    # H = W_i \times input + Bi
    H = cp.dot(Wi, input) + Bi

    # (2) ReLU
    # H = ReLU(H)
    H = relu(H)

    if grad_check == False:
        epsilon = sample_unit_gaussian(latent_size=(latent_size,batch_size))
    # (3) h > mu
    # Estimate the means of the latent distributions
    # mean = Wm \times H + Bm
    mean = cp.dot(Wm, H) + Bm

    # (4) h > log var
    # Estimate the (diagonal) variances of the latent distributions
    # logvar = Wv \times H + Bv
    logvar = cp.dot(Wv, H) + Bv

    # (5) sample the random variable z from means and variances (refer to the "reparameterization trick" to do this)
    # variable (z) which is generated the distribution N(mean, std^2),
    #  because std = sqrt(var)!!!
    z = mean + cp.multiply(cp.exp(logvar/2),epsilon)

    # (6) decode z
    # D = Wd \times z + Bd
    D = cp.dot(Wd, z) + Bd

    # (7) relu
    # D = ReLU(D)
    D = relu(D)

    # (8) dec to output
    # output = Wo \times D + Bo
    output = cp.dot(Wo, D) + Bo

    # # (9) dec to p(x)
    # and (10) reconstruction loss function (same as the
    P = decode(z)

    if loss_function == 'bce':
        # BCE Loss
        rec_loss = -cp.sum(cp.multiply(input, cp.log(P)) + cp.multiply(1 - input, cp.log(1 - P))).take(indices=0).item()

    elif loss_function == 'mse':
        rec_loss = cp.sum(0.5 * (P - input) ** 2).take(indices=0).item()
        # MSE Loss

    # variational loss with KL Divergence between P(z|x) and U(0, 1)

    #kl_div_loss = - 0.5 * (1 + logvar - mean^2 - e^logvar)
    kl_div_loss = cp.sum(-0.5 * (1 + logvar - pow(mean,2) - cp.exp(logvar))).take(indices=0).item()

    # your loss is the combination of
    #loss = rec_loss + kl_div_loss
    loss = rec_loss + kl_div_loss

    # Store the activations for the backward pass
    # activations = ( ... )
    activations = (epsilon, H, mean, logvar, z, D, output, P, rec_loss, kl_div_loss)

    return loss, kl_div_loss, activations


def decode(z):

    # basically the decoding part in the forward pass: mapping z to p
    dec = cp.dot(Wd, z) + Bd
    dec = relu(dec)
    output = cp.dot(Wo, dec) + Bo
    if loss_function == 'bce':
        p = sigmoid(output)

    elif loss_function == 'mse':
        p = output
    # o = W_d \times z + B_d

    # p = sigmoid(o) if bce or o if mse

    return p


def backward(input, activations, scale=True, alpha=1.0):
    # allocating the gradients for the weight matrice
    dWi = cp.zeros_like(Wi)
    dWm = cp.zeros_like(Wm)
    dWv = cp.zeros_like(Wv)
    dWd = cp.zeros_like(Wd)
    dWo = cp.zeros_like(Wo)
    dBi = cp.zeros_like(Bi)
    dBm = cp.zeros_like(Bm)
    dBv = cp.zeros_like(Bv)
    dBd = cp.zeros_like(Bd)
    dBo = cp.zeros_like(Bo)

    batch_size = input.shape[-1]
    scaler = batch_size if scale else 1

    eps, h, mean, logvar, z, dec, output, p, _, _ = activations

    # Perform your BACKWARD PASS (similar to the auto-encoder code)

    if loss_function == 'mse':

        dl_dp = p - input
        
        if scale:
            dl_dp = dl_dp / batch_size

        dl_doutput = dl_dp

    elif loss_function == 'bce':

        dl_dp = (-1 * (input / p - (1 - input) / (1 - p)))

        if scale:
            dl_dp = dl_dp / batch_size
            
        dl_doutput = cp.multiply(dl_dp, dsigmoid(p))

    # backprop from (8) through fully-connected
    dl_ddec = cp.dot(Wo.T, dl_doutput)
    dWo += cp.dot(dl_doutput, dec.T)
    if batch_size == 1:
        dBo += dl_doutput
    else:
        dBo += cp.sum(dl_doutput, axis=-1, keepdims=True) 

    # backprop from (7) through ReLU
    dl_ddec = cp.multiply(drelu(dec), dl_ddec)

    # backprop from (6) through fully-connected
    dl_dz = cp.dot(Wd.T, dl_ddec)
    dWd += cp.dot(dl_ddec, z.T)

    if batch_size == 1:
        dBd += dl_ddec
    else:
        dBd += cp.sum(dl_ddec, axis=-1, keepdims=True)

    # through the mu branch
    #dz_dmean = 1
    dkl_dmean = mean
    #sacle kl loss accordingly to normal loss
    if scale:
        dkl_dmean = dkl_dmean / batch_size

    dl_dmean = dl_dz

    dl_dmean = dl_dmean + dkl_dmean
    # through fully connected of mu branch
    dl_dmean_h = cp.dot(Wm.T, dl_dmean)

    dWm += cp.dot(dl_dmean, h.T)

    if batch_size == 1:
        dBm += dl_dmean
    else:
        dBm += cp.sum(dl_dmean, axis=-1, keepdims=True)

    # through the sigma branch
    dz_dsigma = cp.multiply(eps, cp.multiply(0.5, cp.exp(logvar/2)))
    dkl_dsigma = -0.5 + cp.multiply(0.5, cp.exp(logvar))
    #sacle appropriate to normal loss
    if scale:
        dkl_dsigma = dkl_dsigma / batch_size


    dl_dsigma = cp.multiply(dl_dz, dz_dsigma)
    dl_dsigma = dl_dsigma + dkl_dsigma

    #trough fully connected of sigma branch
    dl_dsigma_h = cp.dot(Wv.T, dl_dsigma)
    
    dWv += cp.dot(dl_dsigma, h.T)

    if batch_size == 1:
        dBv += dl_dsigma
    else:
        dBv += cp.sum(dl_dsigma, axis=-1, keepdims=True)
    

    #bring them together
    dl_dh = dl_dsigma_h + dl_dmean_h
    dl_dh = np.multiply(drelu(h), dl_dh)

    dl_dinput = cp.dot(Wi.T, dl_dh)
    dWi += cp.dot(dl_dh, input.T)

    if batch_size == 1:
        dBi += dl_dh
    else:
        dBi += cp.sum(dl_dh, axis=-1, keepdims=True)
    
    # 1st Note:
    # When performing the BW Pass for mean and logvar, note that they should have 2 different terms
    # One coming from the reconstruction loss, and backprop-ed through the hidden layer z
    # One coming from the KL divergence loss
    # So you should sum them up to have the correct gradient

    # 2nd Note:
    # The z is a sample from the distribution P(z|x), this is backprop-able based on the reparameterization trick
    # In order to do that, one random variable must stay the same between forward and backward passes.

    # The rest of the backward pass should be the same as the AE

    gradients = (dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo)

    return gradients


def train():
    # Momentums for adagrad
    mWi, mWm, mWv, mWd, mWo = cp.zeros_like(Wi), cp.zeros_like(Wm), cp.zeros_like(Wv),\
                              cp.zeros_like(Wd), cp.zeros_like(Wo)

    mBi, mBm, mBv, mBd, mBo = cp.zeros_like(Bi), cp.zeros_like(Bm), cp.zeros_like(Bv), \
                              cp.zeros_like(Bd), cp.zeros_like(Bo)

    # Velocities for Adam
    vWi, vWm, vWv, vWd, vWo = cp.zeros_like(Wi), cp.zeros_like(Wm), cp.zeros_like(Wv), \
                              cp.zeros_like(Wd), cp.zeros_like(Wo)

    vBi, vBm, vBv, vBd, vBo = cp.zeros_like(Bi), cp.zeros_like(Bm), cp.zeros_like(Bv), \
                              cp.zeros_like(Bd), cp.zeros_like(Bo)

    def save_weights():

        print("Saving weights to %s and moments to %s" % ('weights.vae.pkl', 'momentums.vae.pkl'))

        weights = (Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo)
        with open('models/weights.vae.pkl', 'wb') as output:
            pickle.dump(weights, output, pickle.HIGHEST_PROTOCOL)

        momentums = (mWi, mWm, mWv, mWd, mWo, mBi, mBm, mBv, mBd, mBo)
        with open('models/momentums.vae.pkl', 'wb') as output:
            pickle.dump(momentums, output, pickle.HIGHEST_PROTOCOL)

        return

    batch_size = 128
    n_epoch = 100000

    save_every = 2000

    # first we have to shuffle the data
    n_samples = ff.shape[0]
    indices = cp.arange(n_samples)
    total_loss = 0
    total_kl_loss = 0
    total_pixels = 0
    total_samples = 0
    count = 0
    alpha = 0.0

    n_minibatch = math.ceil(n_samples / batch_size)
    for epoch in range(n_epoch):

        rand_indices = cp.random.permutation(indices)

        for i in range(n_minibatch):

            x_i = get_minibatch(batch_size, i, rand_indices)
            bsz = x_i.shape[-1]
            epsilon = sample_unit_gaussian(latent_size=(latent_size,batch_size))

            loss, kl_div_loss, acts = forward(x_i, epsilon, False)
            _, _, _, _, z, _, _, _, rec_loss, kl_loss = acts
            # lol I computed kl_div again here

            total_loss += rec_loss
            total_kl_loss += kl_loss
            total_pixels += bsz * 560

            gradients = backward(x_i, acts, alpha=alpha)

            dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo= gradients

            count += 1

            # perform parameter update with Adagrad
            # perform parameter update with Adam
            for param, dparam, mem, velo in zip([Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo],
                                            [dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo],
                                            [mWi, mWm, mWv, mWd, mWo, mBi, mBm, mBv, mBd, mBo],
                                            [vWi, vWm, vWv, vWd, vWo, vBi, vBm, vBv, vBd, vBo]):
                mem += dparam * dparam
                param += -learning_rate * dparam / cp.sqrt(mem + 1e-8)  # adagrad update

                # Adam update
                # bias_correction1 = 1 - beta1 ** count
                # bias_correction2 = 1 - beta2 ** count
                #
                # mem = mem * beta1 + (1 - beta1) * dparam
                # velo = velo * beta2 + (1 - beta2) * dparam * dparam
                # denom = np.sqrt(velo) / math.sqrt(bias_correction2) + 1e-9
                # step_size = learning_rate / bias_correction1
                #
                # param += -step_size * mem / denom

            total_samples += bsz  # lol it can be total_pixels / 560

            if count % 50 == 0:
                avg_loss = total_loss / total_pixels
                avg_kl = total_kl_loss / total_samples
                print("Epoch %d Iteration %d Updates %d Loss per pixel %0.6f avg KLDIV %0.6f " %
                      (epoch, i, count, avg_loss, avg_kl))

            # save weights to file every 500 updates so we can load to visualize later
            if count % 500 == 0:
                save_weights()

    return


def grad_check():
    batch_size = 8
    delta = 0.0001

    x = get_minibatch(batch_size)

    actual_bsz = x.shape[-1]  # because x can be the last batch in the dataset which has bsz < 8

    epsilon = sample_unit_gaussian(latent_size=(latent_size,batch_size))
    loss, kl_div_loss, acts = forward(x, epsilon, True)

    gradients = backward(x, acts, scale=False)

    dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo = gradients

    for weight, grad, name in zip([Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo],
                                  [dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo],
                                  ['Wi', 'Wm', 'Wv', 'Wd', 'Wo', 'Bi', 'Bm', 'Bv', 'Bd', 'Bo']):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert (weight.shape == grad.shape), str_

        print("Checking grads for weights %s ..." % name)
        n_warnings = 0

        for i in range(weight.size):

            #epsilon = sample_unit_gaussian(latent_size=(latent_size,batch_size))
            #w = weight.flat[i]
            w = weight.flatten().take(indices=i).item()

            #weight.flat[i] = w + delta
            weight.put(indices=i,values = w + delta)
            loss_positive, _, _ = forward(x, epsilon, True)

            #weight.flat[i] = w - delta
            weight.put(indices=i,values = w - delta)
            loss_negative, _, _ = forward(x, epsilon, True)

            weight.put(indices=i,values = w)
            #weight.flat[i] = w  # reset old value for this parameter

            
            grad_analytic = grad.flatten().take(indices=i).item()#grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / (2 * delta)

            if (abs(grad_numerical + grad_analytic) != 0):
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            else:
                rel_error = 0

            if rel_error > 0.001:
                n_warnings += 1
                print('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))

            #print('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
            
        print("%d gradient mismatch warnings found. " % n_warnings)

    return


def eval():
    while True:
        # hard coded batch size, but is not important at this step
        epsilon = sample_unit_gaussian(latent_size=(latent_size,8))
        # read weights from file
        cmd = input("Enter an image number:  ")

        img_idx = int(cmd)

        if img_idx < 0:
            exit()

        fig = plt.figure(figsize=(2, 2))
        n_samples = 1

        sample_ = ff[img_idx]
        org_img = sample_ * 255
        sample_ = cp.reshape(sample_, (1, 560)).T#np.resize(sample_, (1, 560)).T

        sample_ = sample_.flatten()
        loss,kl_div_loss, act = forward(sample_, epsilon, False)

        _,h,_,_, z, dec, output, p,_,_ = act
        # Here the sample_ is processed by the network to produce the reconstruction

        img = cp.sum(p, axis=-1)
        img = img / n_samples

        fig.add_subplot(1, 2, 1)
        org_img = cp.asnumpy(org_img)
        plt.imshow(org_img.reshape(28, 20), cmap='gray')

        fig.add_subplot(1, 2, 2)
        img = cp.asnumpy(img)
        plt.imshow(img.reshape(28, 20), cmap='gray')
        plt.show(block=True)

        print("Done")


def sample():
    while True:
        cmd = input("Press anything to continue:  ")

        z = cp.random.randn(latent_size)
        z = cp.expand_dims(z, 1)

        # The decode function should be implemented before this
        p = decode(z)
        img = p

        fig = plt.figure(figsize=(2, 2))
        # gs = gridspec.GridSpec(4, 4)
        # gs.update(wspace=0.05, hspace=0.05)

        img = cp.asnumpy(img)
        plt.imshow(img.reshape(28, 20), cmap='gray')
        # plt.title('reconstructed face %d' % 0)
        plt.show(block=True)


if len(sys.argv) != 2:
    print("Need an argument train or gradcheck or reconstruct")
    exit()

option = sys.argv[1]

if option == 'train':
    train()
elif option in ['grad_check', 'gradcheck']:
    grad_check()
elif option in ['eval', 'sample']:

    # read trained weights from file
    with open('models/weights.vae.pkl', "rb") as f:
        weights = pickle.load(f)

    Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo = weights

    if option == 'eval':
        eval()
    else:
        sample()
else:
    raise NotImplementedError
