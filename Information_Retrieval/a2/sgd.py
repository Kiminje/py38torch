#!/usr/bin/env python

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 5000

import pickle
import glob
import random
import numpy as np
import os.path as op

def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        params_file = "saved_params_%d.npy" % st
        state_file = "saved_state_%d.pickle" % st
        params = np.load(params_file)
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None


def save_params(iter, params):
    params_file = "saved_params_%d.npy" % iter
    np.save(params_file, params)
    with open("saved_state_%d.pickle" % iter, "wb") as f:
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=False, useSaved=False,
        PRINT_EVERY=10):
    """ Stochastic Gradient Descent

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a loss and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies how many iterations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """

    # Anneal learning rate every several iterations
    # -> parameter들이 이미 저장되어있다면, 매 iteration마다 annealing한다.
    # 예를 들면 20000번 반복시에는 step size가 절반이 된다.
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    exploss = None

    for iter in range(start_iter + 1, iterations + 1):
        # You might want to print the progress every few iterations.

        loss = None
        ### YOUR CODE HERE (~2 lines)
        # run.py 의 sgd 를 참조하여 f(x)를 찾는다.
        # f = word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,naiveSoftmaxLossAndGradient) 이다.
        # f는 batch size = 50으로 하여 skip gram 에 대해 naiveSoftmaxLossAndGradient를 수행한다.
        # grad는 각 인자에 대한 편미분 행렬을 더한 gradient matrix.
        loss, grad = f(x)
        #step size만큼 parameter들을 update한다.
        x -= step * grad
        ### END YOUR CODE

        #cost가 갑자기 커지는 것을 방지하기위해?
        x = postprocessing(x)
        if iter % PRINT_EVERY == 0:

            if not np.any(exploss):

                exploss = loss
            else:
                exploss = .95 * exploss + .05 * loss
            print("iter %d: %f" % (iter, exploss))
        # 5000번 iteration 마다 parameter저장
        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)
        # 20000번 학습마 step size를 절반으로 줄인다.
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x


def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 1 result:", t1)
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print("test 2 result:", t2)
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 3 result:", t3)
    assert abs(t3) <= 1e-6

    print("-" * 40)
    print("ALL TESTS PASSED")
    print("-" * 40)


if __name__ == "__main__":
    sanity_check()
