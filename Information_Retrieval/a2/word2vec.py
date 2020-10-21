#!/usr/bin/env python

import numpy as np
import random


from Information_Retrieval.a2.utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from Information_Retrieval.a2.utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)
    #numpy에 구현된 exponential 함수를 쓰면 간단하게 나타낼 수 있다.
    s = 1./(np.exp(-x)+1.)

    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)
    #y_hat.shape = ( # of words in vocab, )
    y_hat = softmax(np.dot(outsideVectors, centerWordVec))
    #loss 값 계산
    loss = -np.log(y_hat)
    # 문제지 (b)에서 U(y_hat - y)를 구현했었다..
    # 또한 (c)에서 y_hat - y꼴이 등장하므로 계산해놓는다.
    # subtraction = y_hat - y 이다. u_o에만 1(y)값을 뺴줘야 제대로 계산이 된다. (w = 0 인 case)
    # 그 이외에는  y_hat 이기 때문에 indexing으로 해당 element만 -y(-1)을 취한다.
    subtraction = y_hat
    subtraction[outsideWordIdx] -= 1
    #  aJ / av_c는 (U.T(y_hat -y))
    gradCenterVec = outsideVectors.T.dot(subtraction)
    #   aJ / aU의 계산, 실제로 (c)문제를 이용하면, matrix의 형태는 diagonal 값은 (y_hat -y)v_c.T 을 사용,
    #   그 외의 값은 y_hat v_c.T 를 사용한다.
    #   matrix의 형태는 (# of words in vocab, word vec length) 이다. np.dot을 잘 이용해 계산한다.
    gradOutsideVecs = subtraction[:, np.newaxis].dot(np.array([centerWordVec]))

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)

    ### Please use your implementation of sigmoid in here.

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)
    #   word2Ind dictionary로부터 현재 Vc의 index값을 얻는다
    centerWordIdx = word2Ind[currentCenterWord]
    # index 값을 이용해 word vector집합에서 현재 centerword를 얻는다.
    centerWordVec = centerWordVectors[centerWordIdx]
    # outsideWords에서 outside word(string)추출, word2Ind를 이용해 해당 word에 대한 index저장
    outsideWordIndices = [word2Ind[i] for i in outsideWords]

    # J_skipgram 의 v_c, U 에 대한 편미분을 구하고, loss를 구한다.
    # 반복문은 -m <= j <= m, j != 0 이므로 2m번 반복한다.
    for outsideWordIdx in outsideWordIndices:
        one_loss, one_gradCenter, one_gradOutside = \
            word2vecLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset)

        loss += one_loss
        gradCenterVecs[centerWordIdx] += one_gradCenter
        gradOutsideVectors += one_gradOutside

    ### END YOUR CODE
    
    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    # We will NOT test negSamplingLossAndGradient

    # print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
    #     dummy_vectors, "negSamplingLossAndGradient Gradient")

    # grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)


if __name__ == "__main__":
    test_word2vec()

