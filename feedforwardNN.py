import theano
import numpy


class feedforwardNN():
    def __init__(self, IL=100, HL=20, OL=5):
        self.IL = IL
        self.HL = HL
        self.OL = OL
        self.inp = theano.tensor.dvector('inp')
        self.inputlayer = numpy.zeros(IL)
        self.hid = theano.tensor.dvector('hid')
        self.hiddenlayer = numpy.zeros(HL)
        self.out = theano.tensor.dvector('hid')
        self.outputlayer = numpy.zeros(OL)
        self.W1 = theano.tensor.dmatrix("w1")
        self.W2 = theano.tensor.dmatrix("w2")
        self.Connection1 = theano.function([self.inp], theano.dot(self.W1, self.inp))
        self.Connection2 = theano.function([self.hid], theano.dot(self.W2, self.hid))

    def forwardP(self, input):
        self.hiddenlayer = self.Connection1([input])
        self.outputlayer = self.Connection2([self.hiddenlayer])
