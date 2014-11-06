import theano
import math
import numpy
from collections import OrderedDict
FLOAT = theano.config.floatX


class feedforwardNN(object):
    def __init__(self, IL=100, HL=20, OL=5):
        self.IL = IL
        self.HL = HL
        self.OL = OL
        self.alfa = 0.008
        arr = numpy.random.randn(HL, IL) / numpy.sqrt(HL + IL)
        self.W1 = theano.shared(arr.astype(FLOAT), name='W1')
        arr = numpy.random.randn(OL, HL) / numpy.sqrt(HL + OL)
        self.W2 = theano.shared(arr.astype(FLOAT), name='W2')
        self.params = {"W1": self.W1, "W2": self.W2}

        self.inp = theano.tensor.dvector('inp')
        self.ans = theano.tensor.dvector('ans')
        self.hid = theano.tensor.tanh(theano.dot(self.W1, self.inp))
        self.out = theano.dot(self.W2, self.hid)
        self.E = self.Cost()
        self.gradients = self.set_grad()
        self.comp = theano.function([self.inp, self.ans], self.E, updates=self.gradients)

    def Cost(self):
        diff = self.ans - self.out
        error = theano.dot(diff.T, diff)/2
        return error

    def set_grad(self):
        ret = OrderedDict()
        ret[self.W1] = self.W1 + (self.alfa * theano.tensor.grad(self.E, self.W1)).astype(FLOAT)
        ret[self.W2] = self.W2 + (self.alfa * theano.tensor.grad(self.E, self.W2)).astype(FLOAT)
        return ret

    def forwardP(self, input, ans):
        return self.comp(input, ans)

    def training(self, dataset):
        for ans, inp in dataset:
            self.forwardP(inp, ans)

    def test(self, dataset):
        print numpy.mean([math.sqrt(2*self.forwardP(inp, ans)) for ans, inp in dataset])

    def write_(self, filename):
        dic = {}
        for key, item in self.params.items():
            arr = item.get_value()
            dic[key] = arr
        numpy.savez(filename, **dic)
