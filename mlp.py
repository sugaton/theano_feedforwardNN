import theano
import math
import numpy
from collections import OrderedDict
from collections import namedtuple
FLOAT = theano.config.floatX


class mlp(object):
    def __init__(self, IL=100, HL=20, OL=5, iter=2, alfa=0.08, hid_act="tanh", out_act="linear", initupper=0.01):
        # set argments to self.*
        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        #
        self.alfa = numpy.float32(alfa)
        self._params = {}
        self.setparameters(self._parameters())

        self.inp, self.ans = self.setinput_answer()
        self.input = self._makeinput()
        self.answer = self._makeanswer()

        self.setnetwork()
        E = self._Cost()
        gradients = theano.grad(E, self._params.values())
        upd = [(p, p + self.alfa * g) for p, g in zip(self._params.values(), gradients)]
        self.learn_with_cost = theano.function([self.inp, self.ans], E, updates=upd)
        self.system_out = theano.function([self.inp], self.out)

    def setparameters(self, param_infos):
        def setparam(name="", size1=0, size2=0, initupper=0.1, params={}):
            if size2 is None:
                arr = numpy.random.uniform(-initupper, initupper, (size1, )).astype(FLOAT)
            else:
                arr = numpy.random.uniform(-initupper, initupper, (size1, size2)).astype(FLOAT)
            params[name] = theano.shared(arr, name=name)

        for pi in param_infos:
            setparam(name=pi.name, size1=pi.size1, size2=pi.size2, initupper=pi.initupper, params=self._params)

    def _parameters(self):
        paramInfo = namedtuple("paramInfo", "name size1 size2 initupper")
        return [paramInfo("W1", self.HL, self.IL, self.initupper),
                paramInfo("W2", self.OL, self.HL, self.initupper)]

    def setinput_answer(self):
        return (theano.tensor.vector("inp"), theano.tensor.vector("ans"))

    def _makeinput(self):
        return self.inp

    def _makeanswer(self):
        return self.ans

    @staticmethod
    def getfunc(st):
        dic = {"tanh": theano.tensor.tanh, "linear": (lambda x: x), "sigmoid": theano.tensor.nnet.sigmoid}
        return dic[st]

    def setnetwork(self):
        hA = self.getfunc(self.hid_act)
        oA = self.getfunc(self.out_act)
        self.hid = hA(theano.dot(self._params["W1"], self.input))
        self.out = oA(theano.dot(self._params["W2"], self.hid))

    def Cost(self):
        diff = self.answer - self.out
        error = theano.dot(diff.T, diff)/2
        return error

    def training(self, dataset):
        for i in range(self.iter):
            costs = []
            for ans, inp in dataset:
                costs.append(self.learn_with_cost(inp, ans))
            print("average error:", sum(costs) / len(dataset))

    def test_(self, dataset):
        sum = 0
        for ans, inp in dataset:
            out = self.system_out(inp)
            print ans, out
            sum += sum - out
        print("average error: ", sum * 1.0 / len(dataset))

    def write_(self, filename):
        dic = {}
        for key, item in self._params.items():
            arr = item.get_value()
            dic[key] = arr
        numpy.savez(filename, **dic)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        upd = []
        for k, arr in value.items():
            p = self._params[k]
            upd.append((p, theano.tensor.set_subtensor(p[:], arr)))
        f = theano.function([], updates=upd)
        f()
