from mlp import mlp
import theano
from collections import namedtuple

FLOAT = theano.config.floatX
T = theano.tensor


class network(mlp):
    def __init__(self, char_d=50, N=5, wordsize=0, **args):
        args = locals()
        args.pop("self")
        args.pop("args")
        self.__dict__.update(args)
        super(network, self).__init__(**args)

    def _parameters(self):
        paramInfo = namedtuple("paramInfo", "name size1 size2")
        return [paramInfo("W1", self.HL, self.IL, self.initupper),
                paramInfo("b1", self.HL, None, 0),
                paramInfo("W2", self.OL, self.HL, self.initupper),
                paramInfo("b2", self.OL, 0),
                paramInfo("L", self.wordsize, self.char_d, self.initupper)]

    def setinput_answer(self):
        return (T.lvector("inp"), T.vector('ans'))

    def _makeinput(self):
        return T.reshape(self.params["L"][self.inp], (self.IL, ))

    def load_lookup(self, arr):
        v = self.params_v["C"]
        p = self.params["C"]
        setfunc = theano.function([v], updates=[(p, T.set_subtensor(p[:], v))])
        setfunc(arr)


if __name__ == '__main__':
    quit()
