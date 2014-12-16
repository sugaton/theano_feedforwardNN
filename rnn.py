import theano
import theano.tensor as T
import numpy
from collections import namedtuple
import mlp
FLOAT = theano.config.floatX



# batch_version
class rnn_batch(mlp.mlp):
    class one_batch:
        def __init__(self, idx, parent):
            self.P = parent
            sent = parent.getdata(idx)

        def _setnet(self, sent):
            def recurrence(x_t, h_tm1):
                h_t = T.nnet.sigmoid(T.dot(x_t, self.P.params["wx"]) +
                                     T.dot(h_tm1, self.P.params["wh"]) +
                                     self.P.params["bh"])
                s_t = T.nnet.softmax(T.dot(h_t, self.P.params["w"]) + self.P.params["b"])
                return [h_t, s_t]
            x_ = self.P.params["emb"][sent]
            label = T.set_subtensor(T.zeros_like(x)[:-1], x_[1:])
            [h, s], _ = theano.scan(fn=recurrence,
                                    sequences=x_,
                                    outputs_info=[self.P.params["h0"], None])
            self.cost = -T.mean(T.log(s[T.arange(label.shape[0]), label]))
            self.y_pred = T.argmax(s, axis=1)

    def __init__(self,
                 emb_d=50,
                 wsize=0,
                 N=5,
                 buffersize=100000,
                 batchsize=20,
                 **args):
        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        args["IL"] = emb_d * N
        self._idx = T.lvector()
        self._data_idx = theano.shared(numpy.zeros((buffersize/2, 2), dtype="int64"))
        self._data = theano.shared(numpy.zeros(buffersize, dtype="int64"))
        super(rnn, self).__init__(**arg)

    def _parameters(self):
        paramInfo = namedtuple("paramInfo", "name size1 size2 initupper")
        return [paramInfo(name="wx", size1=self.HL, size2=self.IL, initupper=self.initupper),
                paramInfo(name="wh", size1=self.HL, size2=self.HL, initupper=self.initupper),
                paramInfo(name="bh", size1=self.HL, size2=None, initupper=0),
                paramInfo(name="w", size1=self.OL, size2=self.HL, initupper=self.initupper),
                paramInfo(name="b", size1=self.OL, size2=None, initupper=0),
                paramInfo(name="h0", size1=self.HL, size2=None, initupper=0),
                paramInfo(name="emb", size1=self.wsize, size2=self.emb_d, initupper=self.initupper)
                ]

    def setnetwork(self):
        sumcost = []
        nets = range(self.batchsize)
        for i in self.batchsize:
            nets[i] = self.one_batch(self._idx[i], self)
            sumcost.append(nets[i].cost)
        self._cost = sum(sumcost)

    def _Cost(self):
        return self._cost

    def setdata(idxs, loaded):
        _idxs = numpy.array(idxs, dtype="int64")
        data = numpy.array(loaded, dtype="int64")
        T.set_subtensor(self._data_idx[len(_idxs)])

    def load_map(self, func, dataset):
        def setdata(sumload=0):
            loaded = []
            idxs = []
            while len(loaded) < self.buffersize:
                data = dataset[sumload]
                idxs.append((len(loaded), len(data)))
                loaded.extend(data)
                self.setdata(idxs, loaded)
                sumload += 1
            return sumload

        if sum([len(sent) for sent in dataset]) < self.buffersize:
            self.setdata()
            func()
        else:
            while True:

if __name__ == '__main__':
    main()
