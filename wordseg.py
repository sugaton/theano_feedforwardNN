import theano.tensor as T
import numpy
import theano
from collections import OrderedDict

FLOAT = theano.config.floatX


class segmenter(object):
    class network:
        def __init__(self, seg, idx, N=5):
            self.seg = seg
            self.R = self.get_rev_func()
            i, a = self.seg.getdata(idx)
            self.netS, self.tranS = self.setnet(i, a, N)

        @property
        def Scores(self):
            return (self.netS, self.tranS)

        @property
        def count(self):
            return self.count

        @property
        def viterbi_path(self):
            return self.viterbi_path

        def get_rev_func(self):
            """
            return matrix like bellow
            0 0 1
            0 1 0
            1 0 0
            """
            L = self.seg.params['C'].get_value().shape[1]
            arr = numpy.zeros((L, L))
            for i in range(L):
                arr[L-i-1] = 1
            return theano.shared(arr.astype(FLOAT), name="rev")

        def lookup(self, C, inp):
            result, updates = theano.scan(fn=lambda x: C[x],
                                          outputs_info=None,
                                          sequences=[inp])
            return result

        @staticmethod
        def slice(i, X, N=5):
            """
            # concatenate N embeddings
            """
            n = N / 2
            return T.reshape(X[i-n:i+n+1], (X.shape[0]*N, ))

        def makeinputs(self, N, inp, embeddings):
            """
            return concatenated input
            """
            n = N / 2
            sent_len = inp.shape[0]
            ret, upd = theano.scan(fn=self.slice,
                                   outputs_info=None,
                                   sequences=[T.arange(n, sent_len-n)],
                                   non_sequences=[embeddings, N])
            return ret

        @staticmethod
        def getfunc(st):
            """
            return activation function, given by string
            """
            dic = {}
            dic["tanh"] = T.tanh
            dic["linear"] = (lambda x: x)
            dic["sigmoid"] = T.nnet.sigmoid
            dic["softmax"] = T.nnet.softmax
            return dic[st]

        def setlayers(self, inputl):
            """
            set network
            """
            def setlayer(self, inputl, size=(0, 0), name1="w", name2="b", act="sigmoid"):
                w = self.seg.params[name1]
                b = self.seg.params[name2]
                a = theano.dot(inputl, w) + b
                activate = self.getfunc(act)
                o = activate(a)
                return o

            l1 = setlayer(inputl, size=(self.seg.IL, self.seg.HL), name1="W1", name2="b1", act=self.seg.hid_act)
            return setlayer(l1, size=(self.seg.HL, self.seg.OL), name1="W2", name2="b2", act=self.seg.out_act)

        def trace_back_(self, path, maxnode):
            path_rev = theano.dot(self.R, path)
            [_, path_ret], _ = theano.scan(fn=lambda x, i: [x[i], i],
                                           outputs_info=[maxnode, None],
                                           sequences=[path_rev])
            ret = theano.dot(self.R, path_ret)
            return ret

        @staticmethod
        def trans(X, prior, W):
            def maxpath(pri, W):
                res = pri + W.T
                return T.max_and_argmax(res, axis=1)

            maxi = maxpath(prior, W)
            return [maxi[0] + X, maxi[1]]

        def viterbi(self, outputs):
            """
            get viterbi scores and path
            """
            [score, path], upd = theano.scan(fn=self.trans,
                                             outputs_info=[T.zeros_like(outputs[0]), self.startnode],
                                             sequences=[outputs],
                                             non_sequences=self.seg.params['A'])

            maxscore, maxarg = T.max_and_argmax(score[-1])
            return self.trace_back_(path, maxarg)

        def setnet(self, inp, ans, N):
            embeddings = self.lookup(self.seg.params['C'], inp)
            inputs = self.makeinputs(N, inp, embeddings)
            outputs = self.setlayers(inputs)
            out = self.viterbi(outputs)
            dif = theano.tensor.neq(ans, out)
            aS = theano.tensor.sum(outputs[theano.tensor.arange(outputs.shape[0]), ans] * dif)
            oS = theano.tensor.sum(outputs[theano.tensor.arange(outputs.shape[0]), out] * dif)
            itr = (dif > 0).nonzero()
            itr = itr(itr.nonzero())
            transdif = theano.tensor.sum(self.seg.params['A'][itr-1, itr])
            self.viterbi_path = out
            self.count = T.sum(self.seg.X[out, ans])
            return (aS - oS, transdif)

    # class init
    def __init__(self,
                 HL=300,
                 OL=4,
                 chardiclen=0,
                 bufferlen=6000,
                 char_d=50,
                 N=5,
                 iter=10,
                 initupper=0.01,
                 batchsize=20,
                 alfa=0.02,
                 hid_act="tanh",
                 out_act="softmax"):
        """
        """
        args = locals()
        args.pop("self")
        self.__dict__.update(args)

        self.IL = char_d * N
        self.alfa = numpy.float32(alfa)

        self.params = OrderedDict()
        self.params_v = OrderedDict()
        self.idxs = T.lvector('idxs')

        self.setparams(chardiclen, char_d)
        self.set_batch_networks()

    def getdata(self, idx):
        idx0 = self.datas_idx[idx][0].astype("int64")
        idx1 = self.datas_idx[idx][1].astype("int64")
        idx2 = self.datas_idx[idx][2].astype("int64")
        idx3 = self.datas_idx[idx][3].astype("int64")
        return (self.data[idx0:idx1], self.data_ans[idx2:idx3])

    def setparams(self, chardiclen, char_d):
        def setdata(chardiclen):
            """
            define datas and give the set function
            """
            self.datas_idx = theano.shared(numpy.zeros((self.bufferlen, 4)))
            self.data = theano.shared(numpy.zeros((self.bufferlen, ), dtype="int64"), name="data")
            self.data_ans = theano.shared(numpy.zeros((self.bufferlen,)))
            idx0 = self.datas_idx[self.idx][0].astype("int64")
            idx1 = self.datas_idx[self.idx][1].astype("int64")
            idx2 = self.datas_idx[self.idx][2].astype("int64")
            idx3 = self.datas_idx[self.idx][3].astype("int64")
            self.inp = self.data[idx0:idx1].astype("int64")
            self.ans = self.data_ans[idx2:idx3].astype("int64")

            d = T.lvector()
            d_ans = T.lvector()
            d_idx = T.lmatrix()
            u = (self.data, T.set_subtensor(self.data[:d.shape[0]], d))
            u_ans = (self.data_ans, T.set_subtensor(self.data_ans[:d_ans.shape[0]], d_ans))
            u_idx = (self.datas_idx, T.set_subtensor(self.datas_idx[:d_idx.shape[0]], d_idx))
            self.set = theano.function([d, d_ans, d_idx], updates=[u, u_ans, u_idx])

        def setparam(size1, size2=None, name=""):
            upper = self.initupper
            if size2 is None:
                shape = (size1)
                variable = T.vector()
            else:
                shape = (size1, size2)
                variable = T.matrix()
            arr = numpy.random.uniform(-upper, upper, shape)
            self.params[name] = theano.shared(arr.astype(FLOAT), name=name)
            self.params_v[name] = variable

        setdata(chardiclen)
        #  transition score
        setparam(self.OL, self.OL, name="A")
        #  set lookup
        setparam(chardiclen, char_d, name="C")
        #  set weight and bias
        setparam(self.IL, self.HL, name="W1")
        setparam(self.HL, name="b1")
        setparam(self.HL, self.OL, name="W2")
        setparam(self.OL, name="b2")

    def set_batch_networks(self):
        """
        construct networks for batch
        """
        self.X = theano.shared(numpy.zeros((self.OL, self.OL)))
        self.nets = range(self.batchsize)
        net_S = []
        trans_S = []
        match_count = []
        for i in xrange(self.batchsize):
            self.nets[i] = self.network(self, self.idxs[i], N=self.N)
            ns, ts = self.nets[i].Scores
            # append score variables for sum
            net_S.append(ns)
            trans_S.append(ts)
            match_count.append(self.nets[i].count)
        # transition score updates
        trans_p = [self.params["A"]]
        trans_grad = theano.grad(T.sum(trans_S), trans_p)
        trans_upd = [(p, p + self.alfa * g) for p, g in zip(trans_p, trans_grad)]
        # network parameters update
        net_p = [p for k, p in self.params.items() if k != "A"]
        net_grad = theano.grad(T.sum(net_S), net_p)
        net_upd = [(p, p + self.alfa * g) for p, g in zip(net_p, net_grad)]
        # training function
        upd = trans_upd + net_upd
        self.learn = theano.function([self.idxs], updates=upd)
        #  counting function for test
        test_g = theano.grad(T.sum(match_count), self.X)
        test_upd = [(self.X, self.X + test_g)]
        self.test = theano.function([self.idxs], updates=test_upd)

    @property
    def system_out(self):
        return theano.function([self.idxs], [net.viterbi_path for net in self.nets])

    def replacedata(self, inputdata, anslist):
        """
        copy data to gpu memory

        given: inputdata, anslist -- [sent1, sent2, ...],[ans1, ans2, ...]
        return: length of self.datas_idx
        """
        datarr = []
        ansdata = []
        idxarr = []
        for data, ans in zip(inputdata, anslist):
            if len(idxarr) > 0:
                lastidx = idxarr[-1][1]
                lastidx2 = idxarr[-1][3]
            else:
                lastidx = 0
                lastidx2 = 0
            datarr.extend(data)
            ansdata.extend(ans)
            idxarr.append([lastidx, lastidx+len(data), lastidx2, lastidx2 + len(ans)])
        datarr = numpy.array(datarr).astype('int64')
        ansdata = numpy.array(ansdata).astype('int64')
        idxarr = numpy.array(idxarr).astype('int64')
        print datarr.shape
        print ansdata.shape
        print idxarr.shape
        self.set(datarr, ansdata, idxarr)
        return len(idxarr)

    def error_handle(self, e, idxlist):
        print "error:"
        print "type:", str(type(e))
        print "args:", str(e.args)

    def training(self, data):
        """
        training parameters from data

        given: data -- [(sent, ans), ...]
            # sent: character index list of a sentence
            # ans : tag index list of a sequence answer
        return: -
        """
        def learn_(buflen):
            L = range(buflen)
            for x in xrange(buflen / self.batchsize):
                start = x * self.batchsize
                end = (x + 1) * self.batchsize
                if end > buflen:
                    end = buflen
                self.learn(L[start:end])
                """
                # for  debug
                out = self.system_out(L[start:end])
                for o in out:
                    print o
                """

        allsize = sum([len(sent) for sent, _ in data])
        allin = (allsize < self.bufferlen)
        if allin:
            inputbuf = [sent for sent, _ in data]
            ansbuf = [ans for _, ans in data]
            buflen = self.replacedata(inputbuf, ansbuf)
            for i in range(self.iter):
                learn_(buflen)
        # if data can't be set on gpu memory once
        else:
            for i in range(self.iter):
                buffersum = 0
                inputbuf = []
                ansbuf = []
                for sent, ans in data:
                    if (len(sent) + buffersum) > self.bufferlen:
                        buflen = self.replacedata(inputbuf, ansbuf)
                        try:
                            learn_(buflen)
                        except Exception as e:
                            self.error_handle(e)
                        inputbuf = [sent]
                        ansbuf = [ans]
                        buffersum = len(sent)
                    else:
                        inputbuf.append(sent)
                        ansbuf.append(ans)
                        buffersum += len(sent)
                buflen = self.replacedata(inputbuf, ansbuf)
                try:
                    learn_(buflen)
                except Exception as e:
                    self.error_handle(e)

    def test_(self, dataset):
        """
        given: dataset -- [(sent, ans), ...]
            # sent: character index list of a sentence
            # ans : tag index list of a sequence answer

        return: F-score
        """
        def acc_(buflen):
            L = range(buflen)
            for x in xrange(buflen / self.batchsize):
                start = x * self.batchsize
                end = (x + 1) * self.batchsize
                if end > buflen:
                    end = buflen
                acc_(range(buflen)[start:end])
                self.test(L[start:end])

        inputbuf = []
        ansbuf = []
        buffersum = 0
        for sent, ans in dataset:
            if (len(sent) + buffersum) > self.bufferlen:
                buflen = self.replacedata(inputbuf, ansbuf)
                try:
                    acc_(buflen)
                except Exception as e:
                    self.error_handle(e)
                inputbuf = [sent]
                ansbuf = [ans]
                buffersum = len(sent)
            else:
                inputbuf.append(sent)
                ansbuf.append(ans)
        buflen = self.replacedata(inputbuf, ansbuf)
        try:
            acc_(buflen)
        except Exception as e:
            self.error_handle(e)
        # calculate F-score
        x = self.X.get_value()
        precision = numpy.average([(x[i] / sum(x[i]) if sum(x[i]) != 0 else 0) for i in range(self.OL)])
        recall = numpy.average([(x[i] / sum(x.t[i]) if sum(x.t[i]) != 0 else 0) for i in range(self.OL)])
        print x
        return (2 * precision * recall) / (recall + precision)

    def load_param(self, filename):
        """
            loading parameters from .npz file
        """
        setfunc = {}
        npzfile = numpy.load(filename)
        for key in npzfile.keys():
            variable = self.params_v[key]
            arr = self.params[key]
            sets = (arr, T.set_subtensor(arr[:], variable))
            setfunc[key] = theano.function([variable], updates=[sets])
        for key, arr in npzfile.items():
            setfunc[key](arr)

    def write_(self, filename):
        dic = {}
        for key, item in self.params.items():
            arr = item.get_value()
            dic[key] = arr
        numpy.savez(filename, **dic)
