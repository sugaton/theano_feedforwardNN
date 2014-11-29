import theano.tensor as T
import numpy
import theano
from collections import OrderedDict
from theano import ifelse

FLOAT = theano.config.floatX


class segmenter(object):
    class network:
        def __init__(self, seg, idx, N=5):
            self.seg = seg
            i, a = self.seg.getdata(idx)
            netS, tranS = self.setnet(i, a, N)
            # if idx == NULL
            cond = T.eq(self.seg.NULL, idx)
            self.netS = ifelse.ifelse(cond, self.seg.ZERO, netS)
            self.tranS = ifelse.ifelse(cond, self.seg.ZERO, tranS)

        @property
        def Scores(self):
            return (self.netS, self.tranS)

        @property
        def count(self):
            return self.count

        @property
        def sys_out(self):
            return self.viterbi_path

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
            return T.reshape(X[i-n:i+n+1], (X.shape[1]*N, ))

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
            def setL(inputl, name1="w", name2="b", act="sigmoid"):
                w = self.seg.params[name1]
                b = self.seg.params[name2]
                a = theano.dot(inputl, w) + b
                activate = self.getfunc(act)
                o = activate(a)
                return o

            l1 = setL(inputl, name1="W1", name2="b1", act=self.seg.hid_act)
            return setL(l1, name1="W2", name2="b2", act=self.seg.out_act)

        def trace_back_(self, path, maxnode):
            n = path.shape[1]
            path_rev = path[:, n - T.arange(n) - 1]
            [_, path_ret], _ = theano.scan(fn=lambda x, i: [x[i], i],
                                           outputs_info=[maxnode, None],
                                           sequences=[path_rev])
            n = path_ret.shape[0]
            ret = path_ret[n - T.arange(n) - 1]
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
                                             outputs_info=[self.seg.startstate, None],
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
            itr = dif.nonzero()[0]  # element-indexes which satisfy dif[idx]==1
            aS = theano.tensor.sum(outputs[itr, ans[itr]])
            oS = theano.tensor.sum(outputs[itr, out[itr]])
            transdif_a = T.sum(self.seg.params['A'][ans[itr-1], ans[itr]])
            transdif_o = T.sum(self.seg.params['A'][out[itr-1], out[itr]])
            self.viterbi_path = out
            self.count = T.sum(self.seg.X[out, ans])
            return (aS - oS, transdif_a - transdif_o)

    #######
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
                 viterbi_startnode=3,
                 alfa=0.02,
                 hid_act="tanh",
                 out_act="softmax"):
        """
        """
        # set arguments to self.*
        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        #
        self.IL = char_d * N
        self.alfa = numpy.float32(alfa)
        arr = numpy.array([1 if i==viterbi_startnode else -1 for i in range(OL)])
        # special constant shared variable
        self.startstate = theano.shared(arr.astype(FLOAT))
        self.NULL = theano.shared(numpy.array(-1).astype("int64"))
        self.ZERO = theano.shared(numpy.array(0).astype(FLOAT))
        self.params = OrderedDict()
        self.params_v = OrderedDict()
        self.idxs = T.lvector('idxs')
        # compile networks
        self.setparams(chardiclen, char_d)
        self.set_batch_networks()

    def getdata(self, idx):
        idx0 = self.datas_idx[idx][0].astype("int64")
        idx1 = self.datas_idx[idx][1].astype("int64")
        idx2 = self.datas_idx[idx][2].astype("int64")
        idx3 = self.datas_idx[idx][3].astype("int64")
        return (self.data[idx0:idx1].astype("int64"), self.data_ans[idx2:idx3].astype("int64"))

    def setparams(self, chardiclen, char_d):
        def setdata(chardiclen):
            """
            define datas and give the set function
            """
            self.datas_idx = theano.shared(numpy.zeros((self.bufferlen, 4)))
            self.data = theano.shared(numpy.zeros((self.bufferlen, ), dtype="int64"), name="data")
            self.data_ans = theano.shared(numpy.zeros((self.bufferlen,)))
            # define the function to put data to gpu memory
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
        sys_out = []
        for i in xrange(self.batchsize):
            self.nets[i] = self.network(self, self.idxs[i], N=self.N)
            ns, ts = self.nets[i].Scores
            # append score variables for sum
            net_S.append(ns)
            trans_S.append(ts)
            match_count.append(self.nets[i].count)
            o_ = self.nets[i].sys_out
            sys_out.append(o_)
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
        self.learn_with_out = theano.function([self.idxs], sys_out, updates=upd)
        #  counting function for test
        test_g = theano.grad(T.sum(match_count), self.X)
        test_upd = [(self.X, self.X + test_g)]
        self.test = theano.function([self.idxs], updates=test_upd)

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
    def error_handle(self, e):
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
        null = self.NULL.get_value()
        def learn_(buflen):
            L = range(buflen)
            for x in range(int(round(1.0 * buflen / self.batchsize))):
                start = x * self.batchsize
                end = (x + 1) * self.batchsize
                if end > buflen:
                    emplen = end - buflen
                    end = buflen
                    L_ = L[start:end] + [null for i in range(emplen)]
                    # self.learn(L_)
                    outs = self.learn_with_out(L_)
                    anss = [self.getdata(i)[1] for i in L_]
                else:
                    # self.learn(L[start:end])
                    outs = self.learn_with_out(L[start:end])
                    anss = [self.getdata(i)[1] for i in L[start:end]]
                # for debug
                for ans, out in zip(anss, outs):
                    print ""
                    print("ans:", ans.eval())
                    print("out:", out)

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
