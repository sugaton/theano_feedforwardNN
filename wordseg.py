import theano.tensor as T
import scipy
import numpy
import theano
from collections import OrderedDict
from theano import ifelse
import math

FLOAT = theano.config.floatX
IFEL = ifelse.ifelse


class segmenter(object):
    class adagrad:
        def __init__(self, gtype, alfa=0.02, ifreset=False, countmax=100):
            self._alfa = alfa
            self._gradsum = T.ones_like(gtype)
            self._gradsum_init = T.ones_like(gtype)
            # parameters for resetting _grad_sum
            self._ifreset = ifreset
            self._counter = 0
            self._countmax = countmax

        def getupdate(self, grad):
            self._gradsum += grad * grad
            alf = self._alfa * T.sqrt(self._gradsum)
            # if _grad_sum will be reseted
            if self._ifreset:
                self._counter += 1
                if self._counter > self._countmax:
                    self._counter = 0
                    self._gradsum = self._gradsum_init
            return alf * grad

    class constant_alfa:
        def __init__(self, alfa):
            self._alfa = alfa

        def getupdate(self, grad):
            return self._alfa * grad

    class network:
        def __init__(self, seg, idx, N=5):
            self.seg = seg
            self.estimation = self.seg.estimation
            i, a = self.seg.getdata(idx)
            self.setnet(i, a, N)

        @property
        def Scores(self):
            return self.scores

        @property
        def count(self):
            return self.count

        @property
        def sys_out(self):
            return self.viterbi_path

        @property
        def pretrain_upd(self):
            return self.pretraing_grad

        def lookup(self, C, inp):
            return C[inp]
            """
            result, updates = theano.scan(fn=lambda x: C[x],
                                          outputs_info=None,
                                          sequences=[inp])
            return result
            """
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
            self.layers.append(ret)
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
                self.layers.append(a)
                self.layers.append(o)
                return o

            l1 = setL(inputl, name1="W1", name2="b1", act=self.seg.hid_act)
            self.debug1 = l1
            l2 = setL(l1, name1="W2", name2="b2", act=self.seg.out_act)
            return l2

        def lossfunc(self, x, z):
            if self.seg.loss == "norm2":
                err = x-z
                return T.sqrt(T.sum(theano.dot(err.T, err)[T.arange(err.shape[1]), T.arange(err.shape[1])]))
            else:
                return None

        def _pretraining(self):
            """
            return each layer autoencoder's error
            """
            def setL(x, name1="w", name2="b", name3="b_", act="sigmoid"):
                w = self.seg.params[name1]
                b = self.seg.params[name2]
                b_ = self.seg.params[name3]
                activate = self.getfunc(act)
                y = activate(theano.dot(x, w) + b)
                z = activate(theano.dot(y, w.T) + b_)
                return zip([w, b, b_], theano.grad(self.lossfunc(x, z), [w, b, b_]))

            upd1 = setL(self.layers[0], "W1", "b1", "b1_", act=self.seg.hid_act)
            upd2 = setL(self.layers[1], "W2", "b2", "b2_", act=self.seg.hid_act)
            alf = self.seg.alfa
            return [(p, alf[p] * g + p) for p, g in upd1 + upd2]

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
            maxi = T.max_and_argmax(prior + W.T, axis=1)
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

        def _collins_estimation(self, outputs, ans):
            """
            following collins 2002's approach
            """
            out = self.viterbi(outputs)
            dif = theano.tensor.neq(ans, out)
            itr = dif.nonzero()[0]  # element-indexes which satisfy dif[idx]==1
            aS = theano.tensor.sum(outputs[itr, ans[itr]])
            oS = theano.tensor.sum(outputs[itr, out[itr]])
            transdif_a = T.sum(self.seg.params['A'][ans[itr-1], ans[itr]])
            transdif_o = T.sum(self.seg.params['A'][out[itr-1], out[itr]])
            return (aS - oS, transdif_a - transdif_o)

        def _collobert_estimation(self, outputs, ans):
            """
            following collobert 2011's approach
            """
            def logadd(outputs):
                """
                logadd
                """
                def log_sum_exp(X):
                    x = X.max()
                    return x + T.log(T.sum(T.exp(X-x), axis=1))

                score, _ = theano.scan(fn=lambda x_t, x_t_1, W: x_t + log_sum_exp(x_t_1 + W.T),
                                       outputs_info=[self.seg.startstate],
                                       sequences=[outputs],
                                       non_sequences=self.seg.params['A'])
                smax = score[-1].max()
                return smax + T.log(T.sum(T.exp(score[-1] - smax)))

            def ans_score(ans, outputs):
                arr = T.arange(ans.shape[0])
                sum1 = T.sum(outputs[arr, ans])
                arr = T.arange(ans.shape[0] - 1)
                st = self.seg.params["A"][self.seg.viterbi_startnode, ans[0]]
                sum2 = T.sum(self.seg.params["A"][ans[arr], ans[arr + 1]]) + st
                return sum1 + sum2

            return ans_score(ans, outputs) - logadd(outputs)

        def setnet(self, inp, ans, N):
            embeddings = self.lookup(self.seg.params['C'], inp)
            self.layers = []
            inputs = self.makeinputs(N, inp, embeddings)
            outputs = self.setlayers(inputs)
            self.outputs = outputs
            if self.estimation == "collins":
                self.netS, self.tranS = self._collins_estimation(outputs, ans)
                self.scores = (self.netS, self.tranS)
            else:
                self.scores = self._collobert_estimation(outputs, ans)
            if self.seg.pretrain:
                self.pretraing_grad = self._pretraining()
            self.viterbi_path = self.viterbi(outputs)
            self.count = T.sum(self.seg.X[self.viterbi_path, ans])

        def get_gradients(self):
            dot = theano.dot
            _dO = theano.grad(self.netS, self.outputs)
            _b2 = T.sum(_dO, axis=0)
            H = self.layers[-3]
            _dW2 = dot(H.T, _dO)
            _dH = dot(_dO, self.seg.params["W2"].T)
            I = self.layers[0]
            _dA = _dH * (H - H * H)
            _b1 = T.sum(_dA, axis=0)
            _dW1 = dot(I.T, _dA)
            _I = dot(_dA, self.seg.params["W1"].T)
            _C = theano.grad(T.sum(I * _I), self.seg.params["C"])
            return [_C, _dW1, _b1, _dW2, _b2]

    #######
    def __init__(self,
                 HL=300,
                 OL=4,
                 chardiclen=0,
                 bufferlen=6000,
                 char_d=50,
                 N=5,
                 iter=10,
                 initupper=1e-04,
                 batchsize=20,
                 feat_d=0,
                 pretrain=False,
                 viterbi_startnode=3,
                 estimation="collobert",
                 if_debug=False,
                 loss="norm2",
                 learning_rate_method="constant",
                 alfa_=0.02,
                 len_nulldata=3,
                 hid_act="sigmoid",
                 out_act="linear"):
        """
        """
        # set arguments to self.*
        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        #
        self.IL = (char_d + feat_d) * N
        self.OL = OL + 1
        self.alfa_ = numpy.float32(alfa_)
        arr = numpy.array([1 if i==viterbi_startnode else -1 for i in range(self.OL)])
        # special constant shared variable
        self.startstate = theano.shared(arr.astype(FLOAT))
        self.NULL = theano.shared(numpy.array(-1).astype("int64"))
        self.biass = {}
        self.params = OrderedDict()
        self.params_v = OrderedDict()
        self.idxs = T.lvector('idxs')
        # compile networks
        self.setparams(chardiclen, char_d, feat_d)
        self.set_batch_networks()

    def getdata(self, idx):
        idx0 = self.datas_idx[idx][0].astype("int64")
        idx1 = self.datas_idx[idx][1].astype("int64")
        idx2 = self.datas_idx[idx][2].astype("int64")
        idx3 = self.datas_idx[idx][3].astype("int64")
        return (self.data[idx0:idx1].astype("int64"), self.data_ans[idx2:idx3].astype("int64"))

    def setparams(self, chardiclen, char_d, feat_d):
        def setdata(chardiclen):
            """
            define datas and give the set function
            """
            null_ = self.chardiclen
            SSub = T.set_subtensor
            nulld_size = self.len_nulldata + (self.N / 2) * 2
            self.datas_idx = theano.shared(numpy.zeros((self.bufferlen+1, 4)))
            self.data = theano.shared(numpy.zeros((self.bufferlen + nulld_size,), dtype="int64"), name="data")
            self.data_ans = theano.shared(numpy.zeros((self.bufferlen+self.len_nulldata,)))
            # data which is correspond to null input
            nulld_ = theano.shared(numpy.array([null_ for i in range(nulld_size)]).astype("int64"))
            nulla_ = theano.shared(numpy.array([self.OL - 1 for i in range(self.len_nulldata)]).astype("int64"))
            arr = [self.bufferlen, self.bufferlen+nulld_size, self.bufferlen, self.bufferlen+self.len_nulldata]
            nulli_ = theano.shared(numpy.array(arr).astype("int64"))
            # define the function to put data to gpu memory
            d = T.lvector()
            d_ans = T.lvector()
            d_idx = T.lmatrix()
            u = (self.data, SSub(SSub(self.data[:d.shape[0]], d)[-nulld_.shape[0]:], nulld_))
            u_ans = (self.data_ans, SSub(SSub(self.data_ans[:d_ans.shape[0]], d_ans)[-nulla_.shape[0]:], nulla_))
            u_idx = (self.datas_idx, SSub(SSub(self.datas_idx[:d_idx.shape[0]], d_idx)[-1], nulli_))
            self.set = theano.function([d, d_ans, d_idx], updates=[u, u_ans, u_idx])

        def setparam(size1, size2=None, upper=self.initupper, name="", bias=1.):
            if size2 is None:
                shape = (size1)
                variable = T.vector()
            else:
                shape = (size1, size2)
                variable = T.matrix()
            arr = numpy.random.uniform(-upper, upper, shape)
            self.params[name] = theano.shared(arr.astype(FLOAT), name=name)
            self.params_v[name] = variable
            self.biass[self.params[name]] = numpy.float32(bias)

        def setalfa():
            if self.learning_rate_method == "constant":
                self.alfa = dict([(p, self.constant_alfa(self.alfa_)) for p in self.params.values()])
            elif self.learning_rate_method == "adagrad":
                self.alfa = dict([(p, self.adagrad(p, self.alfa_)) for p in self.params.values()])

        setdata(chardiclen)
        #  transition score
        setparam(self.OL, self.OL, upper=0, name="A")
        #  set lookup
        setparam(chardiclen + 1, char_d + feat_d, name="C")
        #  set weight and bias
        setparam(self.IL, self.HL, name="W1")
        setparam(self.HL, upper=0, name="b1")
        if self.pretrain:
            setparam(self.IL, upper=0, name="b1_")
        setparam(self.HL, self.OL, name="W2")
        setparam(self.OL, upper=0, name="b2")
        if self.pretrain:
            setparam(self.HL, upper=0, name="b2_")
        setalfa()

    def set_batch_networks(self):
        """
        construct networks for batch
        """
        def _collins_grad(scores):
            trans_p = [self.params["A"]]
            net_p = [p for k, p in self.params.items() if k != "A"]
            net_S = [ns for ns, ts in scores]
            trans_S = [ts for ns, ts in scores]
            # transition score updates
            transg = [theano.grad(S, trans_p) for S in trans_S]
            trans_grad = [sum([transg[i][j] for i in range(len(transg))]) for j in range(len(trans_p))]
            trans_upd = [(p, p + self.alfa[p].getupdate(g)) for p, g in zip(trans_p, trans_grad)]
            # network parameters update
            netsg = [theano.grad(S, net_p) for S in net_S]
            net_grad = [sum([netsg[i][j] for i in range(len(netsg))]) for j in range(len(net_p))]
            # net_grad = [theano.grad(net_S[i], p) for p in net_p]
            net_upd = [(p, p + self.alfa[p].getupdate(g)) for p, g in zip(net_p, net_grad)]
            return trans_upd + net_upd

        def _collobert_grad(scores):
            def grad_(index, scores):
                ifnull = [T.zeros_like(p) for p in self.params.values()]
                g = IFEL(T.eq(self.idxs[index], self.NULL), ifnull, theano.grad(scores[index], self.params.values()))
                return [T.where(T.isnan(g_), T.zeros_like(g_), g_) for g_ in g]
            grads = [grad_(i, scores) for i in range(self.batchsize)]
            grad = [sum([grads[i][j] for i in range(self.batchsize)]) / self.batchsize for j in range(len(self.params))]
            upd = [(p, p + self.alfa[p].getupdate(g)) for p, g in zip(self.params.values(), grad)]
            self.grad = theano.grad(scores[0], self.nets[0].debug1)
            return upd

        print("setting networks")
        self.X = theano.shared(numpy.ones((self.OL, self.OL)))
        self.nets = range(self.batchsize)
        match_count = []
        scores = []
        sys_out = []
        print("--setting all networks")
        for i in xrange(self.batchsize):
            self.nets[i] = self.network(self, self.idxs[i], N=self.N)
            scores.append(self.nets[i].Scores)
            match_count.append(self.nets[i].count)
            o_ = self.nets[i].sys_out
            sys_out.append(o_)
        debug = self.nets[0].outputs
        print("--compiling gradient function")
        if self.estimation == "collins":
            upd = _collins_grad(scores)
            scores = [n + a for n, a in scores]
        else:
            upd = _collobert_grad(scores)
        print("--compiling learning, testing function")
        # training function
        self.learn = theano.function([self.idxs], None, updates=upd)
        # self.learn_with_out = theano.function([self.idxs], self.grad, updates=upd)
        self.learn_with_out = theano.function([self.idxs], [T.sum(scores) / len(scores)], updates=upd)
        # self.learn_with_out = theano.function([self.idxs], debug, updates=upd)
        # self.learn_with_out = theano.function([self.idxs], sys_out, updates=upd)
        #  counting function for test
        test_g = theano.grad(T.sum(match_count), self.X)
        test_upd = [(self.X, self.X + test_g)]
        self.test = theano.function([self.idxs], sys_out, updates=test_upd)

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

        print idxarr.shape
        self.set(datarr, ansdata, idxarr)
        return len(idxarr)

    def error_handle(self, e):
        print "error:"
        print "type:", str(type(e))
        print "args:", str(e.args)

    def function_apply_data(self, data, func, iter_=1):
        allsize = sum([len(sent) for sent, _ in data])
        allin = (allsize < self.bufferlen)
        if allin:
            inputbuf = [sent for sent, _ in data]
            ansbuf = [ans for _, ans in data]
            buflen = self.replacedata(inputbuf, ansbuf)
            for i in range(iter_):
                func(buflen)
        # if data can't be set on gpu memory once
        else:
            for i in range(iter_):
                buffersum = 0
                inputbuf = []
                ansbuf = []
                for sent, ans in data:
                    if (len(sent) + buffersum) > self.bufferlen:
                        buflen = self.replacedata(inputbuf, ansbuf)
                        try:
                            func(buflen)
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
                    func(buflen)
                except Exception as e:
                    self.error_handle(e)

    @staticmethod
    def debug_print(outs, anss):
        for out, ans in zip(outs, anss):
            print ans.eval()
            print out
            print ""

    def training(self, data):
        """
        training parameters from data

        given: data -- [(sent, ans), ...]
            # sent: character index list of a sentence
            # ans : tag index list of a sequence answer
        return: -
        """
        null = self.NULL.get_value()
        if self.if_debug:
            learn = self.learn_with_out
            dp = self.debug_print
        else:
            learn = self.learn
            dp = lambda x, y: None

        def learn_(buflen):

            L = range(buflen)
            sum_outs = 0
            c = 0
            for x in range(int(math.ceil(1.0 * buflen / self.batchsize))):
                start = x * self.batchsize
                end = (x + 1) * self.batchsize
                if end > buflen:
                    emplen = end - buflen
                    end = buflen
                    L_ = L[start:end] + [null for i in range(emplen)]
                    outs = learn(L_)
                    anss = [self.getdata(i)[1] for i in L_]
                else:
                    outs = learn(L[start:end])
                    # outs = self.learn_with_out(L[start:end])
                    anss = [self.getdata(i)[1] for i in L[start:end]]
                sum_outs += outs
                c += 1
                # dp(outs, anss)
            print("iteration done")
            print "error:", sum_outs / c

        self.function_apply_data(data, learn_, iter_=self.iter)

    def test_(self, dataset):
        """
        given: dataset -- [(sent, ans), ...]
            # sent: character index list of a sentence
            # ans : tag index list of a sequence answer

        return: F-score
        """
        null = self.NULL.get_value()
        dp = self.debug_print

        def acc_(buflen):
            null = self.NULL.get_value()
            L = range(buflen)
            for x in xrange(buflen / self.batchsize):
                start = x * self.batchsize
                end = (x + 1) * self.batchsize
                if end > buflen:
                    emplen = end - buflen
                    end = buflen
                    L_ = L[start:end] + [null for i in range(emplen)]
                    outs = self.test(L_)
                    anss = [self.getdata(i)[1] for i in L_]
                else:
                    outs = self.test(L[start:end])
                    anss = [self.getdata(i)[1] for i in L[start:end]]
                dp(outs, anss)

        self.function_apply_data(dataset, acc_)
        # calculate F-score
        x = self.X.get_value()[:] - 1
        print x
        precision = numpy.average([(x[i] / sum(x[i]) if sum(x[i]) != 0 else 0) for i in range(self.OL)])
        recall = numpy.average([(x[i] / sum(x.T[i]) if sum(x.T[i]) != 0 else 0) for i in range(self.OL)])
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

    def load_feat(self, arr):
        v = self.params_v["C"]
        p = self.params["C"]
        upd = [(p, T.set_subtensor(p[:, -self.feat_d:], v))]
        setfunc = theano.function([v], updates=upd)
        setfunc(arr)

    def load_lookup(self, arr):
        v = self.params_v["C"]
        p = self.params["C"]
        setfunc = theano.function([v], updates=[(p, T.set_subtensor(p[:], v))])
        setfunc(arr)

    def write_(self, filename):
        dic = {}
        for key, item in self.params.items():
            arr = item.get_value()
            dic[key] = arr
        numpy.savez(filename, **dic)
