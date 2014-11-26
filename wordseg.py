import feedforwardNN
import math
import numpy
import theano
from collections import OrderedDict

FLOAT = theano.config.floatX


class segmenter(feedforwardNN.feedforwardNN):
    def __init__(self, HL=300, OL=4, chardiclen=0, bufferlen=6000, char_d=50, N=5, iter=10, alfa=0.02, hid_act="tanh", out_act="softmax"):
        """
        input is a matrix which rows are one-hot representation of character.
        then, the input-matrix's size is
            (sentence-length:N, all-character-num:V) [ V * N ].
        self.ans is a tag sequence of correct answer. each tag is described as indexnumber
        """
        self.IL = char_d * N
        self.HL = HL
        self.OL = OL
        self.iter = iter
        self.alfa = alfa
        self.hid_act = hid_act
        self.out_act = out_act
        self.bufferlen = bufferlen

        self.params = OrderedDict()
        self.params_v = OrderedDict()
        self.idx = theano.tensor.lscalar('idx')

        self.setData(chardiclen)
<<<<<<< HEAD
        self.settransition()
        self.layers = []
=======
        self.layers = [self.local_in]
>>>>>>> 7f12259fc4eea2ef090831be5ed26889e693ab06
        self.setlookup(V=chardiclen, d=char_d)
        self.makeinputs(N=N)

        self.setlayers()
        self.out = self.layers[-1]
        self.maxscore, self.viterbipath, lastnode = self.viterbi()
        self.maxpath = self.trace_back_(self.viterbipath, lastnode).astype("int64")

        self.gradients = self.set_grad()

        self.comp = theano.function([self.idx], [self.out, self.maxpath], updates=self.gradients)
        self.getdata = theano.function([self.idx], [self.inp, self.ans])
        self.getlayers = theano.function([self.idx], [self.out, self.viterbipath, lastnode])
        self.getmaxpath = theano.function([self.idx], [self.maxpath, self.ans])

    def setData(self, chardiclen):
        self.datas_idx = theano.shared(numpy.zeros((self.bufferlen, 4)))
        self.data = theano.shared(numpy.zeros((self.bufferlen, ), dtype="int64"), name="data")
        self.data_ans = theano.shared(numpy.zeros((self.bufferlen,)))
        idx0 = self.datas_idx[self.idx][0].astype("int64")
        idx1 = self.datas_idx[self.idx][1].astype("int64")
        idx2 = self.datas_idx[self.idx][2].astype("int64")
        idx3 = self.datas_idx[self.idx][3].astype("int64")
        self.inp = self.data[idx0:idx1].astype("int64")
        self.ans = self.data_ans[idx2:idx3].astype("int64")

        d = theano.tensor.lvector()
        d_ans = theano.tensor.lvector()
        d_idx = theano.tensor.lmatrix()
        u = (self.data, theano.tensor.set_subtensor(self.data[:d.shape[0]], d))
        u_ans = (self.data_ans, theano.tensor.set_subtensor(self.data_ans[:d_ans.shape[0]], d_ans))
        u_idx = (self.datas_idx, theano.tensor.set_subtensor(self.datas_idx[:d_idx.shape[0]], d_idx))
        self.set = theano.function([d, d_ans, d_idx], updates=[u, u_ans, u_idx])

    def settransition(self):
        arr = numpy.random.randn(self.OL, self.OL) / numpy.sqrt(2 * self.OL)
        A = theano.shared(arr.astype(FLOAT), name="A")
        self.params["A"] = A
        self.params_v["A"] = theano.tensor.matrix()

    def lookup(self, C):
        result, updates = theano.scan(fn=lambda x: C[x],
                                      outputs_info=None,
                                      sequences=[self.inp])
        return result

    def setlookup(self, V=1, d=1):
        arr = numpy.random.randn(V, d) / numpy.sqrt(V + d)
        C = theano.shared(arr.astype(FLOAT), name="C")
        self.params["C"] = C
        self.params_v["C"] = theano.tensor.matrix()
        inputlayer = self.lookup(self.params["C"])
        self.layers.append(inputlayer)

    def set_grad(self):
        ret = OrderedDict()
        for key, v in self.params.items():
            if key == "A":
                [__, r1] , _ = theano.scan(fn=lambda i,p,y: [i, theano.grad(v[p,i],v)+y], outputs_info=[theano.tensor.zeros_like(self.ans[0]), theano.tensor.zeros_like(v)], sequences=[self.ans])
                [__, r] , _ = theano.scan(fn=lambda i,p,y: [i, theano.grad(-1*v[p,i],v)+y], outputs_info=[theano.tensor.zeros_like(self.ans[0]), r1[-1]], sequences=[self.maxpath])
                ret[v] = v + (self.alfa * r[-1]).astype(FLOAT)
            else:
                o = self.out
                grads, _ = theano.scan(fn=lambda i,j,k,y: theano.grad(o[i,j]-o[i,k],v)+y, outputs_info=theano.tensor.zeros_like(v),sequences=[theano.tensor.arange(self.ans.shape[0]), self.ans, self.maxpath])
                ret[v] = v + (self.alfa * grads[-1]).astype(FLOAT)

        return ret

    @staticmethod
    def getfunc(st):
        def softmax_(X):
            result, updates = theano.scan(fn=lambda x: theano.tensor.reshape(theano.tensor.nnet.softmax(x), (x.shape[0], )),
                                          outputs_info=None,
                                          sequences=[dict(input=X)])
            return result
        dic = {}
        dic["tanh"] = theano.tensor.tanh
        dic["linear"] = (lambda x: x)
        dic["sigmoid"] = theano.tensor.nnet.sigmoid
        dic["softmax"] = softmax_
        return dic[st]

    @staticmethod
    def slice(i, X, N=2):
        return X[i-N:i+N+1]

    def makeinputs(self, N=5):
        n = N / 2
        sent_len = self.inp.shape[0]
        ret, upd = theano.scan(fn=self.slice,
                               outputs_info=None,
                               sequences=[theano.tensor.arange(n, sent_len-n)],
                               non_sequences=[self.layers[-1], n])
        inputlayers, upd = theano.scan(fn=lambda x: theano.tensor.reshape(x, (self.IL, )),
                                       outputs_info=None,
                                       sequences=[ret])
        self.layers.append(inputlayers)

    def setlayer(self, size=(0, 0), name1="w", name2="b", act="sigmoid"):
        arr = numpy.random.randn(size[0], size[1]) / numpy.sqrt(size[0] + size[1])
        w = theano.shared(arr.astype(FLOAT), name=name1)
        arr = numpy.random.randn(size[1]) / numpy.sqrt(size[1])
        b = theano.shared(arr.astype(FLOAT), name=name2)
        a = theano.dot(self.layers[-1], w) + b
        activate = self.getfunc(act)
        o = activate(a)
        self.layers.append(a)
        self.layers.append(o)
        self.params[name1] = w
        self.params[name2] = b
        self.params_v[name1] = theano.tensor.matrix()
        self.params_v[name2] = theano.tensor.vector()

    def setlayers(self):
        self.setlayer(size=(self.IL, self.HL), name1="W1", name2="b1", act=self.hid_act)
        self.setlayer(size=(self.HL, self.OL), name1="W2", name2="b2", act=self.out_act)

    @staticmethod
    def trans(X, prior, W):
        def maxpath(pri, W):
            res, upd = theano.scan(fn=lambda x, y: x + y,
                                   outputs_info=None,
                                   sequences=[W.T],
                                   non_sequences=pri)
            return theano.tensor.max_and_argmax(res, axis=1)

        maxi = maxpath(prior, W)
        return [maxi[0] + X, maxi[1]]

    def viterbi(self):
        # return max path and its score
        [score, path], upd = theano.scan(fn=self.trans,
                                         outputs_info=[theano.tensor.ones_like(self.out[0]), None],
                                         sequences=[self.out],
                                         non_sequences=self.params['A'])

        maxscore, maxarg = theano.tensor.max_and_argmax(score[-1])
        return (maxscore, path, maxarg)

    def calc_answer_score(self):
        # return self.ans 's score
        def inner(x, OUT, prior, sumscore, W):
            return [x, sumscore + W[prior, x] + OUT[x]]
        res, upd = theano.scan(fn=lambda x, y: x + y,
                               outputs_info=None,
                               sequences=[self.params['A'].T],
                               non_sequences=theano.tensor.ones_like(self.out[0]))
        res_ = theano.tensor.max(res,axis=1)
        score1 = res_[self.ans[0].astype("int64")]
        startscore = score1 + self.out[0][self.ans[0].astype("int64")]
        [_, score], upd = theano.scan(fn=inner,
                                      outputs_info=[self.ans[0].astype("int64"), startscore],
                                      sequences=[self.ans[1:].astype("int64"), self.out[1:]],
                                      non_sequences=self.params['A'])
        return score[-1]

    def replacedata(self, inputdata, anslist):
        #inputdata: [ sentence1, sentence2,...]
        #sentence: [array1, array2,..]
        #ansdata: [array1, array2,...]
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
        print "Error:"
        print "type:", str(type(e))
        print self.getdata(x)
        print "args:", str(e.args)
        inp,ans =  self.getdata(x)
        print inp, ans
        layers = self.getlayers(x)
        print layers

    def training(self, data):
        #inputdata: [sentence1, sentence2,..]
        #sentence: [array1, array2,..]
        #ansdata: [array1, array2,...]
        def learn(x):
            inp, ans = self.getdata(x)
            out,path = self.comp(x)
        allsize = sum([len(sent) for sent,_ in data])
        allin = (allsize < self.bufferlen)
        if allin:
            inputbuf = [sent for sent,_ in data]
            ansbuf = [ans for _,ans in data]
            buflen = self.replacedata(inputbuf, ansbuf)
            for i in range(self.iter):
                for x in xrange(buflen):
                    learn(x)
        else:
            for i in range(self.iter):
                buffersum = 0
                inputbuf = []
                ansbuf = []
                for sent, ans in data:
                    if (len(sent) + buffersum) > self.bufferlen:
                        buflen = self.replacedata(inputbuf, ansbuf)
                        for x in xrange(buflen):
                            try:
                                learn(x)
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
                for x in xrange(buflen):
                    try:
                        learn(x)
                    except Exception as e:
                        self.error_handle(e)

    @staticmethod
    def trace_back(viterbipath, maxnode):
        p_ = list(viterbipath)
        p_.reverse()
        rev_path = [maxnode]
        for arr in p_:
            rev_path.append(arr[rev_path[-1]])
        rev_path.reverse()
        return rev_path

    @staticmethod
    def trace_back_(path, maxnode):
        rev_arange =  -1*theano.tensor.arange(-1*path.shape[0]+1, 1)
        path_rev, _ = theano.scan(fn=lambda i: path[i], sequences=[rev_arange])
        [_,path_ret], _ = theano.scan(fn = lambda x,i: [x[i], i],outputs_info=[maxnode, None], sequences=[path_rev])
        rev_arange =  -1*theano.tensor.arange(-1*path_ret.shape[0]+1, 1)
        ret, _ = theano.scan(fn=lambda i: path_ret[i], sequences=[rev_arange])
        return ret

    def test_(self, dataset):
        X = numpy.zeros((self.OL, self.OL))

        def acc_(out, ans):
            print a_, o_
            for a_, o_ in zip(list(ans), list(out)):
                X[o_, a_] += 1

        inputbuf = []
        ansbuf = []
        sum_ = 0
        buffersum = 0
        for sent, ans in dataset:
            if (len(sent) + buffersum) > self.bufferlen:
                buflen = self.replacedata(inputbuf, ansbuf)
                for x in xrange(buflen):
                    out, ans = self.getmaxpath(x)
                    acc_(out, ans)
                inputbuf = []
                ansbuf = []
                buffersum = 0
            else:
                inputbuf.append(sent)
                ansbuf.append(ans)
        buflen = self.replacedata(inputbuf, ansbuf)
        for x in xrange(buflen):
            try:
                out, ans = self.getmaxpath(x)
                acc_(out, ans)
            except Exception as e:
                print "Error:"
                print "type:", str(type(e))
                print "args:", str(e.args)
                print self.getdata(x)
        precision = numpy.average([(X[i] / sum(X[i])) for i in range(self.OL)])
        recall = numpy.average([(X[i] / sum(X.T[i])) for i in range(self.OL)])
        print X
        return (2 * precision * recall) / (recall + precision)

    def load_param(self, filename):
        setfunc = {}
        npzfile = numpy.load(filename)
        for key in npzfile.keys():
            variable = self.params_v[key]
            arr = self.params[key]
            sets = (arr, theano.tensor.set_subtensor(arr[:], variable))
            setfunc[key] = theano.function([variable], updates=[sets])
        for key, arr in npzfile.items():
            setfunc[key](arr)

