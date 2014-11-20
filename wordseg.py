import feedfowardNN
import math
import numpy
import theano
from collections import OrderedDict

FLOAT = theano.config.floatX


class segmenter(feedfowardNN.feedfowardNN):
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
        self.idx = theano.tensor.lscalar('idx')

        self.setData()
        self.layers = [self.local_in]
        self.setlookup(V=chardiclen, d=char_d)

        self.setlayers()
        self.out = self.layers[-1]
        self.maxscore, self.maxpath = self.viterbi()
        self.answerscore = self.calc_answer_score()

        self.E = self.answerscore - self.maxscore
        self.gradients = self.set_grad()
        self.comp = theano.function([self.idx], self.E, updates=self.gradients)
        self.getmaxpath = theano.function([self.idx], self.maxpath)

    def setData(self):
        self.datas_idx = theano.shared(numpy.zeros((bufferlen, 3)))
        self.data = theano.shared(numpy.zeros((bufferlen, chardiclen), dtype=FLOAT), name="data")
        self.data_ans = theano.shared(numpy.zeros(bufferlen, OL))
        self.inp = self.data[self.datas_idx[self.idx][0]:self.datas_idx[self.idx][1]]
        self.ans = self.data_ans[self.datas_idx[self.idx]]

        d = theano.tensor.matrix()
        d_ans = theano.tensor.matrix()
        d_idx = theano.tensor.matrix()
        u = (self.data, theano.tensor.set_subtensor(self.data, d))
        u_ans = (self.data_ans, theano.tensor.set_subtensor(self.data_ans, d_ans))
        u_idx = (self.data_ans, theano.tensor.set_subtensor(self.data_ans, d_idx))
        self.set = theano.function([d, d_ans, d_idx], updates=[u, u_ans, u_idx])

    def settransition(self):
        arr = numpy.random.randn(self.OL, self.OL) / numpy.sqrt(2 * self.OL)
        A = theano.shared(arr.astype(FLOAT), name="A")
        self.params["A"] = A

    def setlookup(self, V=1, d=1):
        arr = numpy.random.randn(V, d) / numpy.sqrt(V + d)
        C = theano.shared(arr.astype(FLOAT), name="C")
        inputlayer = theano.dot(self.inp, C)
        self.layers.append(inputlayer)
        self.params["C"] = C

    @staticmethod
    def getfunc(st):
        def softmax_(X):
            result, updates = theano.scan(fn=lambda x: theano.tensor.nnet.softmax(x),
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
                               sequences=[theano.tensor.arange(2, sent_len-2)],
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
        self.parmas[name1] = w
        self.parmas[name2] = b

    def setlayers(self):
        self.setlayer(size=(self.N - 1 * self.IL, self.HL), name1="W1", name2="b1", act=self.hid_act)
        self.setlayer(size=(self.HL, self.OL), name1="W2", name2="b2", act=self.out_act)

    @staticmethod
    def trans(X, prior, W):
        def maxpath(pri, W):
            res, upd = theano.scan(fn=lambda x, y: x + y,
                                   outputs_info=None,
                                   sequences=[W.T],
                                   non_sequences=pri)

        maxi = maxpath(prior, W)
        return [maxi[0] * X, maxi[1]]

    def viterbi(self):
        [score, path], upd = theano.scan(fn=self.trans,
                                         outputs_info=[theano.tensor.ones_like(self.out[0]), None],
                                         sequences=[self.out],
                                         non_sequences=self.params['A'])
        return (score, path)

    def calc_answer_score(self):
        def inner(x, OUT, prior, sumscore, W):
            return [x, sumscore + W[prior, x] + OUT[x]]
        score1 = theano.dot(theano.tensor.ones_like(self.out[0]), self.params['A'])[self.ans[0]]
        startscore = score1 + self.out[0][self.ans[0]]
        [_, score], upd = theano.scan(fn=inner,
                                      outputs_info=[self.ans[0], startscore],
                                      sequences=[self.ans[1:], self.out],
                                      non_sequences=self.params['A'])
        return score[-1]

    def replacedata(self, inputdata, ansdata):
        #inputdata: [ sentence1, sentence2,...]
        #sentence: [array1, array2,..]
        #ansdata: [array1, array2,...]
        datarr = []
        idxarr = []
        for idx, data in enumerate(inputdata):
            if len(idxarr) > 0:
                lastidx = idxarr[-1][1]
            else:
                lastidx = 0
            datarr.append(data)
            idxarr.append([lastidx, lastidx+len(data), idx])
        datarr = numpy.array(datarr).astype(FLOAT)
        ansdata = numpy.array(ansdata).astype('int64')
        idxarr = numpy.array(idxarr).astype('int64')
        self.set(datarr, ansdata, idxarr)
        return len(idxarr)

    def training(self, inputdata, ansdata):
        #inputdata: [sentence1, sentence2,..]
        #sentence: [array1, array2,..]
        #ansdata: [array1, array2,...]
        for i in range(self.iter):
            buffersum = 0
            inputbuf = []
            ansbuf = []
            for sent, ans in zip(inputdata, ansdata):
                if (len(sent) + buffersum) > self.bufferlen:
                    buflen = self.replacedata(inputbuf, ansbuf)
                    for x in xrange(buflen):
                        self.comp(x)
                    inputbuf = []
                    ansbuf = []
                    buffersum = 0
                else:
                    inputbuf.append(sent)
                    ansbuf.append(ans)
            buflen = self.replacedata(inputbuf, ansbuf)
            for x in xrange(buflen):
                self.comp(x)
