import theano
import random
import time
import numpy
from collections import OrderedDict


N = 5
bufferlen = 500
chardiclen = 100
batchsize = 20
chard = 3
IL = chard
HL = 5
OL = 4
alfa = numpy.float32(0.02)
FLOAT = "float32"


def scn(idx, O1, O2, datas):
    idx0 = datas[idx][0]
    idx1 = datas[idx][1]
    idx2 = datas[idx][2]
    idx3 = datas[idx][3]
    return [O1[idx0:idx1], O2[idx2:idx3]]

# set datas
datas_idx = theano.shared(numpy.zeros((bufferlen, 4)).astype("int64"))
data = theano.shared(numpy.zeros((bufferlen, ), dtype="int64"), name="data")
data_ans = theano.shared(numpy.zeros((bufferlen, ), dtype="int64"), name="data_ans")
idxs = theano.tensor.lvector("idxs")
# set parameters
C = theano.shared(numpy.random.randn(chardiclen, chard), name="C")
W = theano.shared(numpy.random.randn(IL, HL), name="W")
params = [C, W]
alf = numpy.float32(0.01)


class network_():
    def __init__(self,idx, C, W):
        i, a = scn(idx, data, data_ans, datas_idx)
        self.S = batch_(i,a,C,W)

    @property
    def S(self):
        return self.S


def batch_(inp, ans, C, W):
    # def setlayer():
    # reference look-up
    def oneinput(inpidx):
        return theano.scan(fn=lambda i, x: x[i], sequences=[inpidx], non_sequences=[C])

    intensor, _ = oneinput(inp)
    h = theano.dot(intensor, W)  # 2d tensor
    softmax = theano.tensor.nnet.softmax

    # softmax for matrix
    def softmax_(mat):
        return theano.scan(fn=lambda x: theano.tensor.reshape(softmax(x), (x.shape[0],)), sequences=[mat])
    o, _ = softmax_(h)

    # argmax each row of input matrix
    def argmax_(mat):
        return theano.tensor.reshape(theano.tensor.argmax(mat, axis=1), (mat.shape[0], ))
    out = argmax_(o)
    dif = theano.tensor.neq(ans, out)
    aS = theano.tensor.sum(o[theano.tensor.arange(o.shape[0]), ans] * dif)
    oS = theano.tensor.sum(o[theano.tensor.arange(o.shape[0]), out] * dif)
    return aS - oS



# define network and learning function

# set network of each idx and define gradient
nets = range(batchsize)
E = range(batchsize)
for i in range(batchsize):
    nets[i] = network_(idxs[i], C, W)
    E[i] = nets[i].S
grad = theano.grad(theano.tensor.sum(E), params)
upd = [(p, p + alf * g) for p, g in zip(params, grad)]
# define learning function
learn = theano.function([idxs], updates=upd)

d = theano.tensor.lvector()
d_ans = theano.tensor.lvector()
d_idx = theano.tensor.lmatrix()
u = (data, theano.tensor.set_subtensor(data[:d.shape[0]], d))
u_ans = (data_ans, theano.tensor.set_subtensor(data_ans[:d_ans.shape[0]], d_ans))
u_idx = (datas_idx, theano.tensor.set_subtensor(datas_idx[:d_idx.shape[0]], d_idx))
set_ = theano.function([d, d_ans, d_idx], updates=[u, u_ans, u_idx])

#set data
size = 50
inpdata = [random.randint(0, 99) for i in range(size*6)]
ansdata = [random.randint(0, 3) for i in range(size*6)]
idx_data = [[6*(i-1), 6*i, 6*(i-1), 6*i] for i in range(1, size+1)]
set_(inpdata, ansdata, idx_data)
# print datas_idx.get_value()
# f = theano.function([idxs], out, updates=upd)
st = time.time()
for i in range(size / batchsize):
    start = i*batchsize
    end = (i+1)*batchsize if (i+1)*batchsize < size else size
    learn(range(50)[start:end+1])
ed = time.time()
print ed - st
# print getdata(range(5))
    # print f(range(5))
