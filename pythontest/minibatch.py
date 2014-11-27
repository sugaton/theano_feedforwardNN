import theano
import random
import time
import numpy
from collections import OrderedDict


N = 5
bufferlen = 500
chardiclen = 100
chard = 3
IL = chard
HL = 5
OL = 4
FLOAT = "float32"



def scn(idx, O1, O2, datas):
    idx0 = datas[idx][0]
    idx1 = datas[idx][1]
    idx2 = datas[idx][2]
    idx3 = datas[idx][3]
    return [O1[idx0:idx1], O2[idx2:idx3]]

datas_idx = theano.shared(numpy.zeros((bufferlen, 4)).astype("int64"))
data = theano.shared(numpy.zeros((bufferlen, ), dtype="int64"), name="data")
data_ans = theano.shared(numpy.zeros((bufferlen, ), dtype="int64"), name="data_ans")
idxs = theano.tensor.lvector("idxs")
[i, a], _ = theano.scan(fn=scn, sequences=[idxs], non_sequences=[data, data_ans, datas_idx])
inp = i.astype("int64")
ans = a.astype("int64")
getdata = theano.function([idxs], [inp,ans])
d = theano.tensor.lvector()
d_ans = theano.tensor.lvector()
d_idx = theano.tensor.lmatrix()
u = (data, theano.tensor.set_subtensor(data[:d.shape[0]], d))
u_ans = (data_ans, theano.tensor.set_subtensor(data_ans[:d_ans.shape[0]], d_ans))
u_idx = (datas_idx, theano.tensor.set_subtensor(datas_idx[:d_idx.shape[0]], d_idx))
set_ = theano.function([d, d_ans, d_idx], updates=[u, u_ans, u_idx])


# def setlayer():
C = theano.shared(numpy.random.randn(chardiclen, chard), name="C")
W = theano.shared(numpy.random.randn(IL, HL), name="W")
params = [C,W]
# reference look-up
def oneinput(inpidx):
    return theano.scan(fn=lambda i,x:x[i], sequences=[inpidx], non_sequences=[C])

intensor, _ = theano.scan(fn=oneinput, sequences=[inp])
h = theano.dot(intensor,W)  # 3d tensor
softmax = theano.tensor.nnet.softmax
# softmax for matrix
def softmax_(mat):
    return theano.scan(fn=lambda x: theano.tensor.reshape(softmax(x), (x.shape[0],)), sequences=[mat])

o,_ = theano.scan(fn=lambda x: softmax_(x), sequences=[h])
# argmax each row of input matrix
def argmax_(mat):
    return theano.tensor.reshape(theano.tensor.argmax(mat, axis=1), (mat.shape[0], ))

out,_ = theano.scan(fn=argmax_, sequences=[o])
# def grad():
def gradloop(o_, a_, tidx):
    gr, _ = theano.scan(fn=lambda i,j,k,y: theano.grad(o[tidx, k, i]-o[tidx, k, j],p),
                        outputs_info=[theano.tensor.zeros_like(p)],
                        sequences=[o_, a_,theano.tensor.arange(ans.shape[1])])
    return gr[-1]

ret = OrderedDict()
for p in params:
    gr, _ = theano.scan(fn=lambda o_, a_, tidx, y: gradloop(o_, a_, tidx)+y,
                        outputs_info=[theano.tensor.zeros_like(p)],
                        sequences=[out, ans, theano.tensor.arange(o.shape[0])])
    ret[p] = p + ((0.01 * gr[-1])).astype("float32")
upd = ret
# def grad():
size = 100
inpdata = [random.randint(0, 99) for i in range(size*6)]
ansdata = [random.randint(0, 3) for i in range(size*6)]
idx_data = [[6*(i-1), 6*i, 6*(i-1), 6*i] for i in range(1, size+1)]
set_(inpdata, ansdata, idx_data)
# print datas_idx.get_value()
f = theano.function([idxs], out, updates=upd)
st = time.time()
i=range(50)
a= f(i)
ed = time.time()
print ed - st
# print getdata(range(5))
# print f(range(5))
