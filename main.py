import feedforwardNN
import random
import sys


def crossval(data):
    datalen = len(data)
    test = []
    for i in range(datalen / 10):
        test.append(data.pop(random.randint(0, len(data)-1)))
    return test


def main(*args):
    filename = args[1]
    data = []
    with open(filename, 'r') as fin:
        for line in fin:
            if len(line.split()) > 25:
                continue
            buf = line.rstrip().split(',')
            ans = [float(buf[0])]
            inp = map(lambda x: float(x), buf[1].split())
            data.append((ans, inp))
    test = crossval(data)
    network = feedforwardNN.feedforwardNN(IL=19, HL=20, OL=1)
    network.training(data)
    network.test(test)

if __name__ == "__main__":
    main(*sys.argv)
