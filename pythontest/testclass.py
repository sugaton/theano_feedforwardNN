import Queue
import threading


class fech(threading.Thread):
    def __init__(self, parent):
        threading.Thread.__init__(self)
        self.parent = parent

    def run(self):
        print "===start sub thread="
        l = []
        for i in range(10):
            l.append([i, range(10)])
        self.parent.queue.put(l)


class iterator(object):
    def __init__(self, quesize=1000):
        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        self.exit_flag = False

    def start(self):
        self.queue = Queue.Queue(maxsize=self.quesize)
        self.gather = fech(self)
        self.gather.daemon = True
        self.gather.start()

    def __iter__(self):
        return self

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def next(self):
        batch = self.queue.get()
        if not batch:
            return None
        self.next_offset = batch[0]
        return batch[1], batch[2]

ite = iterator()
ite.start()
for i,j in ite:
    print i,j
