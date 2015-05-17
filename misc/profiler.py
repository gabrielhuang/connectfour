# -*- coding: utf-8 -*-
"""
@brief: Profiler
@author: Gabriel Huang
"""
from time import clock

class Profiler:
    def __init__(self):
        self.times = {}
        self.last_time = None
        self.last_name = None
    def tic(self, name):
        now = clock()
        if self.last_time is not None:
            self.times[self.last_name] = self.times.get(self.last_name, 0.) + now - self.last_time
        self.last_time = now
        self.last_name = name
    def toc(self):
        self.tic('__toc__')
    def __repr__(self):
        acc = ['Profiler report:']
        sorted_times = sorted(self.times.items(), key=lambda (name,time):time, reverse=True)
        for name, time in sorted_times:
            if name!='__toc__':
                acc.append('{} seconds for {} '.format(time, name))
        return '\n'.join(acc)
        
if __name__=='__main__':
    profiler = Profiler()
    profiler.tic('step 1')
    for i in xrange(1000000): pass
    profiler.tic('step 2')
    for i in xrange(10000000): pass
    profiler.tic('step 3')
    for i in xrange(100000000): pass
    profiler.toc()
    print profiler