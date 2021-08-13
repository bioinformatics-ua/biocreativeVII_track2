import inspect
import time
import collections

elapsed = collections.Counter()
counter = collections.Counter()
	
def start(timer = None):
	if timer is None:
		caller = inspect.stack()[1]
		timer = (caller.filename, caller.function)
	elapsed[timer] -= time.time()
	counter[timer] += 1

def stop(timer = None):
	if timer is None:
		caller = inspect.stack()[1]
		timer = (caller.filename, caller.function)
	elapsed[timer] += time.time()

def count(timer = None):
	if timer is None:
		caller = inspect.stack()[1]
		timer = (caller.filename, caller.function)
	counter[timer] += 1

def clear():
	elapsed.clear()
	counter.clear()

def log(logger):
	elapsed2 = [(e, t) for t, e in elapsed.items()]
	elapsed2.sort(reverse = True)
	for e, t in elapsed2:
		c = counter[t]
		avg = e / c;
		logger.debug("PERFORMANCE {} called {} times, elapsed time = {} ms, average time = {} ms".format(t, c, 1000 * e, 1000 * avg));
	count2 = [(c, t) for t, c in counter.items() if not t in elapsed]
	count2.sort(reverse = True)
	for c, t in count2:
		logger.debug("PERFORMANCE {} called {} times".format(t, c));
	