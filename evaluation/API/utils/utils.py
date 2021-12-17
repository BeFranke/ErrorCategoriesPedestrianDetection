import collections.abc
from threading import Thread, Event

import numpy as np


def is_posix():
    try:
        import posix
        return True
    except ImportError:
        return False


def cross_product(x, y):
    cross_product = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    return cross_product


def function_attribute(key, value):
    """
    Decorator that enables the function that was decorated with it to have an attribute called "key" with value "value"
    key: str - desired name of the attribute (can be accessed as function.key)
    value: Any - initial value of the attribute
    """

    def decorator(f):
        setattr(f, key, value)
        return f

    return decorator


def update_recursive(dictionary, update):
    """
    update a dict of dicts recursively
    """
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            dictionary[k] = update_recursive(dictionary.get(k, {}), v)
        else:
            dictionary[k] = v
    return dictionary


class ReusableThread(Thread):
    """
    This class provides code for a restartale / reusable thread

    join() will only wait for one (target)functioncall to finish
    finish() will finish the whole thread (after that, it's not restartable anymore)
    Source : https://www.codeproject.com/Tips/1271787/Python-Reusable-Thread-Class
    """

    def __init__(self, target, args):
        self._startSignal = Event()
        self._oneRunFinished = Event()
        self._finishIndicator = False
        self._callable = target
        self._callableArgs = args

        Thread.__init__(self)

    def restart(self):
        """make sure to always call join() before restarting"""
        self._startSignal.set()

    def run(self):
        """ This class will reprocess the object "processObject" forever.
        Through the change of data inside processObject and start signals
        we can reuse the thread's resources"""

        self.restart()
        while True:
            # wait until we should process
            self._startSignal.wait()

            self._startSignal.clear()

            if self._finishIndicator:  # check, if we want to stop
                self._oneRunFinished.set()
                return

            # call the threaded function
            self._callable(*self._callableArgs)

            # notify about the run's end
            self._oneRunFinished.set()

    def join(self, **kwargs):
        """ This join will only wait for one single run (target functioncall) to be finished"""
        self._oneRunFinished.wait()
        self._oneRunFinished.clear()

    def finish(self):
        self._finishIndicator = True
        self.restart()
        self.join()
