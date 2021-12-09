"""doc
# kia_dataset.io.annotations

> Some annotations, that make loading the dataset more efficient and keep the code clean.

## Authors and Contributors
* Michael Fürst (DFKI), Lead-Developer
"""

import functools


def cache(size):
    """
    Make a function a cached function.
    
    :param size: The size of the cache to use. Defines for how many different parameters the results are cached.
    """
    def _create_cache_key(obj):
        cache_key = ""
        if isinstance(obj, list):
            for entry in obj:
                cache_key += _create_cache_key(entry)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                cache_key += _create_cache_key(key) + _create_cache_key(value)
        else:
            cache_key += str(hash(obj))
        return cache_key

    def _cache_wrapper(func):
        @functools.wraps(func)
        def _wrapper_cache(*args, **kwargs):
            # Create a key
            cache_key = ""
            for arg in args:
                cache_key += _create_cache_key(arg)
            for arg in kwargs.items():
                cache_key += _create_cache_key(arg)

            # If not in cache call function and cache it
            if cache_key not in _wrapper_cache.cache:
                # Make space in cache
                while len(_wrapper_cache.cache_order) >= size:
                    first_key = _wrapper_cache.cache_order.pop(0)
                    del _wrapper_cache.cache[first_key]
                # Cache call
                _wrapper_cache.cache[cache_key] = func(*args, **kwargs)
                _wrapper_cache.cache_order.append(cache_key)
                _wrapper_cache.misses += 1
            else:
                _wrapper_cache.hits += 1
            # Update cache order
            _wrapper_cache.cache_order.remove(cache_key)
            _wrapper_cache.cache_order.append(cache_key)
            #print("Cache Efficiency: {:.2f}% size={}".format(100 * _wrapper_cache.hits / (_wrapper_cache.hits + _wrapper_cache.misses), len(_wrapper_cache.cache_order)))

            # return cached value
            return _wrapper_cache.cache[cache_key]
        _wrapper_cache.cache = dict()
        _wrapper_cache.cache_order = list()
        _wrapper_cache.hits = 0
        _wrapper_cache.misses = 0
        return _wrapper_cache
    return _cache_wrapper
