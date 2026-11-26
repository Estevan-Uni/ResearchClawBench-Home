try:
    from numba import njit
except ImportError:  # pure-python fallback (debugging only)
    def njit(*a, **k):
        if a and callable(a[0]): return a[0]
        return lambda f: f
