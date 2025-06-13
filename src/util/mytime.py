from timeit import default_timer as timer

def etime(f, n = 1, args=()):
    start = timer()
    for i in range(n):
        f(args)
    return timer() - start