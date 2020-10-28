# NB only use for odd n
def n_root(t, n):
    if (t < 0):
        return -n_root(-t, n)
    else:
        return t ** (1. / n)

nframes = 300
k = 7.
for i in range(nframes):
    t = float(i) / nframes
    s = .5 + (n_root(t - .5, k) / (2. * n_root(.5, k)))
    print("t = {}, s = {}".format(t,s))