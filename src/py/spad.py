""" src.py.spad
"""
import numpy as np

def spad_timg(rates, eps=1e-6):
    return 1.0 / np.clip(rates, eps, None)

def spad_logtimg(rates, eps=1e-6):
    # TODO: see if this name is confusing?
    # mean(logtimgs) or log(meantimgs)?
    # and which one do we need?
    return np.log(spad_timg(rates, eps=eps))

def invert_spad_timg(timg, tmin=1e-3, tmax=1e6):
    return 1.0 / np.clip(timg, tmin, tmax)

def invert_spad_logtimg(logtimg, tmin=1e-3, tmax=1e6):
    return np.exp(-np.clip(logtimg, np.log(tmin), np.log(tmax)))

def sample_spad_timestamps(rates, N=1, tmin=0, tmax=np.inf, eps=1e-6,
                        avg_fn='AM'):
    scale = 1.0 / np.maximum(rates, eps)
    H, W = scale.shape
    scale = scale.reshape((1, H, W))
    # --------------------------------------------------------------------------
    # first we just simulate the arrival times
    # we let tmin be the minimum time period the sensor can sensibly (haha)
    # measure (basically = 1/2 of the clock period in some sense)
    times = tmin + np.random.exponential(scale=scale, size=(N, H, W))
    # --------------------------------------------------------------------------
    # there are 2 ways to take the readings:
    #   1.  take N independent readings, waiting at most tmax for each one, and
    #       average them later
    #   2.  start recording once, and wait for N photons to show up. the reading
    #       is the mean (arithmetic or geometric) of the inter-photon intervals.
    #
    #   i think the paper assumes approach 2, so i've kept that one active.
    #
    #   NOTE: tmax has a different meaning (and scale) in the two approaches
    # --------------------------------------------------------------------------
    # approach 1
    # --------------------------------------------------------------------------
    # T = np.clip(times, None, tmax).mean(axis=0)
    # --------------------------------------------------------------------------
    # approach 2
    # --------------------------------------------------------------------------
    # count the photons we actually got
    total_times = np.cumsum(times, axis=0)
    num_photons = (total_times <= tmax).sum(axis=0) # is a 2D array
    with np.errstate(divide='ignore', invalid='ignore'):
        if avg_fn == 'AM':
            if tmax is not None:
                times[total_times > tmax] = 0 # ignored in sum
            T = np.sum(times, axis=0) / num_photons
        elif avg_fn == 'GM':
            if tmax is not None:
                times[total_times > tmax] = 1 # to get it ignored in sum(log)
            T = np.exp(np.sum(np.log(times), axis=0) / num_photons)
        else:
            raise NotImplementedError
        T[num_photons == 0] = np.inf
    return T

