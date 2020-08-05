#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support script with  useful functions for analysing LHE data and plotting histograms

@author: Joep Geuskens
"""

import numpy as np
import re
from matplotlib import pyplot as plt
import import_lhe as lhe

__combs = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])
__combs2 = np.array([[2,3], [1,3], [1,2], [0,3], [0,2], [1,0]])

def get_m(particles):
    p = particles[0].p
    for part in particles[1:]:
        p += part.p
    mm = p[0]*p[0]-p[1]*p[1]-p[2]*p[2]-p[3]*p[3]
    return np.sqrt(mm)

def get_pt(particles):
    pt = 0
    for p in particles:
        pt += p.pt
    return pt

def delta_R(p1, p2):
    deta = p1.eta-p2.eta
    dphi = np.clip(p1.phi-p2.phi, -np.pi, np.pi)
    return np.sqrt(deta**2+dphi**2)

def sort_particles(particles):
    idx = np.argsort(list([p.pt for p in particles]))
    return [particles[i] for i in idx]

def get_m2(p1, p2):
    p = [p1.E+p2.E, p1.px+p2.px, p1.py+p2.py, p1.pz+p2.pz]
    mm = p[0]*p[0]-p[1]*p[1]-p[2]*p[2]-p[3]*p[3]
    return np.sqrt(mm)

def group_particles(particles):
    if(len(particles)!=4):
        raise ValueError("Expected array of length 4, got {:d}".format(len(particles)))
    m_max = -np.inf
    idx = 0
    for i in range(len(__combs)):
        m = get_m2(particles[__combs[i][0]], particles[__combs[i][1]])
        if(m>m_max):
            m_max = m
            idx = i
    group1 = [particles[__combs[idx][0]], particles[__combs[idx][1]]]
    if(group1[0].pt<group1[1].pt):
        group1 = [group1[1], group1[0]]
    group2 = [particles[__combs2[idx][0]], particles[__combs2[idx][1]]]
    if(group2[0].pt<group2[1].pt):
        group2 = [group2[1], group2[0]]      
    return [group1, group2], m_max, get_m2(*group2)
    
def num_to_latex(txt):
    """
    Replaces all exponents with `\cdot10^{...}` using a regex.
    All occassions of which match `[number1]e[number2]` will be replaced by
    `[number1]\cdot10^{[number2]}`

    Parameters
    ----------
    txt : str
        a string containing one or more numbers in exponential notation.

    Returns
    -------
    str
        a string in which all exponents have been written in LaTeX.

    """
    matches = re.finditer(r"[\d.]+(e[+-]?[\d.]+)",txt)
    newtxt = ""
    lastindex = 0
    for match in matches:
        exp_text = txt[match.start(1)+1:match.end(1)]
        while(exp_text[0]=="+" or exp_text[0]=="0" and len(exp_text)>1):
            exp_text = exp_text[1:]
        newtxt += txt[lastindex:match.start(1)]+r"\cdot10^{"+ exp_text+"}"
        lastindex = match.end(1)
    return newtxt + txt[lastindex:]

def hist(x, bins=None, x_range=None, density=True, xlabel=None, ylabel=None, axes=None, figsize=None, insets=("mu","sigma","N"), inset_pos=(0.75,1), inset_props=None, unit="GeV", inset_fontsize=12,**kwargs):
    """
    Creates a histogram using pyplot.hist

    Parameters
    ----------
    x : array_like
        the data of which a histogram is created
    bins : integer or array_like, optional
        if not none, this is used as the bins argument in pyplot.hist. If given, the under/overflow of the data will be calculated
        and shown in the terminal/console. The default is None.
    xlabel : str, optional
        the x-axis label. The default is None.
    ylabel : str, optional
        the y-axis label. The default is None.
    axes : pyplot axis, optional
        if given, the histogram will be plotted in this axis, otherwise a new figure and axis will be created. The default is None.
    insets : type of str, optional
        the stats which should be shown in the inset. The default is ("mu","sigma","N").
    inset_pos : tuple, optional
        the position of the stats inset. The default is (0.68,1).
    inset_props : dict, optional
        a dictionary containing the bbox properties of the stats inset. The default is None.
    unit : str, optional
        the unit of used in the stats inset. The default is 'GeV'.

    **kwargs : 
        the keyword arguments which are passed on to pyplot.hist.

    Returns
    -------
    fig : pyplot figure
        the figure containing the plot if a new figure was created, otherwise this is None.
    axes : pyplot axes
        the axes of the plot.
    n : array or list of arrays
        the n return value of pyplot.hist.
    bins : array
        the bins return value of pyplot hist.
    patches : list or list of lists
        the patches return value of pyplot hist.
    """
    if(axes is None):
        if(figsize is not None):
            fig, axes = plt.subplots(figsize=figsize)
        else:
            fig, axes = plt.subplots()
    else:
        fig = None
    
    use_x_range=False
    if(bins is not None and isinstance(bins, (list,tuple,np.ndarray))):
        #calculate the under/overflow of the bins
        under = 0
        over = 0
        minimum = min(bins)
        maximum = max(bins)
        for value in x:
            if(value<minimum): under += 1
            elif(value>maximum): over += 1
        print("Histogram underflow={:d}, overflow={:d}".format(under,over))
    elif(x_range is not None and isinstance(x_range, tuple)):
        #calculate the under/overflow of the bins
        use_x_range=True
        under = 0
        over = 0
        for value in x:
            if(value<x_range[0]): under += 1
            elif(value>x_range[1]): over += 1
        print("Histogram underflow={:d}, overflow={:d}".format(under,over))
 
 
    n, bins, patches = axes.hist(x,bins=bins, density=density, **kwargs)   
    if(density==True):
        axes.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    axes.grid()
    if(use_x_range):
        axes.set_xlim(*x_range)
    
    if(xlabel is not None): axes.set_xlabel(xlabel)
    if(ylabel is not None): axes.set_ylabel(ylabel)
    if(insets is not None and len(insets)>0):
        insets_text = []
        if("mu" in insets):
            insets_text.append(r"$\mu={:.2e}$".format(x.mean())+unit)
        if("sigma" in insets):
            insets_text.append(r"$\sigma={:.2e}$".format(x.std(ddof=1))+unit)
        if("N" in insets):
            if(len(x)>1e5):
                d = len(x)/(10**int(np.log10(len(x))))
                if(int(d)==d):
                    insets_text.append(num_to_latex(r"$N={:.0e}$".format(len(x))))
                else:
                    insets_text.append(num_to_latex(r"$N={:.2e}$".format(len(x))))
            else:
                insets_text.append(r"$N={:d}$".format(len(x)))
        inset_text = num_to_latex("\n".join(insets_text))
        if(inset_props is None):
            inset_props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
        # place a text box in upper right in axes coords
        axes.text(*inset_pos, inset_text, transform=axes.transAxes, fontsize=inset_fontsize, verticalalignment='bottom', horizontalalignment='left', bbox=inset_props)
    if(fig is None):
        return n, bins, patches
    return fig, axes, n, bins, patches

def is_sorted(x, order):
    if(order=='asc'):
        for i in range(len(x)-1):
            if(x[i]>x[i+1]):
                return False
        return True
    elif(order=='desc'):
        for i in range(len(x)-1):
            if(x[i]<x[i+1]):
                return False
        return True
    else:
        raise ValueError("Unknown order (expected 'asc' or 'desc'): " + str(order))

types = (list, tuple, np.ndarray)
class Cuts:
    def __init__(self, x, x_min=-np.infty, x_max=np.infty, topdown=False):
        if(isinstance(x_max, types) and not isinstance(x_min, types)):
            x_min = [x_min]*len(x_max)
        elif(isinstance(x_min, types) and not isinstance(x_max,types)):
            x_max = [x_max]*len(x_min)
        
        if(isinstance(x_min, types) and isinstance(x_max, types)):
            if(len(x_min) != len(x_max)):
                raise ValueError("Sizes of the lower and upper bounds of the cuts do not match.")
            if not(is_sorted(x_min, 'asc') and is_sorted(x_max, 'desc')):
                raise ValueError("Cut arrays are not correctly sorted!")
            self._n = len(x_min)
            f = np.zeros if not topdown else np.full
            param = dict()
            if(topdown): param["fill_value"]=int(2**self._n-1)
            i = np.log2(64)-3
            if(i>3):
                raise ValueError("Maximum number of cuts supported is 64, got {:d}".format(self._n))
            param["dtype"]=[np.int8, np.int16, np.int32, np.int64][int(i)]
            self._selections = f(len(x), **param)
            if not topdown:
                r = np.arange(self._n)
                masks = 1 << r
                for j, val in enumerate(x):
                    for i in r: #start with the loosest cut and continue until the value falls outside it
                        if(val >= x_min[i] and val <= x_max[i]):
                            self._selections[j] ^= masks[i] #turn this bit on
                        else:
                            #if the value falls outside this cut, it'll also fall outside the other cuts
                            break
            else:
                r = np.arange(self._n-1,-1,-1)
                masks = 1 << r[::-1]
                for j, val in enumerate(x):
                    for i in r: #start with the tightest cut and continue until the value falls inside it
                        if(val < x_min[i] or val > x_max[i]):
                            self._selections[j] ^= masks[i] #turn this bit off
                        else:
                            #if the value falls inside this cut, it'll also fall inside the other cuts
                            break
        else:
            self._n = 0
            self._selections = (x>=x_min) & (x<=x_max)
    
    def __getitem__(self, key):
        key = self.__check_indices(key)          
        arr = (self._selections & (1<<key)) >> key
        #arr.astype(np.bool_, copy=False)
        return arr
    
    def __check_indices(self, idx):
        if(self._n==0 and idx==0):
            return self._selections
        if(isinstance(idx, (tuple,list))):
            idx = np.array([[idx]], dtype=int).T
        if(isinstance(idx, np.ndarray) and issubclass(idx.dtype.type,np.integer)):
            if(len(idx.shape)!=1):
                raise IndexError("index should be 1D, got {:d}D".format(len(idx.shape)))
            idx = idx.reshape(len(idx),1)
            a = np.where(idx>=self._n)[0]
            if(len(a)!=0):
                raise IndexError("index {:d} is out of bounds for array with size {:d}".format(a[0], self._n))
        elif(isinstance(idx,int) or isinstance(idx, np.integer)):
            if(idx>=self._n):
                raise IndexError("index {:d} is out of bounds for array with size {:d}".format(idx, self._n))
        else:
            raise IndexError("expected integer or array of integers as indices")
        return idx
    
    def __compare(self, idx, arr, arr_shift=None):
        if(arr_shift is None):
            return ((self._selections & (1<<idx)) >> idx) & arr
        return ((self._selections & (1<<idx)) >> idx) & ((arr & (1<<arr_shift)) >> arr_shift)
    
    def compare(self, arr, idx1=None, idx2=None):         
        if(type(arr) is Cuts):
            if(idx1 is None and self._n != 0):
                idx1 = np.arange(self._n)
            if(idx2 is None and arr._n != 0):
                idx2 = np.arange(arr._n)
            if(idx1 is not None): idx1 = self.__check_indices(idx1)
            if(idx2 is not None): idx2 = arr.__check_indices(idx2)
            
            if(idx1 is None and idx2 is None):
                return self._selections & arr._selections
            
            if(idx1 is None): idx1 = 0
            if(idx2 is None): idx2 = 0
            if(isinstance(idx1,types) and not isinstance(idx2, types)):
                arr = (arr & (1<<idx2)) >> idx2
                result = np.empty((len(idx1), len(self._selections)), dtype=np.bool_)
                for i in range(len(result)):
                    result[i] = np.bitwise_and(np.right_shift(np.bitwise_and(self._selections, 1<<idx1[i], dtype=np.int8), idx1[i], dtype=np.int8), arr, dtype=np.int8)
                return result
            else:
                return np.bitwise_and(np.right_shift(np.bitwise_and(self._selections, 1<<idx1, dtype=np.int8), idx1, dtype=np.int8),
                                      np.right_shift(np.bitwise_and(arr, 1<<idx2, dtype=np.int8),idx2, dtype=np.int8), dtype=np.int8)
        elif(type(arr) is np.ndarray and (issubclass(arr.dtype.type, np.integer) or issubclass(arr.dtype.type,np.bool_))):
            if(idx1 is None):
                if(self._n==0): idx1 = 0
                else: idx1 = np.arange(self._n)
            else:
                idx1 = self.__check_indices(idx1)
            if(isinstance(idx1, types)):
                result = np.empty((len(idx1), len(self._selections)), dtype=np.bool_)
                for i in range(len(result)):
                    result[i] = np.bitwise_and(np.right_shift(np.bitwise_and(self._selections, 1<<idx1[i], dtype=np.int8), idx1[i], dtype=np.int8), arr, dtype=np.int8)
                return result
            else:
                return np.bitwise_and(np.right_shift(np.bitwise_and(self._selections, 1<<idx1, dtype=np.int8),idx1, dtype=np.int8), arr, dtype=np.int8)

    def __len__(self):
        return self._n
    
def make_cuts(x, x_min=-np.infty, x_max=np.infty, logical=False):
    if(isinstance(x_max, types) and not isinstance(x_min, types)):
        x_min = [x_min]*len(x_max)
    elif(isinstance(x_min, types) and not isinstance(x_max,types)):
        x_max = [x_max]*len(x_min)
    
    if(isinstance(x_min, types) and isinstance(x_max, types)):
        if(len(x_min) != len(x_max)):
            raise ValueError("Sizes of the lower and upper bounds of the cuts do not match.")
        if not(is_sorted(x_min, 'asc') and is_sorted(x_max, 'desc')):
            raise ValueError("Cut arrays are not correctly sorted!")
        #everything is ok
        r = range(len(x_min))
        if not logical:
            counts = np.zeros(len(x_min))
            for val in x:
                for i in r:
                    if(val >= x_min[i] and val <= x_max[i]):
                        counts[i] += 1
                    else:
                        #if the value falls outside this cut, it'll also fall outside the other cuts
                        break
            return counts
        else:
            selections = np.zeros((len(x_min), len(x)), dtype=np.bool_)
            for j, val in enumerate(x):
               for i in r:
                   if(val >= x_min[i] and val <= x_max[i]):
                       selections[i][j] = 1
                   else:
                       #if the value falls outside this cut, it'll also fall outside the other cuts
                       break
            return selections
    else:
        if not isinstance(x, np.ndarray): x = np.array(x)
        if not logical:
            return np.sum((x>=x_min) & (x<=x_max))
        else:
            return np.bitwise_and(x>=x_min, x<=x_max, dtype=np.int8)

def make_cut(data, cut):
    """
    Selects all entries in `data` which pass througha filter specified by `cut`
    
    `return list(filter(cut, data))`

    Parameters
    ----------
    data : array_like
        array containing the data on which the cut should be applied.
    cut : lambda
        a lambda expression which returns a boolean for each entry in `data`.

    Returns
    -------
    list
        the data which is left after the cut has been applied.
        
    Notes
    -----
    This method should not be used for numpy arrays, as those have (much) more efficient algorithms

    """
    return list(filter(cut, data))

_z_id = lhe.particle_id("z")
def zz_check(selected):
    zz_count = 0
    for event in selected:
        """
        Check if the event contains a ZZ pair
        """
        zz = event.get_particles(_z_id)
        if(len(zz)==2):
            zz_count += 1
        elif(len(zz)!=0):
            pass#print("Event contains {:d} Z's".format(len(zz)))
    return zz_count

_w_ids = lhe.particle_id(["w+","w-"])
def ww_check(selected):
    ww_count = 0
    for event in selected:
        """
        Check if the event contains a WW pair
        """
        ww = event.get_particles(_w_ids)
        if(len(ww)==2):
            ww_count += 1
        elif(len(ww)!=0):
           pass#print("Event contains {:d} W's".format(len(ww)))
    return ww_count
