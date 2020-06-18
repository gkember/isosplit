#!/usr/bin/env python
# coding: utf-8

# ## Isosplit
# ### Jeremy Magland 2015
# #### Unsupervised, non-parametric labeller
# #### Code ported to python March 20-29, 2020, Kember/Sudarshan.

# In[1]:
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import distance_matrix
from scipy.spatial import distance
import math


# #### initial parcelate code
# ##### code algorithm: changed to stop splitting below min parcel size (bug: Maglund: Matlab 2015)

# In[2]:


def parcelate(X, min_parcel_size, max_parcels, split_factor, verbose=False) :
    """
    Parcelation
    -----------
    Created by: Jeremy Maglund
    Ported to python: March, 2020: Sudarshan/Kember

    input: X = [n_pts, dim]: rectangular numpy array
    max_parcels: maximum number of parcels (or clusters)
    min_parcel_size: parcels split down to minimum parcel size

    dynamic vars
    ------------
    parcel_n: current number of parcels: outer while loop
    this_parcel_index: index of current parcel: inner while loop

    parcels: unreachable entries 9 (int), -9.0 (float)
    -------
    parcel_indices: static: [max_parcels, n_pts] == [label-1 (for 0 start index), n_pts]
    parcel_centroids: static: [max_parcels, dim]
    parcel_radii: static: [max_parcels,]

    work vars:
    ----------
    crash_out: fail safe enforce split_factor >= min_cluster_size
    verbose: output verbosity
    this_parcel_index: index of current parcel
    parcel_n: parcel number (not index)
    something_changed: have self and points not self

    output: labels: static: [n_pts,]

    """
    #max number of parcels
    if verbose: print('max_parcels ',max_parcels)

    #min parcel size
    if verbose: print('min_parcel_size ',min_parcel_size)

    #split factor: typically split factor < min_parcel_size
    if verbose: print('split_factor ', split_factor)
        
    #must NOT have split_factor >= min_parcel_size
    #fail safe is embedded in code to prevent inadvertent override
    crash_out = False
    if split_factor >= min_parcel_size :
        crash_out = True
        print('********Violation of split_factor >= min_parcel_size*******',split_factor, min_parcel_size)
        
    if verbose: print('*************************')
    if verbose: print('*******START: INIT*******')
    if verbose: print('*************************')

    #init number of points n_pts and dimension dim
    [n_pts, dim] = np.shape(X)
    if verbose: print('n_pts, dim',n_pts, dim)

    #init number of parcels
    parcel_n = 1

    #init current parcel
    this_parcel_index = 0

    #init labels
    labels = np.full(n_pts, -9)

    #get indices of parcel (all points for init)
    parcel_indices = np.full([max_parcels, n_pts], -9)
    parcel_indices[this_parcel_index] = np.arange(n_pts)
    if verbose: print('indices: this_parcel_index, this_parcel indices ',this_parcel_index, parcel_indices[this_parcel_index])

    #get centroid of initial parcel (all points for init)
    parcel_centroids = np.full([max_parcels, dim],-9.0)
    parcel_centroids[this_parcel_index] = np.mean(np.transpose(X),axis=1)
    if verbose: print('centroids: this_parcel_index, this_parcel_centroid ',this_parcel_index, parcel_centroids[this_parcel_index])

    #get radius of a parcel (all points for init)
    parcel_radii = np.full(max_parcels, -9.0)
    parcel_radii[this_parcel_index] = np.max(np.sqrt(np.sum((X - parcel_centroids[this_parcel_index])**2,axis=1)))
    if verbose: print('radii: this_parcel_index, this_parcel_radius ',this_parcel_index, parcel_radii[this_parcel_index])

    if verbose: print('num parcels max parcels ',parcel_n, max_parcels)

    if verbose: print('**********************')
    if verbose: print('*******END INIT*******')
    if verbose: print('**********************')

    while parcel_n < max_parcels and not crash_out :
        if verbose: print('OUTSIDE WHILE LOOP: parcel_n max_parcels', parcel_n, max_parcels)
        if verbose: print('OUTSIDE WHILE LOOP EXECUTION: depends on size of split_factor reduction')

        #global number of indices in each parcel
        parcel_sizes = np.sum(parcel_indices > -9,axis=1)
        parcel_sizes = parcel_sizes[np.nonzero(parcel_sizes > 0)[0]]
        if verbose: print('sizes ',parcel_sizes)

        #global check if cannot split
        if np.sum(parcel_radii > 0.0) and np.sum((parcel_sizes > min_parcel_size)) == 0 :
            break

        if verbose: print('parcel radii ', parcel_radii)
        # set target radius ~ largest parcel radius exceeding min_parcel_size
        target_radius = np.max(parcel_radii[np.nonzero(parcel_sizes > min_parcel_size)[0]]) * 0.95
        if verbose: print('target radius ',target_radius)

        # step through each parcel
        something_changed = False

        #start at zero index parcel
        this_parcel_index = 0

        while this_parcel_index < parcel_n and not crash_out :
            if verbose: print('\t INSIDE WHILE LOOP this_parcel_index < num_parcels ', this_parcel_index, parcel_n)
            if verbose: print('\n split factor reduction: overwrite 1 parcel: create split_factor - 1 NEW PARCELS')

            #list of parcel indices
            inds = parcel_indices[this_parcel_index][np.nonzero(parcel_indices[this_parcel_index] > -9)[0]]
            if verbose: print('\t inds ', inds)

            #single value: number of indices in parcel
            parcel_size = len(inds)
            if verbose: print('\t size', parcel_size)

            #single value
            rad = parcel_radii[this_parcel_index]
            if verbose: print('\t this radius ',rad)

            #see if need to split
            if parcel_size > min_parcel_size and rad >= target_radius and not crash_out :

                if verbose: print('\t \t IF: parcel_size > minparcel and rad >= target_radius', parcel_size, min_parcel_size, rad, target_radius)
                # the parcel has more than the tnarget number of points and is larger than the target radius                
                iii = np.arange(split_factor).tolist()
                if verbose: print('\t \t iii is ',iii)

                #distance from split_factor list -> to -> all members: dm = [split_factor, inds members]
                dm = distance_matrix(X[inds[iii], :], X[inds, :])
                if verbose: print('\t \t dm \n',dm)


                #find indices of parcel members closest to test set inds[iii]
                assignments = np.argmin(dm, axis=0)
                if verbose: print('\t \t assignments ', assignments)

                # make sure not everything got assigned to 1 and there were self = 0 distances
                if np.sum(assignments > 0) > 0 and np.sum(assignments == 0) > 0 and not crash_out :
                    if verbose: print('\t \t \t some self but not everything: so do split ')

                    if verbose: print('\t \t \t START: assign == 0 ')

                    #is a need to split
                    something_changed = True
                    if verbose: print('\t \t \t **SOMETHING CHANGED**  ',something_changed)

                    #over write this_parcel_index with those parcel locations closest to self
                    this_parcel_indices = inds[np.nonzero(assignments == 0)[0]]

                    #overwrite whole parcel as -9 for rewrite
                    parcel_indices[this_parcel_index] = -9
                    parcel_indices[this_parcel_index][this_parcel_indices] = this_parcel_indices
                    if verbose: print('\t \t \t \t this parcel this_parcel_indices ',this_parcel_index, parcel_indices[this_parcel_index])

                    #reset centroid
                    parcel_centroids[this_parcel_index] = np.mean(np.transpose(X)[:, this_parcel_indices],axis=1)
                    if verbose: print('\t \t \t \t this parcel centroid ',this_parcel_index, parcel_centroids[this_parcel_index])

                    #reset radius
                    parcel_radii[this_parcel_index] = np.max(np.sqrt(np.sum((X[this_parcel_indices, :] - parcel_centroids[this_parcel_index])**2,axis=1)))
                    if verbose: print('\t \t \t \t this parcel radii ',this_parcel_index, parcel_radii[this_parcel_index])

                    #reset label
                    labels[this_parcel_indices] = this_parcel_index + 1
                    if verbose: print('\t \t \t \t labels ', labels)

                    if verbose: print('\t \t \t END: assign == 0 ')

                    #create split_factor - 1 NEW PARCELS due to split assignments
                    if verbose: print('\t \t \t START FOR LOOP: NEW PARCEL: assign ~= 0 ')
                    if verbose: print('\t \t \t global inds ', inds)
                    for jj in np.arange(1, split_factor) :
                        next_inds = inds[np.nonzero(assignments == jj)[0]]
                        if verbose: print('\t \t \t \t assignments jj parcel_n next inds ',jj, parcel_n, next_inds)
                        parcel_indices[parcel_n][next_inds] = next_inds
                        parcel_centroids[parcel_n] = np.mean(np.transpose(X)[:, next_inds],axis=1)
                        if verbose: print('\t \t \t \t jj centroids ', parcel_centroids[parcel_n])
                        parcel_radii[parcel_n] = np.max(np.sqrt(np.sum((X[next_inds, :] - parcel_centroids[parcel_n])**2,axis=1)))
                        if verbose: print('\t \t \t \t jj radii ', parcel_radii[parcel_n])
                        #set number of parcels - after the above - indexing is 1 lower
                        parcel_n = parcel_n + 1
                        labels[next_inds] = parcel_n

                        if parcel_n >= max_parcels:
                            #print('\t \t \t \t ******PARCELATE: BAD APPEND******: split_factor <= max_parcels')
                            crash_out = True
                            break

                        if verbose: print('\t \t \t \t jj labels',jj, labels)                    

                    if verbose: print('\t \t \t DONE FOR LOOP: NEW PARCEL: assign ~= 0 ')

                else :
                    warning('There was an issue splitting a parcel. This could result from the same point being included multiple times in the data.');

            else :
                this_parcel_index = this_parcel_index + 1
                if verbose: print('\t size and radius ok: go to next parcel **SOMETHING CHANGED**  ', this_parcel_index, something_changed)

        if not something_changed :
            if verbose: print('\t could not split **SOMETHING CHANGED** ', something_changed)
            break

    return labels


# # isosplit/cut routines ported to python

# ### jisotonic regression python version

# In[3]:


def jisotonic5(A, weights, verbose=False) :
    """
    jisotonic5: isotonic regression cases
   
    A: input vector
        
    weights: optional A element weight vec
    
    B: output vector
    
    MSEs: external work var 'updown/downup'
    """

    if verbose: print('*******jisotonic5: weights******* ',weights)
    #init
    N = len(A)
    
    unwct = np.full(N, -9)
    count = np.full(N, -9.0)
    sumit = np.full(N, -9.0)
    sumsq = np.full(N, -9.0)
    
    
    #init: [unweightedcount, count, sum, sumsqr]
    #use 5 letter names for algorithm symmetry [unwct, count, sumit, sumsq]
    last = 0
    unwct[last] = 1
    count[last] = weights[0]
    sumit[last] = A[0] * weights[0]
    sumsq[last] = A[0] * A[0] * weights[0]

    MSEs=np.full(N, 0.0)

    #
    for j in np.arange(1, N) :
        #
        last = last + 1 
        #
        unwct[last] = 1
        #
        count[last] = weights[j]
        #
        sumit[last] = A[j] * weights[j]
        #
        sumsq[last] = A[j] * A[j] * weights[j]
        #
        MSEs[j] = MSEs[j-1]

        if verbose: print('for: update: last, count[last] ', last, count[last])
        
        #
        while True :
            #
            if last <= 0 :
                break

            #
            prevMSE = sumsq[last-1] - np.square(sumit[last-1]) / count[last-1] + sumsq[last] - np.square(sumit[last]) / count[last]

            #
            if sumit[last-1] / count[last-1] < sumit[last] / count[last] :
                break
                
            #
            else :

                #
                unwct[last-1] = unwct[last-1] + unwct[last]
                count[last-1] = count[last-1] + count[last]
                sumit[last-1] = sumit[last-1] + sumit[last]
                sumsq[last-1] = sumsq[last-1] + sumsq[last]
                if verbose: print('for: update count[last-1] ', count[last-1])
                #
                newMSE = sumsq[last-1] - sumit[last-1] * sumit[last-1] / count[last-1]

                #
                last = last - 1

                #
                MSEs[j] = MSEs[j] + newMSE - prevMSE

    #
    B = np.full(N, 0.0)
    
    #
    ii = 0

    #inclusive on last: last + 1
    for k in np.arange(0, last + 1) :
    
        #self average
        for cc in np.arange(0, unwct[k]) :
        
            #
            B[ii+cc] = sumit[k] / count[k]
        
        #update ii after self average complete
        ii = ii + unwct[k]

    return B, MSEs


# ### python: call jisotonic5 regression: each case separate: no recursion

# In[4]:


def call_jisotonic5(A, direction, weights, verbose=False) :
    """
    call_jisotonic5: isotonic regression cases

    jisotonic5(A, direction)
    
    A: input vector
    
    direction: 'increasing', 'decreasing', 'updown', or 'downup'
    
    weights: optional A element weight vec
    
    B is the output vector (same size as A)

    MSEs: 'updown' and 'downup'
    """
    N = len(A)

    #set default if needed
    if weights is None: 
        weights = np.ones(N)
    
    if direction == 'increasing' :
        if verbose: print('direction ', direction)
        [B, _] = jisotonic5(A, weights, verbose)
        return B
    #
    elif direction == 'decreasing' :
        if verbose: print('direction ', direction)
        
        #flip sign of A
        [B, _] = jisotonic5(-A, weights, verbose)
        
        #flip sign of B
        return -B
    #
    elif direction == 'updown' :
        if verbose: print('direction ', direction)

        #sign of A unchanged

        #A increasing
        [B1, MSE1] = jisotonic5(A, weights, verbose)
        
        #flip A increasing
        [B2, MSE2] = jisotonic5(np.flip(A), np.flip(weights), verbose)
        
        #flip B2
        B2 = np.flip(B2)
        
        #flip MSE2
        MSE2 = np.flip(MSE2)
        MSE0 = MSE1 + MSE2
        
        #MSE1 and MSE2: monotonic increasing/decreasing
        #sum(MSE1 + MSE2) possesses a global min
        #set breakpoint between up-down
        best_ind = np.argmin(MSE0)

        #match matlab inclusive
        best_ind = best_ind + 1
        
        #fit up
        [C1, _] = jisotonic5(A[0:best_ind],weights[0:best_ind], verbose)
        
        #fit down: decreasing: flip sign of A: share min
        [C2, _] = jisotonic5(-A[best_ind-1:],weights[best_ind-1:], verbose)
        
        #flip sign of C2
        C2 = -C2
        
        #
        B = np.concatenate([C1[0:best_ind],C2[1:]], axis=0)
        if np.isnan(B[0]) :
            print('WARNING: downup: jisotonic5: NaN')
        
        #sign of B unchanged
        return B
        
    #
    elif direction == 'downup' :
        if verbose: print('direction ', direction)

        #flip sign of A: then repeat updown approach
        A = -A
        
        #repeat updown as downup

        #A increasing
        [B1, MSE1] = jisotonic5(A, weights, verbose)
        
        #flip A increasing
        [B2, MSE2] = jisotonic5(np.flip(A), np.flip(weights), verbose)
        
        #flip B2
        B2 = np.flip(B2)
        
        #flip MSE2
        MSE2 = np.flip(MSE2)
        MSE0 = MSE1 + MSE2
        
        #MSE1 and MSE2: monotonic increasing/decreasing
        #sum(MSE1 + MSE2) possesses a global min
        #set breakpoint between up-down
        best_ind = np.argmin(MSE0)

        #match matlab inclusive
        best_ind = best_ind + 1
        
        #fit up
        [C1, _] = jisotonic5(A[0:best_ind],weights[0:best_ind], verbose)
        
        #fit down: decreasing: flip sign of A: share min
        [C2, _] = jisotonic5(-A[best_ind-1:],weights[best_ind-1:], verbose)
        
        #flip sign of C2
        C2 = -C2
        
        #
        B = np.concatenate([C1[0:best_ind],C2[1:]], axis=0)
        if np.isnan(B[0]) :
            print('WARNING: downup: jisotonic5: NaN')
        
        #flip sign of B on return since doing downup
        return -B        
    #
    else :
        print('WARNING: bad direction: call_jisotonic5 ')
        return np.array([])
        
    
    return


# ### compute ks5

# #### python computeks5 and ks

# In[5]:


def compute_ks5(counts1, counts2, verbose=False) :

    N1 = len(counts1)
    best_ks = -np.Inf
    while  N1 >= 4 or N1 == len(counts1):
        ks = np.max(np.abs(np.cumsum(counts1[:N1])/np.sum(counts1[:N1]) - np.cumsum(counts2[:N1])/np.sum(counts2[:N1]))) *                 np.sqrt((np.sum(counts1) + np.sum(counts2)) / 2)
        if ks > best_ks :
            best_ks = ks
            best_N1 = N1
        N1 = math.floor(N1/2)
    return best_ks, best_N1


# ### isocut 5 python version

# In[6]:


def isocut5(samples, sample_weights, num_bins_factor=1, already_sorted=False, verbose=False) :
    
    ks5_verbose=False
        
    N = len(samples)
    if N == 0 :
        print('WARNING: error in isocut5: N is zero.')
        dip_score = -9.0
        cutpoint = -9.0
        return dip_score, cutpoint
    
    num_bins = math.ceil(np.sqrt(N/2) * num_bins_factor)
    if verbose: print('num_bins ',num_bins)
    
    if already_sorted :
        X = samples
    else :
        sort_inds = np.argsort(samples)
        X = samples[list(sort_inds)]
        sample_weights = sample_weights[list(sort_inds)]
    
    if verbose: print('X sorted ', X)
    if verbose: print('sample weights ', sample_weights)
    while 1 :
        num_bins_1 = math.ceil(num_bins/2)
        num_bins_2 = num_bins - num_bins_1
        if verbose: print('numbin 1 2 ',num_bins_1, num_bins_2)
        #intervals=[1:num_bins_1,num_bins_2:-1:1]
        intervals = np.concatenate([np.arange(num_bins_1)+1, np.flip(np.arange(num_bins_2))+1])
        if verbose: print('intervals ',intervals)
        if verbose: print('*****intervals sum*****', np.sum(intervals))
        alpha = (N-1)/np.sum(intervals)
        if verbose: print('alpha ',alpha)
        intervals = intervals*alpha
        if verbose: print('intervals ', intervals)
        inds = 1.0 + np.cumsum(intervals)
        inds = np.floor(np.insert(inds, 0, 1.0, axis=0)).astype(int)
        if verbose: print('inds ',inds)
        N_sub = len(inds)
        if verbose: print('N_sub ',N_sub)
        if np.min(intervals) >= 1 :
            break
    else :
        num_bins = num_bins - 1
        
    cumsum_sample_weights = np.cumsum(sample_weights)
    if verbose: print('cumsum sample weights ',cumsum_sample_weights)
    X_sub = X[inds - 1]
    if verbose: print('X_sub ', X_sub)
    spacings = X_sub[1:] - X_sub[:-1]
    if verbose: print('spacings ', spacings)
    mults = cumsum_sample_weights[list(inds[1:] - 1)] - cumsum_sample_weights[list(inds[:-1] - 1)];
    if verbose: print('multiplicities ', mults)
    densities = np.divide(mults, spacings)
    if verbose: print('densities ', densities)

    densities_unimodal_fit = call_jisotonic5(densities, 'updown', mults)
    if verbose: print('densities_unimodal fit ',densities_unimodal_fit)
        
    peak_density_ind = np.argmax(densities_unimodal_fit)
    if verbose: print('peak_density_ind ',peak_density_ind)
    if verbose: print('mults left call [: peak_density_ind]',mults[:peak_density_ind + 1])
    [ks_left,ks_left_index] =            compute_ks5(mults[:peak_density_ind + 1],                                np.multiply(densities_unimodal_fit[:peak_density_ind + 1],                                            spacings[:peak_density_ind + 1]), ks5_verbose                   )
    if verbose: print('ks left left_index',ks_left, ks_left_index)
    if verbose: print('mults right call [: peak_density_ind]',np.flip(mults)[:len(mults)-peak_density_ind])
    [ks_right, ks_right_index] =            compute_ks5(np.flip(mults)[:len(mults)-peak_density_ind],                        np.multiply(np.flip(densities_unimodal_fit)[:len(mults)-peak_density_ind],                                np.flip(spacings)[:len(mults)-peak_density_ind]), ks5_verbose                   )
    ks_right_index = len(spacings) - ks_right_index + 1
    if verbose: print('ks right right_index',ks_right, ks_right_index)

    
    if ks_left > ks_right :
        if verbose: print('left > right ')
        critical_range = np.arange(ks_left_index)
        dip_score = ks_left
    else :
        if verbose: print('left <= right len(spacings) ',len(spacings))
        critical_range = np.arange(len(spacings) -(ks_right_index - 1)) + (ks_right_index - 1)
        dip_score = ks_right
    if verbose: print('dip critical range ',dip_score, critical_range)
    
    densities_resid = densities - densities_unimodal_fit
    if verbose: print('densities_resid ',densities_resid)
    if verbose: print('dens_resid[crit range]',densities_resid[critical_range])
    densities_resid_fit = call_jisotonic5(densities_resid[critical_range],'downup',spacings[critical_range])
    if verbose: print('dens_resid_fit ',densities_resid_fit)
    cutpoint_ind = np.argmin(densities_resid_fit)
    if verbose: print('cutpoint_ind ',cutpoint_ind)
    cutpoint_ind = critical_range[0] + cutpoint_ind
    if verbose: print('cutpoint_ind ',cutpoint_ind)
    cutpoint = (X_sub[cutpoint_ind] + X_sub[cutpoint_ind + 1])/2
    if verbose: print('cutpoint ',cutpoint)
    
    if verbose: print('dip_score cutpoint',dip_score, cutpoint)

    return dip_score, cutpoint


# ## compute_centers

# ### compute centers python version

# In[7]:


def compute_centers(X, labels) :
    
    """
    input:  
        X: X[n_pts, dim]
        labels: labels(1,n_pts)
        
    output: 
        centers[dim, n_pts]: arithmetic average of labelled points    
    """
    
    #shape of X
    [n_pts, dim] = np.shape(X)
    
    #unique labels
    unique_labels = np.unique(labels)
    
    #number unique labels
    num_unique_labels = len(unique_labels)
    
    #centers same shape as X
    centers = np.full([n_pts,dim], -9.0)
    
    #loop through labels: these labels are not indices: start 1, end has +1
    for this_label in np.arange(num_unique_labels) :
        #obtain indices where labels 
        inds = np.argwhere(labels == unique_labels[this_label])
        #store center label col: as index -1: must transpose for broadcast
        centers[unique_labels[this_label] - 1] = np.mean(np.transpose(X)[:, inds],axis=1).transpose()
    return centers.transpose()


# ## get_pairs_to_compare

# ### get pairs to compare python version

# In[8]:


def get_pairs_to_compare(centers, comparisons_made, verbose=False) :

    """
    algorithm: find mutually close pairs and recursively eliminate
    centers: cluster center coordinates
    comparisons_made: labels compared
    
    dim: dimension
    n_centers: number of centers
    
    inds1: close index static
    inds2: mutually close index static
    
    """

    [dim, n_centers] = centers.shape
    inds1 = np.full(n_centers, -9)
    inds2 = np.full(n_centers, -9)
    pair_dists = np.full(n_centers, -9.0)

    #
    dists = distance.cdist(centers.transpose(), centers.transpose(), 'euclidean')
    #GCK
    dists[np.nonzero(comparisons_made > 0)] = np.Inf
    #
    np.fill_diagonal(dists,np.Inf)
    if verbose: print('\t \t get pairs to compare: comparisons made \n',comparisons_made)
    if verbose: print('\t \t get pairs to compare: dists \n',dists)
    
    best_inds = np.argmin(dists, axis=0)
    
    for j in np.arange(n_centers) :
        if best_inds[j] > j :
            if best_inds[best_inds[j]] == j :
                if dists[j, best_inds[j]] < np.Inf :
                    inds1[j] = j
                    inds2[j] = best_inds[j]
                    pair_dists[j] = dists[j,best_inds[j]]
                    dists[j,:] = np.Inf
                    dists[:,j] = np.Inf
                    dists[best_inds[j],:] = np.Inf
                    dists[:,best_inds[j]] = np.Inf
    #GCK                
    return inds1[np.nonzero(inds1 > -9)[0]], inds2[np.nonzero(inds2 > -9)[0]]


# ## whiten two clusters

# #### whiten two clusters python version

# In[9]:


def whiten_two_clusters_b(X1, X2) :

    #set up dim and n_pts1 and n_pts2
    [dim, n_pts1] = X1.shape
    [_, n_pts2] = X2.shape
    
    #build centroids [1, dim] each
    centroid1 = np.mean(X1,axis=1).reshape(1,dim)
    centroid2 = np.mean(X2,axis=1).reshape(1,dim)

    #subtract centroid.transpose: [dim, 1]
    X1_centered = X1 - centroid1.transpose()
    X2_centered = X2 - centroid2.transpose()

    #get covariance matrix
    C1 = np.matmul(X1_centered, X1_centered.transpose()) / n_pts1
    C2 = np.matmul(X2_centered, X2_centered.transpose()) / n_pts2
    #get average covariance matrix
    avg_cov = (C1 + C2) / 2
    
    #get position vector
    V = centroid2 - centroid1
    #print('V is ',V)
    
    #check for same vectors
    if np.linalg.norm(V) < 1e-10 :
        V[0][0] = V[0][0] + 1e-10
        print('whiten_clusters_b: WARNING: same vectors')

    #skew position vector: toward direction covariance
    if np.abs(np.linalg.det(avg_cov))>1e-6 :
        inv_avg_cov = np.linalg.inv(avg_cov)
        V = np.matmul(V, inv_avg_cov)
    V = V / np.sqrt(np.matmul(V, V.transpose()))
    
    return V


# ## merge_test

# ### merge test python version

# In[10]:


def merge_test(X1_in, X2_in, isocut_threshold, L2_eps, do_whiten_clusters=False, verbose_merge_test=False) :
    
    verbose = verbose_merge_test

    #copy inputs: [dim, n_pts]: **already transposed from [n_pts, dim]**
    X1 = X1_in
    X2 = X2_in
    if verbose: print('X1 ',X1)
    if verbose: print('X2 ',X2)

    if do_whiten_clusters :
        #average out cluster variance: adjust V
        V = whiten_two_clusters_b(X1, X2)
        if verbose: print('V from whiten clusters ',V)

    else :
        #build centroids [1, dim] each
        centroid1 = np.mean(X1,axis=1)
        centroid2 = np.mean(X2,axis=1)
        V = centroid2 - centroid1;
        V = V/(np.sqrt(np.matmul(V.transpose(), V)) + L2_eps)
        if verbose: print('V NOT from whiten clusters ',V)
        

    #number points in X1,2: [dim, n_pts]
    n1_pts = X1.shape[1]
    n2_pts = X2.shape[1]
    
    if n1_pts == 0 or n2_pts == 0 :
        print('Warning: merge_test: N1 or N2 is zero')
    
    #projection: [1, n_pts] = X[dim, n_pts] * {V.transpose = [dim, 1]}
    if verbose: print('\t \t \t \t X1 V.transpose', X1, V.transpose())
    projection1 = np.matmul(V, X1)
    if verbose: print('\t \t \t \t projection 1 ',projection1)
    projection2 = np.matmul(V, X2)
    if verbose: print('\t \t \t \t projection 2 ',projection2)
    projection12 = np.concatenate([projection1.flatten(), projection2.flatten()], axis=0)
    
    if verbose: print('\t \t \t \t projection12 ',projection12)
    
    #
    [dipscore,cutpoint] = isocut5(projection12,np.ones(len(projection12)))
    #
    if verbose: print('\t \t \t \t dipscore was ', dipscore)
    do_merge = (dipscore < isocut_threshold)
    #
    new_labels = np.full(n1_pts + n2_pts, 1)
    #
    new_labels[list(np.nonzero(projection12 >= cutpoint)[0])] = 2;
    
    return do_merge, new_labels, dipscore


# ## compare_pairs

# ### compare pairs python version

# In[11]:


def compare_pairs(X, labels, k1s, k2s, min_cluster_size, isocut_threshold, L2_eps, do_whiten_clusters=False, verbose_compare_pairs=False, verbose_merge_test=False) :

    verbose = verbose_compare_pairs

    #init
    dipscore = -9.0
    #
    clusters_changed_vec = np.full(np.max(labels), 0)
    #
    new_labels = labels

    #
    for i1 in np.arange(len(k1s)) :

        #select a label (as index = as is) from one cluster
        k1 = k1s[i1]

        #select a label (as index = as is) from one cluster
        k2 = k2s[i1]
        
        #tuple[0] -> **row** vec: as label +1
        inds1 = np.nonzero(labels == k1 + 1)[0]
        inds2 = np.nonzero(labels == k2 + 1)[0]
        if verbose: print('inds 1 2', inds1, inds2)
      
        #
        if len(inds1)>0 and len(inds2)>0 :
            if len(inds1) < min_cluster_size or len(inds2) < min_cluster_size :
                if verbose : print('below min size so do merge')
                do_merge = True
            else :
                if verbose : print('run merge test ')
                #
                inds12 = np.concatenate([inds1, inds2], axis=0)
                #
                L12_old = np.concatenate([np.ones(len(inds1)),2*np.ones(len(inds2))], axis=0)
                #send transposed X samples
                [do_merge, L12, dipscore] = merge_test(X.transpose()[:,inds1],                                             X.transpose()[:,inds2],                                             isocut_threshold,                                             L2_eps,                                             do_whiten_clusters,                                             verbose_merge_test                                            )

            if do_merge :
                if verbose : print('do merge labels k2+1 -> k1+1 ', k2+1, k1+1)
                #new_labels[list(np.locations(new_labels is k2))[array]]: k1 + 1 as label
                new_labels[list(np.nonzero(new_labels == k2 + 1)[0])] = k1 + 1

                #identify locations with new labels
                clusters_changed_vec[k1] = 1
                clusters_changed_vec[k2] = 1

            else :
                #redistribute
                if verbose : print('redistribute ')
                if verbose : print('redistribute: do merge labels to k1+1 and k2+1 ', k1+1, k2+1)
                #new_labels(inds12(find(L12==1)))=k1; k1 + 1 as label
                new_labels[list(inds12[list(np.nonzero(L12 == 1)[0])])] = k1 + 1

                #new_labels(inds12(find(L12==2)))=k2; k2 + 1 as label
                new_labels[list(inds12[list(np.nonzero(L12 == 2)[0])])] = k2 + 1

                if len(np.nonzero(L12 != L12_old)[0]) > 0 :
                    #identify locations with new labels
                    clusters_changed_vec[k1] = 1
                    clusters_changed_vec[k2] = 1

    #return clusters_changed row vec
    clusters_changed = np.nonzero(clusters_changed_vec)[0]
    
    return new_labels, clusters_changed, dipscore


# ### generate isosplit data
# #### code is matched on matlab side for code testing using identical data

# In[12]:


def generate_isosplit_data(num_dimensions=2) :
    np.set_printoptions(precision=2)
    #testdata 1
    #X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1], [11, 11], [11, 9], [9, 11], [9, 9]])
    
    #testdata 2
    #i_stride = 3
    #j_stride = 3
    #X = np.full([2,9],-9.0)
    #for i in np.arange(0,3) :
    #    for j in np.arange(0,3) :
    #        X[0, i*j_stride + j] = i*0.1
    #        X[1, i*j_stride + j] = j*0.1
    #X = np.concatenate((X.transpose(),X.transpose()+1),axis=0)
    
    #set seed
    #trial = 1
    #N0 = 1000
    #np.random.seed(trial)

    #box-muller transform uniform -> standard normal
    #X11 = np.random.rand(N0); X12 = np.random.rand(N0);
    #X21 = np.random.rand(N0); X22 = np.random.rand(N0);
    #size 10*N0
    #Z11 = np.multiply(np.sqrt(-2*np.log(X11)),np.cos(2*np.pi*X21))
    #size 10*N0
    #Z12 = np.multiply(np.sqrt(-2*np.log(X11)),np.sin(2*np.pi*X21))
    #size 3*N0
    #Z21 = np.multiply(np.sqrt(-2*np.log(X12)),np.cos(2*np.pi*X22))
    #size 3*N0
    #Z22 = np.multiply(np.sqrt(-2*np.log(X12)),np.sin(2*np.pi*X22))
    
    #cat clusters
    #X_coord = np.concatenate([Z11-5, Z21+5, Z11+5, Z21-5])
    #X_coord = X_coord.reshape(len(X_coord),1)
    #Y_coord = np.concatenate([Z12, Z22, Z12+3, Z22+3])
    #Y_coord = Y_coord.reshape(len(Y_coord),1)
    #X = np.concatenate([X_coord, Y_coord + 6.0123], axis=1)
    
    #read input file
    X = np.loadtxt('x_multimodal_nd.txt',usecols=range(0,num_dimensions), dtype=np.float64)
    
    return X


# In[14]:
#freq-based labelling
def freq_based_label(labels) :

    label_list = np.unique(labels)
    label_freq = [np.sum(labels == j) for j in label_list]
    freq_label = np.flip(np.argsort(label_freq))
    for j in np.arange(len(label_list)) :
        labels[np.nonzero(labels == label_list[j])[0]] = freq_label[j]
    
    return labels

# In[15]:
def plot_X_labels(X, labels) :
    colors = cm.rainbow(np.linspace(0.0, 1.0, len(np.unique(labels))))
    x=X[:,0]
    y=X[:,1]
    #fig = plt.figure(figsize=(8,8))
    plt.scatter(x,y, c=labels, cmap=matplotlib.colors.ListedColormap(colors), s=1)
    plt.axis('equal')
    cb = plt.colorbar()
    loc = np.arange(0,max(labels),max(labels)/float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(colors)
    plt.show()

#iso_split main

# In[13]:
parser = argparse.ArgumentParser(description='Variable mapping: default value, Exceptions if:\
                                 split_factor >= min_cluster_size,\
                                 n_pts < 1.25 * max_parcels,\
                                 max_parcels <= 2 * min_parcel_size')

parser.add_argument("--a", default=10, type=int, help="a: min_cluster_size: def=10")
parser.add_argument("--b", default=3, type=int, help="b: split_factor: def=3")
parser.add_argument("--c", default=200, type=int, help="c: max_parcels : def=200")
parser.add_argument("--d", default=1.0, type=float, help="d: isocut_threshold: def=1.0")
parser.add_argument("--e", default=500, type=int, help="e: max_iterations_per_pass: def=500")
parser.add_argument("--f", default=2, type=int, help="f: number of dimensions in data: def=2")

args = parser.parse_args()

#dump vars
min_cluster_size = args.a
split_factor = args.b
max_parcels = args.c
isocut_threshold = args.d
max_iterations_per_pass = args.e
num_dimensions = args.f

#print vars to be sure
#print('min_cluster_size ',min_cluster_size)
#print('split_factor ',split_factor)
#print('max_parcels ',max_parcels)
#print('isocut_threshold ',isocut_threshold)
#print('max_iterations_per_pass ',max_iterations_per_pass)

#iso_split main
#build fake data
X = generate_isosplit_data(num_dimensions=num_dimensions)

[n_pts, dim] = X.shape

#parameters
#min_cluster_size = 10
min_parcel_size = min_cluster_size
#max_parcels = 200
#split_factor = 3
main_verbose=False
verbose_merge_test=False
#isocut_threshold = 1.0
L2_eps = 0.0

pass_iterations = 0
#max_iterations_per_pass = 500

final_pass = False



#check bounds
if split_factor >= min_cluster_size :
    print('Exception: split_factor >= min_cluster_size ',split_factor, min_cluster_size)
    sys.exit()
#check bounds
if n_pts < 1.25 * max_parcels :
    print('Exception: n_pts < 1.25 * max_parcels ',n_pts, max_parcels)
    sys.exit()
#check bounds
if max_parcels <= 2 * min_parcel_size :
    print('Exception: max_parcels <= 2 * min_parcel_size',max_parcels, min_parcel_size)
    sys.exit()



if main_verbose: print('n_pts ,dim',n_pts, dim)

#get labels
labels = parcelate(X, min_parcel_size, max_parcels, split_factor)
if main_verbose: print('unique labels ',np.unique(labels))

#number of unique labels
Kmax = np.max(labels)

#set labels as indices
labels_as_indices = np.unique(labels) - 1

#get centers
centers = compute_centers(X, labels)

#init comparisons made
data_comparisons_made = np.full([Kmax, Kmax], 0)

#Passes
while 1 :
    #
    pass_iterations = pass_iterations + 1
    #
    if pass_iterations > max_iterations_per_pass :
        break
        
    #
    something_merged = False

    #track changed clusters for update comparisons_made
    clusters_changed_vec_in_pass = np.full(Kmax, 0)
    
    #
    iteration_number = 0

    #Iterations
    while 1 :

        #
        iteration_number = iteration_number + 1
        if main_verbose: print('iteration ',iteration_number)
        
        #init **row vec**
        active_labels_vec = np.full(Kmax, 0)
        if main_verbose: print('\t #####active_labels#####')
           
        #set active_labels_vec[labels_as_indices] is not **row vec**
        active_labels_vec[np.unique(labels) - 1] = 1
        if main_verbose: print('\t active labels VEC ',active_labels_vec)
                
        #active_labels: tuple[0]: are indices
        active_labels = np.nonzero(active_labels_vec == 1)[0]
        if main_verbose: print('\t active_labels ', active_labels)
        
        #active centers **row vec** = select active centers[**row vec**]
        if main_verbose: print('\t current centers ', centers)        
        active_centers = centers[:, active_labels]
        if main_verbose: print('\t active_centers ', active_centers)

        if main_verbose: print('\t #####CALL GET PAIRS TO COMPARE#####')
        #data_comparisons[Kmax, Kmax] = comparisons_made[select **row vec**][:,select cols **vec**]
        [inds1, inds2] = get_pairs_to_compare(                                              active_centers,                                              data_comparisons_made[active_labels][:,active_labels],                                              verbose=False                                             )
        if main_verbose: print('\t RESULT: inds1 inds2 ',inds1, inds2)
        if len(inds1) == 0 :
            #nothing else to compare
            if main_verbose: print('\t #####ITERATIONS##### PUNCH OUT inds1 something_merged final pass',inds1, something_merged, final_pass)
            break
                   
        if main_verbose: print('\t #####CALL COMPARE PAIRS#####')
        #finish call to isocut in merge_test and open up merge_test in compare_pairs
        if main_verbose: print('\t BEFORE: labels ', labels)
        [labels, clusters_changed, dipscore] = compare_pairs(\
                                                                X,\
                                                                labels,\
                                                                active_labels[inds1],\
                                                                active_labels[inds2],\
                                                                min_cluster_size,\
                                                                isocut_threshold,\
                                                                L2_eps,\
                                                                do_whiten_clusters=True,\
                                                                verbose_compare_pairs=False,\
                                                                verbose_merge_test=False\
                                                            )

        if main_verbose: print('\t RESULT: AFTER: labels clusters changed', labels, clusters_changed)
        clusters_changed_vec_in_pass[clusters_changed] = 1
        
        #update which comparisons have been made
        for j in np.arange(len(inds1)) :
            data_comparisons_made[active_labels[inds1[j]],active_labels[inds2[j]]] = 1
            data_comparisons_made[active_labels[inds2[j]],active_labels[inds1[j]]] = 1
        if main_verbose: print('\t ###################')
        if main_verbose: print('\t ITERATIONS: active_labels inds 1 2', active_labels[inds1], active_labels[inds2])
        if main_verbose: print('\t ITERATIONS: clusters_changed ',clusters_changed)
        if main_verbose: print('\t ITERATIONS: data comparisons \n',data_comparisons_made)
        if main_verbose: print('\t ###################')

        #recompute the centers -- for those that changed and those that did not change
        centers = compute_centers(X, labels)
        if main_verbose: print('\t RESULT: from compute centers: ', centers)

        #determine whether something has merged
        if len(np.unique(labels)) < len(np.unique(active_labels)) :
            something_merged = True
        
        if iteration_number > max_iterations_per_pass :
            break

        if main_verbose: print('\t ****** SOMETHING MERGED ****** ', something_merged)
     
    #zero changed clusters: comparisons made matrix
    #find changed cluster indices
    clusters_changed = np.nonzero(clusters_changed_vec_in_pass)[0]
    #zero row
    data_comparisons_made[clusters_changed, :] = 0
    #zero col
    data_comparisons_made[:, clusters_changed] = 0
    if main_verbose: print('###################')
    if main_verbose: print('PASSES: clusters_changed ',clusters_changed)
    if main_verbose: print('PASSES: data comparisons \n',data_comparisons_made)
    if main_verbose: print('###################')

    #ensure that get one last pass: order required in last 3 lines
    #if something merged: final_pass set for one more pass
    if something_merged : 
        final_pass = False

    #if final_pass: done
    if final_pass : 
        break
    
    #if done: one last pass for final redistributes
    if not something_merged :
        final_pass = True
            
    if main_verbose: print('dipscore is ', dipscore)   

#map labels to frequency based labels    
labels = freq_based_label(labels)

#plot X and labels
plot_X_labels(X, labels)


