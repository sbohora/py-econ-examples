# simple value function iteration example: optimal investment/consumption problem
# utility function is log() and production function is k^alpha
from __future__ import division #use Python 3 division operation (Python 2 does integer division by default)
import numpy as np

#set parameters
A = 1
alpha = 0.3
beta = 0.6 #time discount factor
ab  = alpha * beta

# closed form for A=1: v(k) = c0 + c1*log(k)
c0 = 1/(1-beta) * ( ab/(1-ab) * np.log(ab) + np.log(1-ab) )
c1 = alpha / (1-ab)

#analytical solution
def v_star(k):
    return c0 + c1 * np.log(k)

#grid (domain) on which value function is defined
grid = np.array([0.04,0.08,0.12,0.16,0.2])

#intialize v_0 = 0
v0 = np.zeros(len(grid))

#instantiate v_star on grid
v = np.array([v_star(k) for k in grid])

def T_g(w):
    """
    Bellman Operator
    Input: a flat numpy array of same length as grid
    Returns one update of the value function iteration and associated policy function 
    """
    Tw = np.zeros(len(w))
    g = np.zeros(len(w))
    for i,k in enumerate(grid): #each (index,point) pair in the domain of the value function
        values = [] #this list will hold the values of the objective function for all k' which satisfy the constraint
        for j,kp in enumerate(grid): #values k' that the objective function can take
            if kp < A*k**alpha: #non-negative consumption constraint
                values.append(np.log(A*k**alpha - kp) + beta*w[j])
        Tw[i] = np.amax(values) #select the max 
        g[i] = grid[np.argmax(values)] # policy function: the maximizer k' corresponding to each k
    return Tw,g

def iterate_fixed_point(F_h,w,tol=10e-4,max_it=100):
    """
    Computes a fixed point in function space.
    Inputs: Bellman Operator that returns a (value function, policy function) pair, initial function guess
    Returns: converged (or max iterate) (value function, policy function) pair
    """
    Fw,h = F_h(w) # initialize (T_g returns a pair: value function, policy function)
    for k in range(max_it):
        Fw_new,h = F_h(Fw)
        error = np.max(np.abs(Fw_new-Fw))
        if error < tol:
            print "Complete: %d iterations" % (k+1)
            return Fw,h
        Fw = Fw_new
    print "Maximum number of iterations (%d) complete, max error = %f" % (max_it, error)
    return Fw,h

# run this code
Tv_star,g_star = T_g(v)
v1,g1 = T_g(v0)
v_approx,g_approx = iterate_fixed_point(T_g,v0)

print "Analitic solution v* = ",v
print "T(v*) = ",Tv_star
print "v1 = T(v0) = ",v1
print "Fixed point approximation of v*: ",v_approx
print "Fixed point approximation of policy function: ",g_approx
