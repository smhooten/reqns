#!/usr/bin/env python
"""
Provides useful functions and variables that can be used for parallel
operations.
"""

from math import pi, ceil
import copy
import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

__author__ = "Sean Hooten"
__license__ = "BSD-2-Clause"
__version__ = "0.1"
__maintainer__ = "Sean Hooten"
__status__ = "development"

#def parallel_partition(num_sims, size, rank):
#    # Returns curated 1D partition for each processor given the total number of
#    # simulations required
#
#    simdisc = num_sims / size
#    extra = num_sims % size
#
#    partition = range(simdisc*rank, simdisc*(rank+1))
#
#    if rank < extra:
#        partition.append(simdisc*(rank+1))
#
#    partition = [x+rank if rank<extra else x+extra for x in partition]
#
#    print('Node {0} partition:'.format(rank))
#    print partition
#
#    return partition

def parallel_partition(num_sims):
    # Returns curated 1D partition for each processor given the total number of
    # simulations required

    simdisc = num_sims / SIZE
    extra = num_sims % SIZE

    partition = range(simdisc*RANK, simdisc*(RANK+1))

    if RANK < extra:
        partition.append(simdisc*(RANK+1))

    partition = [x+RANK if RANK<extra else x+extra for x in partition]

    print('Node {0} partition:'.format(RANK))
    print partition

    return partition

def create_master_array(num_sims, *args):
    # After gathering arrays from parallel simulation, collect them into a
    # master numpy array rather than a list. Multidimensional numpy array
    # created with multiple arguments

    # include check for num_sims??

    if RANK == 0:
        master_array = np.zeros((num_sims, len(args)))
        for i, arg in enumerate(args):
            tmp = True
            for element in arg:
                if tmp:
                    m_array = np.copy(element)
                    tmp = False
                else:
                    m_array = np.concatenate((m_array, element), axis=0)

            master_array[:,i] = m_array

    else:
        master_array = None

    return master_array

##############
## NOT WORKING DO NOT USE
##############

class SimulationManager(object):
    def __init__(self, reqns, LED_option, params, results=None):
        # LED_option allowed strings : 'both' , 'LED' , 'nanoLED'

        # params is a dictionary of inputs to the reqns object
        # Allowed keywords : 'dc_carrier_concentration', 'current'
        
        # item associated with 'dc_carrier_concentration' is another dictionary
        # must contain keywords : 'time', 'ns', 'I_guess',
        # t_start' and 'tolerance' optional

        # 'time' : 2 element array of start and end time
        # 'ns' : array of desired dc_carrier_concentrations
        # 'I_guess' : 2 element array bounding smallest and largest carrier concentrations

        # 't_start' : time to start measurement of n as fraction of t[1] (default: 0.95)
        # 'tolerance' : relative tolerance between desired n and meaured n (default: 1.0e-4)


        # item associated with 'current' is another dictionary
        # must contain keywords: 'option'
        # if 'option' : 'TCSPC' is specified, must contain another keyword: 'Is'
        # 'Is' is either an array of current values or string 
        # 'use dc carrier concentration' in which case
        # 'dc_carrier_concentration' must have been specified in params
        
        # if 'option' : 'I_func' is specified, must contain another keyword: 'Is'
        # 'Is' is either an array of current values or string 'use dc carrier
        # concentration' which will be plugged into reqns.I_func. 
        # Thus, reqns.I_func must be specified in the reqns
        # object as lambda t,I: current_function(t, I)

        #### IN DEVELOPMENT
        # if 'option' : 'I_func list' in which a list of lambda functions can
        # be directly specified, no list of currents required
        ####

        # results is a list of desired results (beyond n and t)
        # Allowed words: 'Rspon', 'rspon', 'QE'


        self.reqns = reqns
        self.rank = RANK-1

        if self.rank == -1:
            self.size = SIZE-1
            self.LED_option = LED_option
            self.params = params
            self.results = results
            self.slave = False
        else:
            self.size = None
            self.LED_option = LED_option
            self.params = None
            self.results = None
            self.slave = True

            self.__run = False
            self.__initi = True


    def run(self):
        if self.slave:
            self.__run = True
            while self.__run is True:
                self.wait_for_signal()
                if self.__run is False:
                    break

                if self.__option == 'n_dc':
                    reqns = self.reqns
                    t = reqns.t
                    max_step = reqns.max_step
                    n = self.__n
                    I_guess = self.__I_guess
                    LED_option = self.LED_option
                    tol = self.__tol
                    t_start = self.__t_start

                    Iout, _ = reqns.find_desired_Idc(t, max_step, n, I_guess,
                                        option=LED_option, tol=tol,
                                        t_start=t_start)

                    req = COMM.isend((Iout, self.__ind), dest=0, tag = 7)

                elif self.__option == 'current':
                    pass

        else:
            for pkey, pvalue in self.params.iteritems():
                if pkey == 'dc_carrier_concentration':
                    self.send_to_all('n_dc',0)

                    for dckey, dcvalue in pvalue.iteritems():
                        if dckey == 'time':
                            self.send_to_all(dcvalue, 1)

                        elif dckey == 'I_guess':
                            self.send_to_all(dcvalue, 2)

                        elif dckey == 'ns':
                            self.__ns = dcvalue

                        elif dckey == 't_start':
                            self.send_to_all(dcvalue, 3)

                        elif dckey == 'tolerance':
                            self.send_to_all(dcvalue, 4)

                    self.send_to_all(True, 5)

                    Is = self.ndc_routine()
                    self.Is = Is
                    print Is

                elif pkey == 'current':
                    pass

    def wait_for_signal(self):
        if self.__initi:
            self.__initi = False
            req = COMM.irecv(source = 0, tag = 0)
            procedure = req.wait()
            self.__option = procedure

            if procedure == 'n_dc':
                flag = False
                self.__option = procedure

                req = COMM.irecv(source = 0, tag = 1)
                self.reqns.t = req.wait()

                req = COMM.irecv(source = 0, tag = 2)
                self.__I_guess = req.wait()

                while not flag:
                    req = COMM.irecv(source = 0, tag = 3)
                    flag2, value = req.test()
                    if flag2:
                        self.__t_start = value

                    req = COMM.irecv(source = 0, tag  = 4)
                    flag3, value = req.test()
                    if flag3:
                        self.__tol = value

                    req = COMM.irecv(source = 0, tag = 5)
                    flag4, value = req.test()
                    if value == True:
                        flag = True


        if self.__option == 'n_dc':
            waiting = True
            req = COMM.isend(waiting, dest=0, tag = 100)

            req = COMM.irecv(source = 0, tag = 6)
            n = req.wait()

            self.__n = n[0]
            self.__ind = n[1]
            self.__run = n[2]

        elif self.__option == 'current':
            pass


    def ndc_routine(self):
        ns = copy.deepcopy(self.__ns)
        indices = range(ns.size)
        indices_fin = range(ns.size)
        Is = np.zeros(ns.size)

        init = [True for i in range(self.size)]

        while len(indices)>0:
            num = self.test_all(100)

            if init[num-1] is True:
                init[num-1] = False
                ind = indices.pop(0)
                req = COMM.isend((ns[ind], ind, True), dest=num, tag = 6)
            else:
                req = COMM.irecv(source = num, tag = 7)
                II = req.wait()
                Is[II[1]]=II[0]
                _ = indices_fin.pop()

                ind = indices.pop(0)
                req = COMM.isend((ns[ind], ind, True), dest=num, tag = 6)

        while len(indices_fin)>0:
            num = self.test_all(100)
            
            if init[num-1] is False:
                req = COMM.irecv(source = num, tag = 7)
                II = req.wait()
                Is[II[1]]=II[0]
                _ = indices_fin.pop()

                req = COMM.isend((None, None, False), dest=num, tag = 6)
            else:
                init[num-1] = False
                req = COMM.isend((None, None, False), dest=num, tag = 6)

        return Is

    def test_all(self, tag):
        switch = False

        while not switch:
            for i in range(self.size):
                req = COMM.irecv(source=i+1, tag = tag)
                switch, _ = req.test()
                if switch:
                    num = i+1
                    break
                del req
        return num

    def send_to_all(self, value, tag):
        for i in range(self.size):
            COMM.isend(value, dest=i+1, tag=tag)

if __name__ == '__main__':
    pass

