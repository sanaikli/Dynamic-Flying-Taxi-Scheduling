# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:48:57 2021
@author: saikli
"""

""" 
This program contains the implementation of the Genetic Algorithm (GA)
of Panwadee, adatpted to the Rolling Horizon approach (RH). 
The RH is implemented in the function << rh_ga2 >> 
"""


#-----      important packages    ------
import utility_functions as uf
import numpy as np
import cProfile
import pstats
import random
import time
import copy

#---------------------------------------


# the function "cal_obj2" implements the Second objective-function that
# maximizes the total service time
# input : 
#        * nb_taxis         : total number of taxis
#        * m_task_chrom     : matrix task for chromosome, wich represents a python 
#                             dictionary representing the tasks of each taxi
#                             the key represents the taxi
#                             the values are a list of tasks for the taxi
#
#        * m _st_chrom      : python dictionary representing the start time of each task.
#                             the key represents the taxi
#                             the values are a list of starting time of each task
#        * matrix_ft_chrom  : python dictionary representing the finish time of each task.
#                             the key represents the taxi
#                             the values are a list of finishing time of each task
# output: 
#        * accumulate_service_time : the value of the objective-function

def cal_obj2(nb_taxis, m_task_chrom, m_st_chrom, m_ft_chrom):
    accumulate_service_time = 0
    for j in range(nb_taxis):
        for i in range(len(m_task_chrom["taxi"+str(j+1)])):
            if m_task_chrom["taxi"+str(j+1)][i] != "b":
                accumulate_service_time += m_ft_chrom["taxi"+str(j+1)][i] - m_st_chrom["taxi"+str(j+1)][i]
    return accumulate_service_time



# the function "sorting_population" sort a popolation of chromosomes according
# to the fitness function (value of the objective-function)
#         * arr_pop : a python list reprsenting the population (set of chromosomes)
#         * arr_res : a python list of the objective-value of the corresponding chromosome 

# output: * sorted population

def sorting_population(arr_pop, arr_res):
    n = len(arr_pop)
    while n > 1:
        for i in range (n-1):
            if (arr_res[i] < arr_res[i+1]):
                arr_res[i], arr_res[i+1] = arr_res[i+1], arr_res[i]
                arr_pop[i], arr_pop[i+1] = arr_pop[i+1], arr_pop[i]
        n = n - 1

# the function "decode_GA" represents the decoding of the chromosome in the 
# genetic algorithm proposed by Panwadee (function "genetic_algorithm");
# the chromosome is a python list; each element of the list represents
# a score corresponding to a request serviced by a flying taxi

# input : 
#        * chromo     : chromosome, which is a python list that represents the chromosome 
#                       (Individual solution in the GA algorithm)
#        * center_val : [x,y]--coordinates (values) of the recharging center     
#        * av_taxis   : list of available taxis
#        * av_data    : available data, which is a pandas dataframe with parameters
#                       for the available requests
#        * all_data   : pandas dataframe with parameters for all requests
#        * serv       : list of the served requests
#        * av_bl      : available battery level in the current horizon
#        * m_task     : python dict that represents the matrix task (explained above)
#        * m_st       : python dict representing the (matrix) start time of each task
#        * m_ft       : python dictionary representing the finishing time of each task


# output: (if details == None)
#        * obj_val_chrom    : the value of the objective-function

# output: (if details != None)
#        * serv          : list of the served requests
#        * new_bl        : updated battery level
#        * new_m_task    : updated matrix task (with new served requests)
#        * new_m_st      : updated matrix start time
#        * new_m_ft      : updated matrix finished time
#        * obj_val_chrom : the value of the objective-function

#----------------------------------------------
def decode_GA(chromo, center_val, av_taxis, av_data, all_data, 
              serv_req, av_bl, m_task, m_st, m_ft, detail):
    
    new_m_task = copy.deepcopy(m_task)
    new_m_st   = copy.deepcopy(m_st)
    new_m_ft   = copy.deepcopy(m_ft)
    new_bl     = av_bl.copy()
    #----------------------------------------------
    
    
    nb_taxis            = len(av_taxis)
    chrom_priority      = chromo.copy()
    combine_ideal_chrom = []
    
    for i in range(len(chrom_priority)):
        new_value_i = ((uf.T-av_data["pick_t"][i//nb_taxis])/uf.T)*((1+chrom_priority[i])/2)
        combine_ideal_chrom.append(new_value_i)
        
    chrom_priority = combine_ideal_chrom
    #### End of ideal priority combination

    order_assign = sorted(range(len(chrom_priority)),
                          key=lambda k: chrom_priority[k], reverse=True)
    #print("\n *********** order_assign = ", order_assign[:9],"...")
    serv_req = []
    for i_gene in order_assign:
        i_req = (i_gene//nb_taxis)+1
        j_taxi = (i_gene%nb_taxis)+1
        
        #----------------------------------------------
        req_id          = av_data.req_id[i_req-1]
        req_id_data_idx = av_data.loc[av_data['req_id'] == req_id].index[0]
        """print("\n * req_id  =  ", req_id, " * j_taxi = ", j_taxi)
        print(" * pick_t: ", av_data.pick_t[req_id_data_idx])"""
        #----------------------------------------------
        
        dur_prep       = uf.cal_dur_move_time(center_val, av_data, req_id_data_idx)
        prev_fini_time = 0 
        prev_req       = 'c'
        if (len(new_m_task["taxi"+str(j_taxi)]) != 0):
            prev_req          = new_m_task["taxi"+str(j_taxi)][-1]
            if (prev_req != "b"): 
                #----------------------------------------------
                #print(" prev_req : ", prev_req)
                prev_req_data_idx = all_data.loc[all_data['req_id'] == prev_req].index[0]
                #----------------------------------------------
                
                dur_prep = uf.cal_dur_move_time(center_val, all_data, req_id_data_idx, prev_req_data_idx)
            prev_fini_time = new_m_ft["taxi"+str(j_taxi)][-1]


        #Check enough time to pick up  
        enough_time_to_pick = False
        if prev_fini_time + dur_prep <= av_data["pick_t"][req_id_data_idx]:
            enough_time_to_pick = True

        
        dur_t_req       = av_data["dur_t"][req_id_data_idx]
        dur_back_center = uf.cal_dur_back_to_center(req_id_data_idx, center_val, av_data)
        
        # Check enough battery to service and go back to center if needed
        enough_battery = False
        if new_bl[j_taxi-1] - uf.bcr*(dur_prep+dur_t_req+dur_back_center) >= uf.b_min:
            enough_battery = True
        
        # Possible to response the demand?
        if enough_time_to_pick:
            if enough_battery:
                if av_data["pick_t"][req_id_data_idx]+dur_t_req <= uf.T:
                    new_m_task["taxi"+str(j_taxi)].append(req_id)
                    new_m_st["taxi"+str(j_taxi)].append(av_data["pick_t"][req_id_data_idx])
                    new_m_ft["taxi"+str(j_taxi)].append(round((av_data["pick_t"] [req_id_data_idx]+dur_t_req), 2))
                    new_bl[j_taxi-1] = round(new_bl[j_taxi-1] - uf.bcr*(dur_prep+dur_t_req), 2) 
                    #----------------------------------------------
                    # add req_id to served requests
                    serv_req.append(req_id)
                    #----------------------------------------------
                    # remove the assigned demand from the chromosome
                    for j_count in range(nb_taxis):
                        elem_remove = nb_taxis*(i_req-1) + j_count
                        #print(" * elem_remove= ", elem_remove)
                        if (i_gene != elem_remove):
                            order_assign.remove(elem_remove)
      
            #----------------------------------------------
            # added by Sana for serving requests inside time-windows 
            else: # charge battery if battery level is not enough
                new_m_task["taxi"+str(j_taxi)].append("b")
                dur_prev_to_center = uf.cal_dur_back_to_center(prev_req_data_idx, center_val, all_data)
                new_m_st["taxi"+str(j_taxi)].append(prev_fini_time + dur_prev_to_center)
                new_m_ft["taxi"+str(j_taxi)].append(prev_fini_time + dur_prev_to_center + uf.R)
                new_bl[j_taxi-1] = 100
        
                dur_prep1 = uf.cal_dur_move_time(center_val, av_data, req_id_data_idx)
                prev_fini_time1 = new_m_ft["taxi"+str(j_taxi)][-1]

                # check if it's possible to serve in the time interval [early_t, late_t]
                if ((prev_fini_time1 + dur_prep1 <= av_data["late_t"][req_id_data_idx]) 
                    &
                    (prev_fini_time1 + dur_prep1 >= av_data["early_t"][req_id_data_idx])
                    &
                    (new_bl[j_taxi-1] - uf.bcr*(dur_prep1+dur_t_req+dur_back_center) >= 
                     uf.b_min)):
                    if av_data["pick_t"][req_id_data_idx]+dur_t_req <= T_sup:
                        new_m_task["taxi"+str(j_taxi)].append(req_id)
                        req_id_new_pick_time = round(prev_fini_time1+dur_prep1, 2)
                        new_m_st["taxi"+str(j_taxi)].append(req_id_new_pick_time)
                        new_m_ft["taxi"+str(j_taxi)].append(round((req_id_new_pick_time+dur_t_req), 2))
                        new_bl[j_taxi-1] = round(new_bl[j_taxi-1] - uf.bcr*(dur_prep1+dur_t_req), 2)
                        
                        serv_req.append(req_id)
                        # remove the assigned demand from the chromosome
                        for j_count in range(nb_taxis):
                            elem_remove = nb_taxis*(i_req-1) + j_count
                            if (i_gene != elem_remove):
                                order_assign.remove(elem_remove)
                        
        elif ((prev_fini_time + dur_prep <= av_data["late_t"][req_id_data_idx]) 
              &
              (prev_fini_time+ dur_prep >= av_data["early_t"][req_id_data_idx])
              &
              (new_bl[j_taxi-1] - uf.bcr*(dur_prep+dur_t_req+dur_back_center) >= 
               uf.b_min)):
            req_id_new_pick_time = round(prev_fini_time+dur_prep, 2)
            if req_id_new_pick_time+dur_t_req <= T_sup:
                new_m_task["taxi"+str(j_taxi)].append(req_id)
                new_m_st["taxi"+str(j_taxi)].append(req_id_new_pick_time)
                new_m_ft["taxi"+str(j_taxi)].append(round((req_id_new_pick_time+dur_t_req), 2))
                new_bl[j_taxi-1] = round(new_bl[j_taxi-1] - uf.bcr*(dur_prep+dur_t_req), 2)
                            
                serv_req.append(req_id)
                # remove the assigned demand from the chromosome
                for j_count in range(nb_taxis):
                    elem_remove = nb_taxis*(i_req-1) + j_count
                    if (i_gene != elem_remove):     
                        order_assign.remove(elem_remove)
            #----------------------------------------------
        """print(" * new_m_task [taxi",j_taxi,"] = ", new_m_task["taxi"+str(j_taxi)][-1] )
        print(" * new_m_st [taxi",j_taxi,"] = ", round(new_m_st["taxi"+str(j_taxi)][-1], 2) )
        print(" * new_m_ft [taxi",j_taxi,"] = ", round(new_m_ft["taxi"+str(j_taxi)][-1], 2) )
        print(" * new_m_task = " ,new_m_task)"""

    obj_val_chrom = cal_obj2(nb_taxis, new_m_task, new_m_st, new_m_ft)   
    if detail != None:
        return serv_req, new_bl, new_m_task, new_m_st, new_m_ft, obj_val_chrom
    else:
        return obj_val_chrom
        



# the function "genetic_algorithm" the GA main loop, that performs cross-over and mutations
# to evolve the population in order to find the "best" chromosome/solution

# input : 
#        * center_val : [x,y]--coordinates (values) of the charging center     
#        * av_taxis   : list of available taxis
#        * av_data    : a pandas dataframe with the available requests data
#        * serv       : list of the served requests
#        * av_bl      : available battery level
#        * m_task     : python dictionary representing the tasks/requests serviced by each taxi
#        * m_st       : python dictionary representing the start time of each task
#        * m_ft       : python dictionary representing the finish time of each task

# output: 
#        * serv          : list of the served requests
#        * final_obj     : new value of the objective-function
#        * new_bl        : new battery level
#        * new_task      : new mtri _task 
#        * new_m_st_time : new mtrix start time of each request
#        * new_m_fi_time : new mtrix finished time of each request
#----------------------------------------------
def genetic_algorithm(center_val, av_taxis, av_data, all_data, serv_req, av_bl,
                      m_task, m_st, m_ft):
    #----------------------------------------------
    nb_req     = len(av_data)
    nb_taxis   = len(av_taxis)
    chrom_size = nb_req*nb_taxis
    nb_chrom   = 2*chrom_size
    population = np.random.rand(nb_chrom, chrom_size)

    result_GA = []
    for i in range(len(population)):
        chromosome     = population[i].copy()
        result_GA_elem = decode_GA(chromosome, center_val, av_taxis, av_data, all_data,
                                   serv_req, av_bl,m_task, m_st, m_ft, detail = None)
        result_GA.append(result_GA_elem)
    
    sorting_population(population, result_GA)
    best_res      = result_GA[0]
    stop_value    = 30
    nb_select     = int(0.1*nb_chrom)
    nb_crossover  = int(0.7*nb_chrom)
    nb_mutation   = nb_chrom - nb_select - nb_crossover
    n_not_improve = 0
    n_iteration   = 0

    while n_not_improve < stop_value:
        new_pop = []
        for i in range (nb_select):
            new_pop.append(population[i])
        for i in range (nb_crossover):
            chrom_p1 = population[random.randint(0,nb_select-1)]
            chrom_p2 = population[random.randint(nb_select,nb_chrom-1)]
            crossover_chrom = []
            for j in range (chrom_size):
                if random.randint(1,10) <= 7:
                    crossover_chrom.append(chrom_p1[j])
                else:
                    crossover_chrom.append(chrom_p2[j])
            new_pop.append(crossover_chrom)
        for i in range (nb_mutation):
            new_pop.append(np.random.rand(chrom_size))
    
        population = new_pop
        result_GA = []
        for i in range(nb_chrom):
            chromosome     = population[i].copy()
            result_GA_elem = decode_GA(chromosome, center_val, av_taxis, av_data, all_data,
                                       serv_req, av_bl, m_task, m_st, m_ft, detail = None)
            result_GA.append(result_GA_elem)
    
        sorting_population(population, result_GA)
        best_res_iter = result_GA[0]
        if (best_res_iter > best_res):
            best_res = best_res_iter
            n_not_improve = 0
        else:
            n_not_improve = n_not_improve + 1
        n_iteration = n_iteration + 1

    final_chrom = population[0]
    serv, new_bl, new_m_task, new_m_st, new_m_ft, new_obj = decode_GA(final_chrom,
                                                                        center_val,
                                                                        av_taxis,
                                                                        av_data,
                                                                        all_data,
                                                                        serv_req,
                                                                        av_bl,
                                                                        m_task,
                                                                        m_st,
                                                                        m_ft,
                                                                        detail = True)

    return serv, new_obj, new_bl, new_m_task, new_m_st, new_m_ft


# "rh_ga2" implements the rolling-horizon approach for the genetic algorithm. 
# It shcedules the requests inside the first window using
# the GA. then, it moves the unserved requests to the next 
# window and schedules them with the new requests ; this process continues
# untill reaching the last time window

# input : 
#        * T_inf, T_sup : upper and lower bounds on the scheduling horizon
#                         [T_inf, T_sup] = [0, 1440](24 hours for example) 
#        * window_len   : length of the (rolling)  window
#        * av_taxis     : list of available taxis
#        * all_data     : pandas dataframe representing the data for an instance
#        * center_val   : [x,y]--coordinates of the center   

# output : 
#        * cpu          : cpu time
#        * cumul_obj    : the value of the objective-function at the end of the horizon
#        * serv_req     : list of the served requests
#        * av_req       : list of the (remaining) available requests (unserved requests)
#        * m_task, m_st, and m_ft : matrices of tasksc as defined in the above function descriptions


def rh_ga2(T_inf, T_sup, window_len, av_taxis, all_data, center_val):
    av_req     = []
    serv_req   = []
    
    # matrix initializations
    m_task    = {"taxi"+str(j): [] for j in range(1, len(av_taxis)+1)}
    m_st      = {"taxi"+str(j): [] for j in range(1, len(av_taxis)+1)}
    m_ft      = {"taxi"+str(j): [] for j in range(1, len(av_taxis)+1)}
    nb_taxis      = len(av_taxis)
    av_bl         = [100] * nb_taxis # battery level (100% for all av_taxis)
    cumul_obj     = 0
    t_i           = time.time()      # initial time (to compute the cpu time)
    
    for t in range(T_inf, T_sup, window_len):
        print("\n * t = ", t) 
        # data preparation
        av_req.extend( [all_data.req_id[i] for i in all_data.index 
                           if all_data.pick_t[i] >= t and all_data.pick_t[i] < t+window_len] )
        av_data = all_data.loc[all_data['req_id'].isin(av_req)]
        l = [i for i in range(len(av_data))]
        av_data.index = l
        print(" * available requests =", av_req)
        new_serv_req, obj, new_bl, new_m_task, new_m_st, new_m_ft = genetic_algorithm(center_val,
                                                                             av_taxis,
                                                                             av_data, 
                                                                             all_data, 
                                                                             serv_req,
                                                                             av_bl,
                                                                             m_task,
                                                                             m_st,
                                                                             m_ft)

        cumul_obj += obj
        # update requests list: available requests (av_req) and served requests (serv_req)
        av_req = [i for i in av_req if i not in new_serv_req]
        serv_req.extend(new_serv_req)
        
        av_bl  = new_bl # battery level updates
        m_task = new_m_task
        m_st   = new_m_st
        m_ft   = new_m_ft

    t_f       = time.time()                              # final time
    cpu       = round(t_f - t_i, 4)                      # cpu calculation
    cumul_obj = uf.obj_value(m_task, m_st, m_ft, nb_taxis) # final objective-value
    print(" * unserved requests : ", av_req)
    return cpu, cumul_obj, serv_req, av_req, m_task, m_st, m_ft



#___________________            main          _______________________________

# Main program for numerical tests                       
if __name__ == "__main__":
    
    T_inf, T_sup   = 0, 1440 # start and end timesof the day (in minutes)                
    window_lengths = [480]#, 90, 180, 240, 360, 480, 540, 720, 840, 1440] 
    instances      = ["instances/instance10_2.txt", "instances/instance20_2.txt" ,
                      "instances/instance20_3.txt","instances/instance30_2.txt"  ,
                      "instances/instance30_3.txt", "instances/instance30_4.txt" , 
                      "instances/instance50_2.txt", "instances/instance50_3.txt" ,
                      "instances/instance50_5.txt", "instances/instance100_3.txt",
                      "instances/instance100_5.txt"]
    
    
    ga2_avg_unserv = []
    ga2_avg_obj    = []
    ga2_avg_profit = []
    ga2_avg_cpu    = []

    instances = ["instances/instance10_2.txt"]
    
    for window_len in window_lengths:
        print("\n\n__________  window length =  ", window_len," min _____")
        argument = 'w+'
        ga_obj_values  = []
        ga_non_profit  = []
        ga_cpus        = []
        ga_unserv_perc = [] # percentage of unserved demands
        ga_averages_morning, fcfs_averages_evening = [], []

        for instance_name in instances :
            print("\n--->", instance_name)
            nb_req, nb_taxis, all_data, center_val = uf.prepare_data(instance_name, 'panwadee')
            #all_data=all_data[:6]
            all_data.req_id= all_data.req_id.astype(int)
            av_taxis = [i for i in range(1, nb_taxis+1)] 
        
            for i_run in range(1,2):
                cpu, obj, s_req, uns_req, ga_m_task, ga_m_st, ga_m_ft = rh_ga2(T_inf, T_sup, 
                                                                               window_len, 
                                                                               av_taxis,
                                                                               all_data, 
                                                                               center_val)
            
                #-------------------------------------------------------------
                # cProfile to analyze the code
                """cProfile.run('rh_ga2(T_inf, T_sup, window_len, av_taxis,data, center_val)',
                                'ga_output.dat')
                with open("ga_output_time.txt", "w") as f:
                    p = pstats.Stats("ga_output.dat", stream = f)
                    p.sort_stats("time").print_stats()
                
                with open("ga_output_calls.txt", "w") as f:
                    p = pstats.Stats("ga_output.dat", stream = f)
                    p.sort_stats("calls").print_stats()"""
                #-------------------------------------------------------------

                ga_obj_values.append(round(obj,2))
                ga_cpus.append(round(cpu,2))
                ga_non_profit.append(round(uf.dur_non_profit_trip(ga_m_task, all_data, center_val), 2))
                ga_unserv_perc.append(round(len(uns_req)/len(all_data)*100,2))

        print("\n\n * Objective values : ", ga_obj_values)
        print(" * Percentage of unserved requests: ", ga_unserv_perc)
        print(" * GA CPU time : ", ga_cpus)
        
    


