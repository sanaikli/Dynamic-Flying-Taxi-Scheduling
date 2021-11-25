# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:41:22 2021

@author: saikli
"""

""" This program implements the FCFS (First-Come, First-Served)
heuristic  for the dynamic flying taxis scheduling """


#-----      important packages    ------
import utility_functions as uf
import statistics
import operator
#import time
#---------------------------------------
               
# the fcfs heuristic schedules requests in a "first-come first-serve"
# fashion, according to requests start time (static case) 

# the FCFS (First-Come, Firs-Served) heuristic schedules the requests according 
# to requests start time, i.e., the realiest requests are the first to be scheduled
# input : 
#        * av_req           : dictionary of available requests
#                             the keys represent the requests index
#                             the values are the pick-up times
#        * av_taxis         : list of available taxis
#        * battery_level    : list for battery level of each taxi
#        * matrix_task      : a python dictionary representing the tasks of each taxi
#                             the key represents the taxi
#                             the values are a list of tasks for the taxi
#        * matrix_start_time: a python dictionary representing the start time
#                             of each task.
#                             the key represents the taxi
#                             the values are a list of starting time of each task
#        * matrix_fini_time  : a python dictionary representing the finishing time
#                              of each task.
#                              the key represents the taxi
#                              the values are a list of finishing time of the request
#        * data              : pandas dataframe representing the instance
#        * T_sup             : upper value of the scheduling horizon (1440 minutes)
# output : 
#        * serv_req          : dictionary of served requests
#                              the keys represent the requests index
#                              the values are the serve times (pick-up time
#        * unserv_req        : dictionary of unserved requests
#                              the keys represent the requests index
#                              the values are the pick-up times
#                              or a time inside the request interval)
#        * matrix_task       : the previous matrix_start_time with updates
#        * matrix_start_time : the previous matrix_task with updates
#        * matrix_fini_time  : the previous matrix_fini_time with updates
#        * battery_level     : updates of the battery level

def fcfs_heuristic(av_req, av_taxis, battery_level, matrix_task,
         matrix_start_time, matrix_fini_time, data, T_sup):
    
    # sorting requests according to earliest pickup time (fcfs fashion)
    av_req = dict(sorted(av_req.items(), key=operator.itemgetter(1)))
    av_req_copy = av_req.copy()

    # served requests initialization   
    serv_req = {}
    
    prev_req = 'c'
    for i_req in list(av_req_copy.keys()):
        
        # i_req index in data table
        i_req_data_idx = data.index[data['req_id'] == i_req].tolist()[0]
        dur_prep = uf.cal_dur_move_time(center_val, data, i_req_data_idx, -1)
        
        #--------------------------------------------------------------------
        # choose the first available taxi
        temp = [key for key, value in matrix_fini_time.items() if value == []]
        if temp != []:
            taxi   = temp[0]
            j_taxi = int(taxi[4:])
            temp.remove(taxi)
        else :
            temp = {key: matrix_fini_time[key][-1] for key in matrix_fini_time.keys()}
            j_taxi = int(min(temp, key=temp.get)[4:])
        #-----------------------------------------------------------------
        
        prev_fini_time = 0
        if (len(matrix_task["taxi"+str(j_taxi)]) != 0):
            prev_req = matrix_task["taxi"+str(j_taxi)][-1]
            if (prev_req != "b"):
                prev_req_data_idx = data.index[data['req_id'] == 
                                           prev_req].tolist()[0]
                dur_prep = uf.cal_dur_move_time(center_val,data,i_req_data_idx, prev_req_data_idx)
            
            prev_fini_time = matrix_fini_time["taxi"+str(j_taxi)][-1]
   
        #Check enough time to pick up  
        enough_time_to_pick = False
        if prev_fini_time + dur_prep <= data["pick_t"][i_req_data_idx]:
            enough_time_to_pick = True
            
        dur_t_req = data["dur_t"][i_req_data_idx]
        dur_back_center = uf.cal_dur_back_to_center(i_req_data_idx, center_val,data)
        
        #Check enough battery to serve and go back to center if needed
        enough_battery = False
        if battery_level[j_taxi-1] - uf.bcr*(dur_prep+dur_t_req+dur_back_center) >= uf.b_min:
            enough_battery = True
            
        # check if it's possible te serve the request
        if enough_time_to_pick:
            if enough_battery:
                if data["pick_t"][i_req_data_idx]+dur_t_req <= T_sup:
                    matrix_task["taxi"+str(j_taxi)].append(i_req)
                    matrix_start_time["taxi"+str(j_taxi)].append(data["pick_t"][i_req_data_idx])
                    
                    matrix_fini_time["taxi"+str(j_taxi)].append(round((data["pick_t"][i_req_data_idx]+dur_t_req), 2))
                    
                    battery_level[j_taxi-1] = round(
                        battery_level[j_taxi-1] - uf.bcr*(dur_prep+dur_t_req),
                        2)

                    # remove the assigned demand from av_requests
                    serv_req[i_req] = av_req[i_req]
                    del av_req[i_req]

            else: # charge battery if battery level is not enough
                matrix_task["taxi"+str(j_taxi)].append("b")
                dur_prev_to_center = uf.cal_dur_back_to_center(prev_req_data_idx,center_val,data)
                matrix_start_time["taxi"+str(j_taxi)].append(prev_fini_time + dur_prev_to_center)
                matrix_fini_time["taxi"+str(j_taxi)].append(prev_fini_time + dur_prev_to_center + uf.R)
                battery_level[j_taxi-1] = 100
        
                dur_prep1 = uf.cal_dur_move_time(center_val,data,i_req_data_idx)
                prev_fini_time1 = matrix_fini_time["taxi"+str(j_taxi)][-1]

                #----------------------------------------------------------
                # check if it's possible to serve in the time interval [early_t, late_t]
                if ((prev_fini_time1 + dur_prep1 <= data["late_t"][i_req_data_idx]) 
                    &
                    (prev_fini_time1 + dur_prep1 >= data["early_t"][i_req_data_idx])
                    &
                    (battery_level[j_taxi-1] - uf.bcr*(dur_prep1+dur_t_req+dur_back_center) >= 
                     uf.b_min)):
                    if data["pick_t"][i_req_data_idx]+dur_t_req <= T_sup:
                        matrix_task["taxi"+str(j_taxi)].append(i_req)
                        # new pick_t: serve the requests as soon as the taxi 
                        # finiches its previous task
                        i_req_new_pick_time = round(prev_fini_time1+dur_prep1, 2)
                        matrix_start_time["taxi"+str(j_taxi)].append(i_req_new_pick_time)
                        matrix_fini_time["taxi"+str(j_taxi)].append(round((i_req_new_pick_time+
                                                                           dur_t_req), 2))
                        battery_level[j_taxi-1] = round(battery_level[j_taxi-1] - uf.bcr*(dur_prep1+
                                                                                       dur_t_req), 2)
                        # remove the assigned demand from av_req
                        serv_req[i_req] = av_req[i_req]
                        del av_req[i_req]
                        
        elif ((prev_fini_time + dur_prep <= data["late_t"][i_req_data_idx]) 
              &
              (prev_fini_time+ dur_prep >= data["early_t"][i_req_data_idx])
              &
              (battery_level[j_taxi-1] - uf.bcr*(dur_prep+dur_t_req+dur_back_center) >= 
               uf.b_min)):
                        # new pick_t: serve the requests as soon as the taxi 
                        # finiches its previous task
                        i_req_new_pick_time = round(prev_fini_time+dur_prep, 2)
                        if i_req_new_pick_time+dur_t_req <= T_sup:
                            matrix_task["taxi"+str(j_taxi)].append(i_req)
                            matrix_start_time["taxi"+str(j_taxi)].append(i_req_new_pick_time)
                            matrix_fini_time["taxi"+str(j_taxi)].append(round((i_req_new_pick_time+
                                                                           dur_t_req), 2))
                            battery_level[j_taxi-1] = round(battery_level[j_taxi-1] - 
                                                            uf.bcr*(dur_prep+dur_t_req), 2)
                            
                            serv_req[i_req] = av_req[i_req]
                            del av_req[i_req] # remove the assigned demand from av_req

    unserv_req = av_req #update unserved demands
    return serv_req, unserv_req, matrix_task, matrix_start_time, matrix_fini_time, battery_level

               

  
# Main program for numerical tests                       
if __name__ == "__main__":
    
    # RH parameters
    T_inf, T_sup = 0, 1440 # start and end times (respectively) 
                           # of the day (in minutes) : 1440 = 24h
                           # the scheduling horizon T=[0, 1440]                  
    
    window_lengths = [720]#, 90, 180, 240, 360, 480, 720, 840, 1440] 

    instances = ["new_instances/instance50_2.txt","new_instances/instance100_3.txt","new_instances/instance100_5.txt",
             "new_instances/instance250_5.txt","new_instances/instance250_10.txt","new_instances/instance500_4.txt",
             "new_instances/instance500_10.txt","new_instances/instance1000_9.txt","new_instances/instance1000_15.txt",
             "new_instances/instance10000_20.txt"]

      
    fcfs_avg_unserv = []
    fcfs_avg_obj    = []
    fcfs_avg_profit = []
    fcfs_avg_cpu    = []
    
    instances = ["instances/instance10_2.txt"]
    
    for window_len in window_lengths:
        print("\n\n________________  window_len =  ", window_len,"________________")
        argument = 'w+'
        fcfs_obj_values = []
        fcfs_non_profit = []
        fcfs_cpus = []
        fcfs_unserv_perc = [] # percentage of unserved demands
        fcfs_averages_morning, fcfs_averages_evening = [], []
        for instance_name in instances:
            print("\n*********", instance_name)
            # loading data from Panwadee's data generator
            nb_req, nb_taxis, data, center_val = uf.prepare_data(instance_name, 'panwadee')
            #data=data[1:7]
            data.req_id= data.req_id.astype(int)
        
            # all (known) requests and their pick-up times
            # in the begining of the scheduling horizon
            req = {data.req_id[i]:data.pick_t[i] for i in data.index}
        
            # available taxis in the begining of the scheduling horizon
            av_taxis = [i for i in range(1, nb_taxis+1)] 

            # rolling_time_window call with FCFS heuristic
            fcfs_cpu, fcfs_cumul_obj_val, fcfs_serv_req, fcfs_unserv_req, fcfs_matrix_task, fcfs_matrix_start_time, fcfs_matrix_fini_time= uf.rolling_time_window(fcfs_heuristic,
                                                                                                                        T_inf,T_sup,
                                                                                                                        window_len,
                                                                                                                        req, av_taxis,
                                                                                                                        data)
            fcfs_obj_values.append(round(fcfs_cumul_obj_val,2))
            fcfs_non_profit.append(round(uf.dur_non_profit_trip(fcfs_matrix_task,
                                                                data, center_val),2))
            fcfs_cpus.append(fcfs_cpu)
            fcfs_unserv_perc.append(round(len(fcfs_unserv_req)/len(data),2)*100)
        
            # save results in ".txt" file named 
            # "results_<dynamic/static>_fcfs.txt
            """if window_len == 1440:
                file_name = "results_fcfs_static.txt"
            else:
                file_name = "results_fcfs_rh.txt"
            with open(file_name, argument) as f:
                f.write("\n\n\n--------\t"+ str(instance_name)+"\t--------")
                f.write("\n* matrix_task : "+str(fcfs_matrix_task))
                f.write("\n* unserved req : "+str(fcfs_unserv_req))
                f.write("\n* objective value : "+str(round(fcfs_cumul_obj_val,2)))
                f.write("\n* cpu time : "+str(fcfs_cpu))
            argument = 'a' """
            
            #print("\nmatrix task : ", fcfs_matrix_task)
            print("\n* objective value : ", round(fcfs_cumul_obj_val,2),
                  " (total service time)")
            print("* cpu : ", round(fcfs_cpu,4))
            print("* percentage of unserved requests : ", 
                  round(len(fcfs_unserv_req)/len(data),2)*100, " %") 
        
        
            #-----------------------------------------------------------------
            """Instance statistics """
            fcfs_total_peak_morning = data.loc[data.pick_t.between(360,600),
                                          'pick_t'].count()
            fcfs_ave_morning = round(fcfs_total_peak_morning/6,2)
            fcfs_averages_morning.append(fcfs_ave_morning)
        
            fcfs_total_peak_evening = data.loc[data.pick_t.between(1020,1260),
                                      'pick_t'].count()
            fcfs_ave_evening = round(fcfs_total_peak_evening/6,2)
            fcfs_averages_evening.append(fcfs_ave_evening)
            #-----------------------------------------------------------------
            
        fcfs_avg_unserv.append(round(statistics.mean(fcfs_unserv_perc),2))
        fcfs_avg_obj.append(round(statistics.mean(fcfs_obj_values),2))
        fcfs_avg_profit.append(round(statistics.mean(fcfs_non_profit),2))
        fcfs_avg_cpu.append(round(statistics.mean(fcfs_cpus),2))
        
    print("\n\n--> fcfs_avg_unserv : ", fcfs_avg_unserv)
    print("\n-->  fcfs_avg_obj : "     , fcfs_avg_obj,)
    print("\n-->  fcfs_avg_profit : "  , fcfs_avg_profit)
    print("\n-->  fcfs_avg_cpu : "     , fcfs_avg_cpu)
        
        
        
        
        
        
                     