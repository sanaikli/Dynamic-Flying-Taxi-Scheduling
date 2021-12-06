# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:37:59 2021

@author: saikli
"""

""" This file contains the implementation of important functions that are recurently
called when implementing a taxi scheduling algorithm (FCFS, NN or GA)
- FCFS: First-Come, First-Served
- NN:   Nearest Neighbor
- GA:   Genetic Algorithm 
"""


#-----      important packages    ------
import pandas as pd
import math
import time
#---------------------------------------

# Constant values of the problem (from Panwadee's GA2)
v_fly  = 50   # in km/h
bcr    = 0.67 # battery consumption rate in %/minute
s_need = 5    # take-off and landing time in minutes
b_min  = 5    # minimum reamaining battery level in %
R      = 60   # battery recharging time to full in minutes
T      = 1440 # available operating time in minutes


# the function "prepare_data" extracts the data (problem parameters) from the ".txt" file
# input : 
#        * file_instance : ".txt" file name
# output: 
#        * nb_req        : total number of requests in the instance
#        * nb_taxi       : total number of taxis
#        * data          : pandas dataframe representing the instance,
#                          each row represents a request, and
#                          each column represents an information about the
#                          request : index (req_id), origine (ori_x, ori_y),
#                          destination (des_x, des_y),...
#        * center_val    : [x,y]--coordinates of the center

def prepare_data(file_instance, instance_orig):
    # Read data from text file
    with open(file_instance) as f:
        line_req_taxi = f.readline().split()
        nb_req = int(line_req_taxi[0])
        nb_taxi = int(line_req_taxi[1])
        line_center = f.readline().split()
        center_val = [int(elem) for elem in line_center]
        matrix = []
        for line in f:
            line1 = line.split()
            value = [float(elem) for elem in line1]
            matrix.append(value)
            
    
    if instance_orig == "sana":        
        data = pd.DataFrame(matrix, columns=['req_id','ori_x','ori_y','des_x',
                                             'des_y','early_t','pick_t','late_t',
                                             'dist_t','dur_t'])
        
    
    if instance_orig == "panwadee":    
        data = pd.DataFrame(matrix, columns=['req_id','ori_x','ori_y','des_x','des_y','pick_t'])
        #Compute duration time from origin to destination + takeoff and landing time
        data['dist'] = data.apply(lambda x: round(math.sqrt((x['ori_x']-x['des_x'])**2 + (x['ori_y']-x['des_y'])**2),2), axis=1)
        data['dur_t'] = data.apply(lambda x: round(x['dist']/(v_fly*1000/60)+2*s_need,2), axis=1)
        data['early_t'] = data.apply(lambda x: x['pick_t'], axis=1)
        data['late_t'] = data.apply(lambda x: x['pick_t'], axis=1)
        #data['early_t'] = data.apply(lambda x: x['pick_t']-2*x['dur_t'], axis=1)
        #data['late_t'] = data.apply(lambda x: x['pick_t']+2*x['dur_t'], axis=1)
    # print(data)
    return nb_req, nb_taxi, data, center_val



# "cal_dur_move" computes the trip duration between two locations:
# destination location of the previous request and the 
# origin location of the current request
# input : 
#        * center_val: [x,y]--coordinates of the center
#        * data      : pandas dataframe representing the instance
#        * next_ori  : index of the current request
#        * prev_des  : index of the previous request
# output: 
#        * move_t    : trip duration in minutes

def cal_dur_move_time(center_val, data, next_ori, prev_des=-1):
    x_prev_des = center_val[0]
    y_prev_des = center_val[1]
    if prev_des != -1:
        x_prev_des = data['des_x'][prev_des]
        y_prev_des = data['des_y'][prev_des]
    dist_move = round(math.sqrt((data['ori_x'][next_ori]-x_prev_des)**2 + 
                                (data['ori_y'][next_ori]-y_prev_des)**2),2)
    move_t = round(dist_move/(v_fly*1000/60),2) + 2*s_need
    return move_t



# "cal_dur_back_to_center" computes the trip duration from the current request
# location to the center
# input : 
#        * req_test  : request index
#        * center_val: [x,y]--coordinates of the center
#        * data      : pandas dataframe representing the instance
# output: 
#        * back_t    : trip duration in minutes to go back to center
def cal_dur_back_to_center(req_test, center_val, data):
    dist_back = round(math.sqrt((data['des_x'][req_test]-center_val[0])**2 +
                                (data['des_y'][req_test]-center_val[1])**2),2)
    back_t = round(dist_back/(v_fly*1000/60),2) + 2*s_need
    return back_t



# "obj_value" computes the value of the objective-function: the cumulative service time 
# input : 
#        * matrix_task      : a python dictionary representing the tasks of each taxi
#                             the key represents the taxi
#                             the values are a list of tasks for the taxi
#
#        * matrix_start_time: a python dictionary representing the start time
#                             of each task.
#                             the key represents the taxi
#                             the values are a list of starting time of each task
#
#        * matrix_fini_time : a python dictionary representing the finish time
#                             of each task.
#                             the key represents the taxi
#                             the values are a list of finishing time of each task
# output: 
#        * accumulate_service_time: the value of the objective-function

def obj_value(matrix_task, matrix_start_time, 
            matrix_fini_time, nb_taxis):
    accumulate_service_time = 0
    for j in range(nb_taxis):
        for i in range(len(matrix_task["taxi"+str(j+1)])):
            if matrix_task["taxi"+str(j+1)][i] != "b":
                accumulate_service_time += matrix_fini_time["taxi"+str(j+1)][i] - matrix_start_time["taxi"+str(j+1)][i]
    return accumulate_service_time



# "dur_non_profit_trip" computes the time duration of non-profitable trips, 
# such as the trips from the center to the location of a request, the trip between
# the destination location of a requests  and the origin location of the next request...
# input : 
#        * matrix_task: a python dictionary representing the tasks of each taxi
#                       the key represents the taxi
#                       the values are a list of tasks for the taxi
#        * data       : pandas dataframe representing the instance
#        * center_val : [x,y]--coordinates of the center                            the values are a list of finishing time of each task
# output: 
#        * dur_t      : the time duration of non-profitable trips
#                       according to the distribution of tasks in matrix_task
def dur_non_profit_trip(matrix_task, data, center_val):
    dur_t = 0
    for taxi in matrix_task.keys():
        taxi_task = [i if i!='b' else -1 for i in matrix_task[taxi] ]
        for i in range(1,len(taxi_task)):
            if taxi_task[i]==-1:# check if current request is battery charging
                i_1_data_idx = data.index[data['req_id'] ==taxi_task[i-1]].tolist()[0]
                dur_t += cal_dur_back_to_center(i_1_data_idx, center_val, data)
                
            elif taxi_task[i-1]==-1:# check if previous request is battery charging
                i_data_idx = data.index[data['req_id']==taxi_task[i]].tolist()[0]
                dur_t += cal_dur_move_time(center_val, data,i_data_idx)
                
            else:
                i_data_idx = data.index[data['req_id']==taxi_task[i]].tolist()[0]
                i_1_data_idx = data.index[data['req_id']==taxi_task[i-1]].tolist()[0]
                dur_t += cal_dur_move_time(center_val, data,
                                     i_data_idx,i_1_data_idx)
        if taxi_task != []:
            first_elem = data.index[data['req_id'] == taxi_task[0]].tolist()[0]
            dur_t += cal_dur_move_time(center_val,data, first_elem)
        
            if taxi_task[-1] !=-1:
                last_elem = data.index[data['req_id'] == taxi_task[-1]].tolist()[0]
                dur_t += cal_dur_back_to_center(last_elem, center_val, data)

    return dur_t


# "rolling_time_window" is a function that implements the rolling-horizon approach
# with rolling time windows. It shcedules the requests inside the first window using
# the scheduling method "heuristic". then, it moves the unserved requests to the next 
# window and schedules them with the new requests ; this process continues
# untill reaching the last time window
# input : 
#        * heuristic    : the scheduling heuristic (FCFS, NN or GA)
#
#        * T_inf, T_sup : upper and lower bounds on the scheduling horizon
#                         [T_inf, T_sup] = [0, 1440](24 hours for example) 
#        * window_len   : length of the (rolling) time window
#        * req          : dictionary of available requests
#                         the keys represent the requests index
#                         the values are the pick-up times
#        * av_taxis     : list of available taxis
#        * data         : pandas dataframe representing the instance
# output : 
#        * cpu          : cpu time
#        * cumul_obj_val: the value of the objective-function at the end of the horizon
#
#        * serv_req     : dictionary of the served requests
#                         the keys represent the requests index
#                         the values are the pick-up times
#        * unserv_req   : dictionary of the (remaining) unserved requests
#        * matrix_task, matrix_start_time, and matrix_fini_time and defined above
#         (see description of the function "obj_value"  )


def rolling_time_window(heuristic,T_inf,T_sup,window_len,req, av_taxis,data):
    av_req, serv_req, unserv_req = {}, {}, {}
    # matrix initializations
    matrix_task       = {"taxi"+str(j): [] for j in range(1, len(av_taxis)+1)}
    matrix_start_time = {"taxi"+str(j): [] for j in range(1, len(av_taxis)+1)}
    matrix_fini_time  = {"taxi"+str(j): [] for j in range(1, len(av_taxis)+1)}
    
    # battery level initialization
    nb_taxis = len(av_taxis)
    battery_level = [100]*nb_taxis # (100% for all av_taxis)
    
    t_i = time.time() # initial time (to compute the cpu time)
    for t in range(T_inf, T_sup, window_len):
        av_req.update({i: req[i] for i in req.keys() if  
                  req[i] >= t and req[i] < t+window_len})

        serv_req, unserv_req, new_matrix_task, new_matrix_start_time, new_matrix_fini_time, new_battery_level = heuristic(av_req,
                                                  av_taxis,battery_level,
                                                  matrix_task,matrix_start_time,
                                                  matrix_fini_time, data,
                                                  T_sup)
        matrix_task      = new_matrix_task
        matrix_start_time= new_matrix_start_time
        matrix_fini_time = new_matrix_fini_time

        # move unserved requests from previous window to the start of current window
        for i_req in unserv_req.keys():
            i_req_data_idx = data.index[data['req_id'] == i_req].tolist()[0]
            if unserv_req[i_req]>=t and unserv_req[i_req]<t+window_len:
                unserv_req[i_req] = t+window_len
                data.at[i_req_data_idx,'pick_t'] = av_req[i_req] 
                
        # update av_req with unserved requests from previous window
        av_req = {i:v for i,v in av_req.items() if i not in serv_req}
        av_req.update(unserv_req)
        serv_req.update(serv_req)
        
        # battery level updates
        battery_level = new_battery_level
        
    t_f = time.time()         # final time
    cpu = round(t_f - t_i, 4) # cpu calculation
    
    # final objective-value
    cumul_obj_val = obj_value(matrix_task,matrix_start_time,matrix_fini_time, nb_taxis)
    return cpu, cumul_obj_val, serv_req, unserv_req, matrix_task, matrix_start_time, matrix_fini_time
