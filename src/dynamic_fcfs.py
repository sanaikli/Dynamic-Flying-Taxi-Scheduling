# -*- coding: utf-8 -*-
""" This program implements the FCFS (First-Come, First-Served)
heuristic  for the dynamic flying taxis scheduling """
__date__ = '10/06/2021'
__author__ = 'saikli'

# -----      important packages    ------
import utility_functions as uf
import statistics
import operator
import pandas as pd
import numpy as np
# import time


# ---------------------------------------
class DynamicFCFS:
    """First-Come, First-Served"""
    def __init__(self, window_lengths, instances, result, matrix_result, instance_orig):
        """Main program for numerical tests

        input :
           * window_len   : length of the (rolling) window
           * instances    : list of instance_name
        """

        # RH parameters
        t_inf, t_sup = 0, 1440  # start and end times (respectively)
        # of the day (in minutes) : 1440 = 24h
        # the scheduling horizon T=[0, 1440]

        fcfs_avg_unserv = []
        fcfs_avg_obj = []
        fcfs_avg_profit = []
        fcfs_avg_cpu = []

        # creation of results DataFrame
        i = 0
        if result:
            df_column = ['Objective values', '% unserved requests', 'GA CPU time', 'window length',
                         'Heuristic']
            if matrix_result:
                df_column = ['Objective values', '% unserved requests', 'GA CPU time', 'window length',
                             'matrix task', 'matrix start time', 'matrix finish time', 'Heuristic']
            self.__test_results = pd.DataFrame(index=np.arange(0, len(window_lengths)),
                                               columns=df_column)
        else:
            self.__test_results = None

        print("\n\n________________  Heuristic =  FCFS  ________________")
        for window_len in window_lengths:
            print("\n\n________________  window_len =  ", window_len, "________________")
            try:
                # argument = 'w+'  # For Test
                fcfs_obj_values = []
                fcfs_non_profit = []
                fcfs_cpus = []
                fcfs_unserv_perc = []  # percentage of unserved demands
                fcfs_averages_morning, fcfs_averages_evening = [], []

                for instance_name in instances:
                    print("\n*********", instance_name)
                    # loading data from Panwadee's data generator
                    nb_req, nb_taxis, data, center_val = uf.prepare_data(instance_name, instance_orig)
                    # data=data[1:7]
                    data.req_id = data.req_id.astype(int)

                    # all (known) requests and their pick-up times
                    # in the beginning of the scheduling horizon
                    req = {data.req_id[i]: data.pick_t[i] for i in data.index}

                    # available taxis in the beginning of the scheduling horizon
                    # av_taxis = [i for i in range(1, nb_taxis + 1)]

                    # ----------------------------------------------------------------
                    # data with zones
                    n_zone_x, n_zone_y = 1, 1
                    zone_id, zone_coord, zone_ori, zone_des, data, center_zone = uf.define_zone(data,
                                                                                                center_val,
                                                                                                n_zone_x,
                                                                                                n_zone_y)
                    # ----------------------------------------------------------------

                    # rolling_time_window call with FCFS heuristic
                    fcfs_cpu, fcfs_cumul_obj_val, fcfs_serv_req, fcfs_unserv_req, fcfs_m_task, fcfs_m_st, fcfs_m_ft = \
                        uf.rolling_time_window(self.fcfs_heuristic,
                                               t_inf, t_sup,
                                               window_len,
                                               req, nb_taxis,
                                               data,
                                               center_zone, center_val)
                    fcfs_obj_values.append(round(fcfs_cumul_obj_val, 2))
                    fcfs_non_profit.append(round(uf.dur_non_profit_trip(fcfs_m_task, data, center_val), 2))
                    fcfs_cpus.append(fcfs_cpu)
                    fcfs_unserv_perc.append(round(len(fcfs_unserv_req) / len(data), 2) * 100)
                    print("\n--> fcfs_cpus : ", fcfs_cpus)
                    print("--> fcfs_obj_values : ", fcfs_obj_values)
                    print("--> fcfs_unserv_perc : ", fcfs_unserv_perc)

                    # -----------------------------------------------------------------
                    """Instance statistics """
                    fcfs_total_peak_morning = data.loc[data.pick_t.between(360, 600),
                                                       'pick_t'].count()
                    fcfs_ave_morning = round(fcfs_total_peak_morning / 6, 2)
                    fcfs_averages_morning.append(fcfs_ave_morning)

                    fcfs_total_peak_evening = data.loc[data.pick_t.between(1020, 1260),
                                                       'pick_t'].count()
                    fcfs_ave_evening = round(fcfs_total_peak_evening / 6, 2)
                    fcfs_averages_evening.append(fcfs_ave_evening)
                    # -----------------------------------------------------------------

                fcfs_avg_unserv.append(round(statistics.mean(fcfs_unserv_perc), 2))
                fcfs_avg_obj.append(round(statistics.mean(fcfs_obj_values), 2))
                fcfs_avg_profit.append(round(statistics.mean(fcfs_non_profit), 2))
                fcfs_avg_cpu.append(round(statistics.mean(fcfs_cpus), 2))
            except Exception:
                print('error with window length:', window_len)
                fcfs_obj_values, fcfs_unserv_perc, fcfs_cpus = None, None, None
                if matrix_result:
                    fcfs_m_task, fcfs_m_st, fcfs_m_ft = None, None, None
            if result:
                self.__test_results['Objective values'][i] = fcfs_obj_values
                self.__test_results['% unserved requests'][i] = fcfs_unserv_perc
                self.__test_results['GA CPU time'][i] = fcfs_cpus
                self.__test_results['window length'][i] = window_len
                self.__test_results['Heuristic'][i] = 'FCFS'
                if matrix_result:
                    self.__test_results['matrix task'][i] = fcfs_m_task
                    self.__test_results['matrix start time'][i] = fcfs_m_st
                    self.__test_results['matrix finish time'][i] = fcfs_m_ft
                i += 1

        print("\n\n--> fcfs_avg_unserv : ", fcfs_avg_unserv)
        print("-->  fcfs_avg_obj : ", fcfs_avg_obj, )
        print("-->  fcfs_avg_cpu : ", fcfs_avg_cpu)

    def get_test_results(self):
        return self.__test_results

    def fcfs_heuristic(self, av_req, nb_taxis, battery_level, m_task, m_st, m_ft, data, center_zone, center_val, t_sup):
        """the fcfs heuristic schedules requests in a "first-come first-serve"
        fashion, according to requests start time (static case)

        the FCFS (First-Come, First-Served) heuristic schedules the requests according
        to requests start time, i.e., the realiest requests are the first to be scheduled
        input :
               * av_req        : dictionary of available requests
                                 the keys represent the requests index
                                 the values are the pick-up times
               * nb_taxis      : number of available taxis
               * battery_level : list for battery level of each taxi
               * m_task        : a python dictionary representing the tasks of each taxi
                                 the key represents the taxi
                                 the values are a list of tasks for the taxi
               * ma_st         : a python dictionary representing the start time of each task.
                                 The key represents the taxi, and the values are a list
                                 of starting time of each task
               * m_ft          : a python dictionary representing the finishing time of each task.
               * data          : pandas dataframe representing the instance
               * center_zone   : zone of the recharging center
               * center_val    : [x,y]-coordinates of the recharging center
               * t_sup         : upper value of the scheduling horizon (1440 minutes)
        return :
               * serv_req      : list of served requests
               * unserv_req    : dictionary of unserved requests
                                 the keys represent the requests index, and the values are
                                 the pick-up timers a time inside the request interval)
               * m_task        : the previous matrix_start_time with updates
               * m_st          : the previous matrix_task with updates
               * m_ft          : the previous matrix_fini_time with updates
               * battery_level : updates of the battery level.
        """

        # sorting requests according to earliest pickup time (fcfs fashion)
        av_req = dict(sorted(av_req.items(), key=operator.itemgetter(1)))
        av_req_copy = av_req.copy()

        # served requests initialization
        serv_req = {}
        prev_req_data_idx = None
        prev_req = 'c'
        for i_req in list(av_req_copy.keys()):

            # i_req index in data table
            i_req_data_idx = data.index[data['req_id'] == i_req].tolist()[0]
            dur_prep = uf.cal_dur_move_time(center_val, data, i_req_data_idx, -1)

            # --------------------------------------------------------------------
            # choose the first available taxi
            temp = [key for key, value in m_ft.items() if value == []]
            if temp:
                taxi = temp[0]
                j_taxi = int(taxi[4:])
                temp.remove(taxi)
            else:
                temp = {key: m_ft[key][-1] for key in m_ft.keys()}
                j_taxi = int(min(temp, key=temp.get)[4:])
            # -----------------------------------------------------------------

            prev_fini_time = 0
            if len(m_task["taxi" + str(j_taxi)]) != 0:
                prev_req = m_task["taxi" + str(j_taxi)][-1]
                if prev_req != "b":
                    prev_req_data_idx = data.index[data['req_id'] == prev_req].tolist()[0]
                    dur_prep = uf.cal_dur_move_time(center_val, data, i_req_data_idx, prev_req_data_idx)

                prev_fini_time = m_ft["taxi" + str(j_taxi)][-1]

            # Check enough time to pick up
            enough_time_to_pick = False
            if prev_fini_time + dur_prep <= data["pick_t"][i_req_data_idx]:
                enough_time_to_pick = True

            dur_t_req = data["dur_t"][i_req_data_idx]
            dur_back_center = uf.cal_dur_back_to_center(i_req_data_idx, center_val, data)

            # Check enough battery to serve and go back to center if needed
            enough_battery = False
            if battery_level[j_taxi - 1] - uf.bcr * (dur_prep + dur_t_req + dur_back_center) >= uf.b_min:
                enough_battery = True

            # check if it's possible te serve the request
            if enough_time_to_pick:
                if enough_battery:
                    if data["pick_t"][i_req_data_idx] + dur_t_req <= t_sup:
                        m_task["taxi" + str(j_taxi)].append(i_req)
                        m_st["taxi" + str(j_taxi)].append(round(data["pick_t"][i_req_data_idx], 2))
                        m_ft["taxi" + str(j_taxi)].append(round((data["pick_t"][i_req_data_idx] + dur_t_req), 2))
                        battery_level[j_taxi - 1] = round(
                            battery_level[j_taxi - 1] - uf.bcr * (dur_prep + dur_t_req),
                            2)

                        # remove the assigned demand from av_requests
                        serv_req[i_req] = av_req[i_req]
                        del av_req[i_req]

                else:  # charge battery if battery level is not enough
                    m_task["taxi" + str(j_taxi)].append("b")
                    dur_prev_to_center = uf.cal_dur_back_to_center(prev_req_data_idx, center_val, data)
                    m_st["taxi" + str(j_taxi)].append(round(prev_fini_time + dur_prev_to_center, 2))
                    m_ft["taxi" + str(j_taxi)].append(round(prev_fini_time + dur_prev_to_center + uf.R, 2))
                    battery_level[j_taxi - 1] = 100

                    dur_prep1 = uf.cal_dur_move_time(center_val, data, i_req_data_idx)
                    prev_fini_time1 = m_ft["taxi" + str(j_taxi)][-1]

                    # ----------------------------------------------------------
                    # check if it's possible to serve in the time interval [early_t, late_t]
                    if ((data["late_t"][i_req_data_idx] >= prev_fini_time1 + dur_prep1 >=
                         data["early_t"][i_req_data_idx])
                            &
                            (battery_level[j_taxi - 1] - uf.bcr * (dur_prep1 + dur_t_req + dur_back_center) >=
                             uf.b_min)):
                        if data["pick_t"][i_req_data_idx] + dur_t_req <= t_sup:
                            m_task["taxi" + str(j_taxi)].append(i_req)
                            # new pick_t: serve the requests as soon as the taxi
                            # finishes its previous task
                            i_req_new_pick_time = round(prev_fini_time1 + dur_prep1, 2)
                            m_st["taxi" + str(j_taxi)].append(i_req_new_pick_time)
                            m_ft["taxi" + str(j_taxi)].append(round((i_req_new_pick_time + dur_t_req), 2))
                            battery_level[j_taxi - 1] = round(battery_level[j_taxi - 1] - uf.bcr * (dur_prep1 +
                                                                                                    dur_t_req), 2)
                            # remove the assigned demand from av_req
                            serv_req[i_req] = av_req[i_req]
                            del av_req[i_req]

            elif ((data["late_t"][i_req_data_idx] >= prev_fini_time + dur_prep >= data["early_t"][i_req_data_idx])
                  &
                  (battery_level[j_taxi - 1] - uf.bcr * (dur_prep + dur_t_req + dur_back_center) >=
                   uf.b_min)):
                # new pick_t: serve the requests as soon as the taxi
                # finishes its previous task
                i_req_new_pick_time = round(prev_fini_time + dur_prep, 2)
                if i_req_new_pick_time + dur_t_req <= t_sup:
                    m_task["taxi" + str(j_taxi)].append(i_req)
                    m_st["taxi" + str(j_taxi)].append(i_req_new_pick_time)
                    m_ft["taxi" + str(j_taxi)].append(round((i_req_new_pick_time + dur_t_req), 2))
                    battery_level[j_taxi - 1] = round(battery_level[j_taxi - 1] -
                                                      uf.bcr * (dur_prep + dur_t_req), 2)

                    serv_req[i_req] = av_req[i_req]
                    del av_req[i_req]  # remove the assigned demand from av_req

        unserv_req = av_req  # update unserved demands
        return serv_req, unserv_req, m_task, m_st, m_ft, battery_level
