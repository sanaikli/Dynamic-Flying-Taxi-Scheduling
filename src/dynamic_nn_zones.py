# -*- coding: utf-8 -*-
""" This program implements the NN (Nearest Neighbor) heuristic
for the dynamic flying taxis scheduling """
__author__ = 'saikli'
__date__ = '12/08/2021'

# -----      important packages    ------
import utility_functions as uf
import statistics
import math


# import cProfile
# import pstats
# ---------------------------------------


def nn_heuristic(av_req, nb_taxis, battery_level, matrix_task,
                 matrix_start_time, matrix_fini_time, data, center_zone, t_sup):
    """The NN (Nearest Neighbor) heuristic schedules the requests according to their proximity
    to taxi locations, i.e., the closest requests are the first to be scheduled
    input :
           * av_req           : dictionary of available requests
                                the keys represent the requests index
                                the values are the pick-up times
           * nb_taxis         : number of available taxis
           * battery_level    : list for battery level of each taxi
           * matrix_task      : a python dictionary representing the tasks of each taxi
                                the key represents the taxi
                                the values are a list of tasks for the taxi
           * matrix_start_time: a python dictionary representing the start time
                                of each task.
                                the key represents the taxi
                                the values are a list of starting time of each task
           * matrix_fini_time  : a python dictionary representing the finishing time
                                 of each task.
                                 the key represents the taxi
                                 the values are a list of finishing time of the request
           * data              : pandas dataframe representing the instance
           * t_sup             : upper value of the scheduling horizon (1440 minutes)
    return :
           * serv_req          : dictionary of served requests
                                 the keys represent the requests index
                                 the values are the serve times (pick-up time
           * unserv_req        : dictionary of unserved requests
                                 the keys represent the requests index
                                 the values are the pick-up times
                                 or a time inside the request interval)
           * matrix_task       : the previous matrix_start_time with updates
           * matrix_start_time : the previous matrix_task with updates
           * matrix_fini_time  : the previous matrix_fini_time with updates
           * battery_level     : updates of the battery level.
    """

    # create a copy of av_req, because the latter changes in the
    # For-Loop
    av_req_copy = av_req.copy()

    # served requests initialization
    serv_req = {}
    prev_req = 'c'
    for i in range(len(av_req_copy)):
        # choose the first available taxi
        # i.e., the one with the smallest fini_time
        temp = [key for key, value in matrix_fini_time.items() if value == []]
        if temp:
            taxi = temp[0]
            j_taxi = int(taxi[4:])
            temp.remove(taxi)
        else:
            temp = {key: matrix_fini_time[key][-1] for key in matrix_fini_time.keys()}
            j_taxi = int(min(temp, key=temp.get)[4:])

        # current taxi location = center (at start)
        if not matrix_task['taxi' + str(j_taxi)]:
            current_taxi_loc = center_val

            # --------- NN per zone
            current_taxi_zone = center_zone
            # ------------------
            # print("current taxi location: ", current_taxi_loc)
        # current taxi location = last request destination
        # (or center location in case of battery charging)
        else:
            last_req = matrix_task['taxi' + str(j_taxi)][-1]
            if last_req == 'b':
                current_taxi_loc = center_val

                # --------- NN per zone
                current_taxi_zone = center_zone
                # ------------------
            else:
                last_req_data_idx = data.index[data['req_id'] == last_req].tolist()[0]
                current_taxi_loc = [data.des_x[last_req_data_idx], data.des_y[last_req_data_idx]]

                # --------- NN per zone
                current_taxi_zone = data.zone_des[last_req_data_idx]
                # ------------------

        # prints
        # print("\n j_taxi = ", j_taxi,
        #       " taxi loc = ", current_taxi_loc,
        #       " taxi zone = ", current_taxi_zone)
        # for i in av_req.keys():
        #     print(" current i zone = ",
        #           data.zone_ori[data.index[data['req_id'] == i].tolist()[0]],
        #           " x = ", data.ori_x[data.index[data['req_id'] == i].tolist()[0]] ,
        #           " y = ", data.ori_y[data.index[data['req_id'] == i].tolist()[0]])
        # --------- NN per zone
        # distances between the current taxi and requests location
        # that belong to the same zone
        distances = {i: round(math.sqrt(
            (data.ori_x[data.index[data['req_id'] == i].tolist()[0]] - current_taxi_loc[0]) ** 2 +
            (data.ori_y[data.index[data['req_id'] == i].tolist()[0]] - current_taxi_loc[1]) ** 2), 2)
            for i in av_req.keys()
            if data.zone_ori[data.index[data['req_id'] == i].tolist()[0]] == current_taxi_zone}

        if distances == {}:
            distances = {i: round(math.sqrt(
                (data.ori_x[data.index[data['req_id'] == i].tolist()[0]] - current_taxi_loc[0]) ** 2 +
                (data.ori_y[data.index[data['req_id'] == i].tolist()[0]] - current_taxi_loc[1]) ** 2), 2)
                for i in av_req.keys()}
        # print("\n * distances keys : ", distances.keys())
        # ------------------

        i_req = min(distances, key=distances.get)
        i_req_data_idx = data.index[data['req_id'] == i_req].tolist()[0]
        dur_prep = uf.cal_dur_move_time(center_val, data, i_req_data_idx, -1)

        prev_fini_time = 0
        if len(matrix_task["taxi" + str(j_taxi)]) != 0:
            prev_req = matrix_task["taxi" + str(j_taxi)][-1]
            prev_fini_time = matrix_fini_time["taxi" + str(j_taxi)][-1]
            if prev_req != "b":
                prev_req_data_idx = data.index[data['req_id'] ==
                                               prev_req].tolist()[0]
                dur_prep = uf.cal_dur_move_time(center_val, data,
                                                i_req_data_idx, prev_req_data_idx)
        # Check enough time to serve at pick-up time
        enough_time_to_pick = False
        if prev_fini_time + dur_prep <= data["pick_t"][i_req_data_idx]:
            enough_time_to_pick = True

        # Check enough battery to serve (at pick-up time) and go back to center if needed
        dur_t_req = data["dur_t"][i_req_data_idx]
        dur_back_center = uf.cal_dur_back_to_center(i_req_data_idx, center_val, data)
        enough_battery = False
        if battery_level[j_taxi - 1] - uf.bcr * (dur_prep + dur_t_req + dur_back_center) >= uf.b_min:
            enough_battery = True

        # check if it's possible te serve the request in its interval if
        # enough_time and battery to pick it up
        if enough_time_to_pick:
            if enough_battery:
                if data["pick_t"][i_req_data_idx] + dur_t_req <= t_sup:
                    matrix_task["taxi" + str(j_taxi)].append(i_req)
                    matrix_start_time["taxi" + str(j_taxi)].append(data["pick_t"][i_req_data_idx])
                    matrix_fini_time["taxi" + str(j_taxi)].append(
                        round((data["pick_t"][i_req_data_idx] + dur_t_req), 2))
                    battery_level[j_taxi - 1] = round(battery_level[j_taxi - 1] - uf.bcr * (dur_prep + dur_t_req), 2)

                    # remove the assigned demand from av_requests
                    serv_req[i_req] = av_req[i_req]
                    del av_req[i_req]

            else:  # charge the battery otherwise
                matrix_task["taxi" + str(j_taxi)].append("b")
                dur_prev_to_center = uf.cal_dur_back_to_center(prev_req_data_idx,
                                                               center_val, data)
                matrix_start_time["taxi" + str(j_taxi)].append(prev_fini_time + dur_prev_to_center)
                matrix_fini_time["taxi" + str(j_taxi)].append(prev_fini_time + dur_prev_to_center + uf.R)
                battery_level[j_taxi - 1] = 100
                dur_prep1 = uf.cal_dur_move_time(center_val, data, i_req_data_idx)
                prev_fini_time1 = matrix_fini_time["taxi" + str(j_taxi)][-1]

                # ----------------------------------------------------------
                # check if it's possible to serve in the time interval [early_t, late_t]
                # after battery charging
                if ((prev_fini_time1 + dur_prep1 <= data["late_t"][i_req_data_idx])
                        &
                        (prev_fini_time1 + dur_prep1 >= data["early_t"][i_req_data_idx])
                        &
                        (battery_level[j_taxi - 1] - uf.bcr * (dur_prep1 + dur_t_req + dur_back_center) >= uf.b_min)):
                    # new pick_t: serve the requests as soon as the taxi
                    # finiches its previous task
                    i_req_new_pick_time = round(prev_fini_time1 + dur_prep1, 2)
                    if i_req_new_pick_time + dur_t_req <= t_sup:
                        matrix_task["taxi" + str(j_taxi)].append(i_req)
                        # ---------------------------------------------------

                        matrix_start_time["taxi" + str(j_taxi)].append(i_req_new_pick_time)
                        matrix_fini_time["taxi" + str(j_taxi)].append(round((i_req_new_pick_time +
                                                                             dur_t_req), 2))
                        battery_level[j_taxi - 1] = round(battery_level[j_taxi - 1] - uf.bcr * (dur_prep1 +
                                                                                                dur_t_req), 2)
                        # remove the assigned demand from av_req
                        serv_req[i_req] = av_req[i_req]
                        del av_req[i_req]

        elif ((prev_fini_time + dur_prep <= data["late_t"][i_req_data_idx])
              &
              (prev_fini_time + dur_prep >= data["early_t"][i_req_data_idx])
              &
              (battery_level[j_taxi - 1] - uf.bcr * (dur_prep + dur_t_req + dur_back_center) >= uf.b_min)):
            # new pick_t: serve the requests as soon as the taxi
            # finiches its previous task
            i_req_new_pick_time = round(prev_fini_time + dur_prep, 2)
            if i_req_new_pick_time + dur_t_req <= t_sup:
                matrix_task["taxi" + str(j_taxi)].append(i_req)

                matrix_start_time["taxi" + str(j_taxi)].append(i_req_new_pick_time)
                matrix_fini_time["taxi" + str(j_taxi)].append(round((i_req_new_pick_time +
                                                                     dur_t_req), 2))
                battery_level[j_taxi - 1] = round(battery_level[j_taxi - 1] - uf.bcr * (dur_prep +
                                                                                        dur_t_req), 2)
                # remove the assigned demand from av_req
                serv_req[i_req] = av_req[i_req]
                del av_req[i_req]

    unserv_req = av_req  # update unserved demands
    return serv_req, unserv_req, matrix_task, matrix_start_time, matrix_fini_time, battery_level


# Main program for numerical tests
if __name__ == "__main__":

    # RH parameters
    t_inf, t_sup = 0, 1440  # start and end times (respectively)
    # of the day (in minutes) : 1440 = 24h
    # the scheduling horizon T=[0, 1440]

    instances = ["new_instances/instance50_2.txt", "new_instances/instance100_3.txt",
                 "new_instances/instance100_5.txt", "new_instances/instance250_5.txt",
                 "new_instances/instance250_10.txt", "new_instances/instance500_4.txt",
                 "new_instances/instance500_10.txt", "new_instances/instance1000_9.txt",
                 "new_instances/instance1000_15.txt", "new_instances/instance10000_20.txt"]

    nn_avg_unserv = []
    nn_avg_obj = []
    nn_avg_profit = []
    nn_avg_cpu = []

    instances = ["new_instances/instance50_2.txt", "new_instances/instance100_3.txt",
                 "new_instances/instance100_5.txt", "new_instances/instance250_5.txt",
                 "new_instances/instance250_10.txt", "new_instances/instance500_4.txt",
                 "new_instances/instance500_10.txt"]

    window_lengths = [60]  # , 180, 360, 720, 1440]
    for window_len in window_lengths:
        print("\n\n________________  window_len =  ", window_len, "________________")
        argument = 'w+'
        nn_obj_values = []
        nn_non_profit = []
        nn_cpus = []
        nn_unserv_perc = []  # percentage of unserved demands
        nn_averages_morning, nn_averages_evening = [], []
        for instance_name in instances:
            print("\n*********", instance_name)
            # loading data from Panwadee's data generator
            nb_req, nb_taxis, data, center_val = uf.prepare_data(instance_name,
                                                                 instance_orig='sana')
            # data=data[:20]
            data.req_id = data.req_id.astype(int)

            # ----------------------------------------------------------------
            # data with zones
            n_zone_x, n_zone_y = 4, 5
            zone_id, zone_coord, zone_ori, zone_des, data, center_zone = uf.define_zone(data,
                                                                                        center_val,
                                                                                        n_zone_x,
                                                                                        n_zone_y)
            # ----------------------------------------------------------------

            # all (known) requests and their pick-up times
            # in the beginning of the scheduling horizon
            req = {data.req_id[i]: data.pick_t[i] for i in data.index}

            # available taxis in the beginning of the scheduling horizon
            # av_taxis = [i for i in range(1, nb_taxis + 1)]

            # rolling_time_window call with FCFS heuristic
            nn_cpu, nn_cumul_obj_val, nn_serv_req, nn_unserv_req, nn_matrix_task, nn_matrix_start_time, nn_matrix_fini_time = \
                uf.rolling_time_window(nn_heuristic,
                                       t_inf, t_sup,
                                       window_len,
                                       req, nb_taxis,
                                       data,
                                       center_zone)
            # -------------------------------------------------------------
            # cProfile to analyze the code
            # cProfile.run('uf.rolling_time_window(nn_heuristic,t_inf,t_sup,window_len,req, nb_taxis,data)',
            #              'output.dat')
            # with open("output_time.txt", "w") as f:
            #     p = pstats.Stats("output.dat", stream = f)
            #     p.sort_stats("time").print_stats()
            #
            # with open("output_calls.txt", "w") as f:
            #     p = pstats.Stats("output.dat", stream = f)
            #     p.sort_stats("calls").print_stats()

            # -------------------------------------------------------------

            nn_obj_values.append(round(nn_cumul_obj_val, 2))
            nn_non_profit.append(round(uf.dur_non_profit_trip(nn_matrix_task,
                                                              data, center_val), 2))
            nn_cpus.append(nn_cpu)
            nn_unserv_perc.append(round(len(nn_unserv_req) / len(data), 2) * 100)

            # save results in ".txt" file named
            # "results_<dynamic/static>_nn.txt
            if window_len == 1440:
                file_name = "results_nn_static.txt"
            else:
                file_name = "results_nn_rh.txt"
            with open(file_name, argument) as f:
                f.write("\n\n\n--------\t" + str(instance_name) + "\t--------")
                f.write("\n* matrix_task : " + str(nn_matrix_task))
                f.write("\n* unserved req : " + str(nn_unserv_req))
                f.write("\n* objective value : " + str(round(nn_cumul_obj_val, 2)))
                f.write("\n* cpu time : " + str(nn_cpu))
            argument = 'a'
            # print("\nmatrix task : ", nn_matrix_task)
            print("\n* objective value : ", round(nn_cumul_obj_val, 2), " (total service time)")
            print("* cpu : ", round(nn_cpu, 4))
            print("* percentage of unserved requests : ",
                  round((len(nn_unserv_req) / len(data)) * 100, 2), " %")

            # -----------------------------------------------------------------
            # Instance statistics
            nn_total_peak_morning = data.loc[data.pick_t.between(360, 600),
                                             'pick_t'].count()
            nn_ave_morning = round(nn_total_peak_morning / 6, 2)
            nn_averages_morning.append(nn_ave_morning)

            nn_total_peak_evening = data.loc[data.pick_t.between(1020, 1260),
                                             'pick_t'].count()
            nn_ave_evening = round(nn_total_peak_evening / 6, 2)
            nn_averages_evening.append(nn_ave_evening)
            # -----------------------------------------------------------------

        nn_avg_unserv.append(round(statistics.mean(nn_unserv_perc), 2))
        nn_avg_obj.append(round(statistics.mean(nn_obj_values), 2))
        nn_avg_profit.append(round(statistics.mean(nn_non_profit), 2))
        nn_avg_cpu.append(round(statistics.mean(nn_cpus), 2))

    print("\n\n--> nn_avg_unserv : ", nn_avg_unserv)
    print("--> nn_avg_obj : ", nn_avg_obj, )
    print("--> nn_avg_profit : ", nn_avg_profit)
    print("--> nn_avg_cpu : ", nn_avg_cpu)
