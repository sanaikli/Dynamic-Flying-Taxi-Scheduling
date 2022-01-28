# -*- coding: utf-8 -*-
"""
This program contains the implementation of the Genetic Algorithm (GA)
of Panwadee, adatpted to the Rolling Horizon approach (RH).
The RH is implemented in the function << rh_ga2 >>
"""
__date__ = '11/15/2021'
__author__ = 'saikli'

# -----      important packages    ------
import utility_functions as uf
import numpy as np
import time
import copy
import pandas as pd
# import cProfile
# import pstats


class DynamicGa2:
    """Genetic Algorithm (GA)"""
    def __init__(self, window_lengths, instances, result, matrix_result, instance_orig):
        """Main program for numerical tests

        input :
           * window_len   : length of the (rolling) window
           * instances    : list of instance_name
           * result       : boolean, True to save the results in a csv file

        output: (if result == True)
            * self.__test_results : results DataFrame
        """

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
        self.t_inf, self.t_sup = 0, 1440  # start and end times of the day (in minutes)

        print("\n\n________________  Heuristic =  GA  ________________")
        for window_len in window_lengths:
            print("\n\n__________  window length =  ", window_len, " min _____")
            try:
                # argument = 'w+'  # For Test
                ga_obj_values = []
                ga_non_profit = []
                ga_cpus = []
                ga_unserv_perc = []  # percentage of unserved demands
                # ga_averages_morning, fcfs_averages_evening = [], []  # For Test

                for instance_name in instances:
                    print("\n--->", instance_name)
                    nb_req, self.nb_taxis, all_data, center_val = uf.prepare_data(instance_name, instance_orig)
                    # all_data=all_data[:6]
                    all_data.req_id = all_data.req_id.astype(int)

                    for i_run in range(1, 2):
                        cpu, s_req, uns_req = self.rh_ga2(window_len,
                                                          all_data,
                                                          center_val)

                        # -------------------------------------------------------------
                        # cProfile to analyze the code
                        # cProfile.run('rh_ga2(t_inf, t_sup, window_len, av_taxis,data, center_val)',
                        #                 'ga_output.dat')
                        # with open("ga_output_time.txt", "w") as f:
                        #     p = pstats.Stats("ga_output.dat", stream = f)
                        #     p.sort_stats("time").print_stats()
                        #
                        # with open("ga_output_calls.txt", "w") as f:
                        #     p = pstats.Stats("ga_output.dat", stream = f)
                        #     p.sort_stats("calls").print_stats()
                        # -------------------------------------------------------------

                        ga_obj_values.append(round(self.obj_val, 2))
                        ga_cpus.append(round(cpu, 2))
                        ga_non_profit.append(round(uf.dur_non_profit_trip(self.m_task, all_data, center_val), 2))
                        ga_unserv_perc.append(round(len(uns_req) / len(all_data) * 100, 2))
            except Exception:
                print('error with window length:', window_len)
                ga_obj_values, ga_unserv_perc, ga_cpus = None, None, None
            if result:
                self.__test_results['Objective values'][i] = ga_obj_values
                self.__test_results['% unserved requests'][i] = ga_unserv_perc
                self.__test_results['GA CPU time'][i] = ga_cpus
                self.__test_results['window length'][i] = window_len
                self.__test_results['Heuristic'][i] = 'GA'
                if matrix_result:
                    self.__test_results['matrix task'][i] = self.m_task
                    self.__test_results['matrix start time'][i] = self.m_st
                    self.__test_results['matrix finish time'][i] = self.m_ft
                i += 1

        print("\n\n * Objective values : ", ga_obj_values)
        print(" * Percentage of unserved requests: ", ga_unserv_perc)
        print(" * GA CPU time : ", ga_cpus)

    def get_test_results(self):
        return self.__test_results

    def sorting_population(self, arr_pop, arr_res):
        """the function "sorting_population" sort a population of chromosomes according to the fitness function
        (value of the objective-function)
            * arr_pop : a python list representing the population (set of chromosomes)
            * arr_res : a python list of the objective-value of the corresponding chromosome
        output: * sorted population.
        """
        if not isinstance(arr_pop, np.ndarray):
            arr_pop = np.array(arr_pop)
        arr_res_sort = copy.deepcopy(arr_res)
        order = np.argsort(arr_res_sort)[::-1]  # order of the index of the objective value sorted in descending order
        arr_pop_sort = arr_pop[order]
        arr_res_sort.sort(reverse=True)
        return arr_pop_sort, arr_res_sort

    def decode_ga(self, chromo, center_val, all_data, detail):
        """the function "decode_ga" represents the decoding of the chromosome in the
        genetic algorithm proposed by Panwadee (function "genetic_algorithm");
        the chromosome is a python list; each element of the list represents
        a score corresponding to a request serviced by a flying taxi

        input :
               * chromo          : chromosome, which is a python list that represents the chromosome
                                  (Individual solution in the GA algorithm)
               * center_val      : [x,y]--coordinates (values) of the recharging center
               * all_data        : pandas dataframe with parameters for all requests
               * detail          : boolean parameter, True only for the last decode_ga
               * self.av_data    : available data, which is a pandas dataframe with parameters
                                    for the available requests
               * self.av_bl      : available battery level in the current horizon
               * self.m_task     : python dict that represents the matrix task (explained above)
               * self.m_st       : python dict representing the (matrix) start time of each task
               * self.m_ft       : python dictionary representing the finishing time of each task


        return: (if detail is False)
               * obj_val_chrom    : the value of the objective-function

        return: (if detail is True)
               * serv          : list of the served requests
        output: (if detail is True)
               * self.av_bl     : updated battery level
               * self.m_task    : updated matrix task (with new served requests)
               * self.m_st      : updated matrix start time
               * self.m_ft      : updated matrix finished time
               * self.obj_val   : the value of the objective-function
        """
        # matrix initializations
        self.new_m_task = copy.deepcopy(self.m_task)
        self.new_m_st = copy.deepcopy(self.m_st)
        self.new_m_ft = copy.deepcopy(self.m_ft)
        self.new_bl = copy.deepcopy(self.av_bl)
        # ----------------------------------------------
        self.obj_val_chrom = 0
        self.na_i_gene = []  # request list already used
        self.serv_req = []  # list of the served requests
        if not isinstance(chromo, np.ndarray):
            chromo = np.array(chromo)
        chrom_priority = ((uf.T - self.av_data["pick_t"][np.arange(len(chromo)) // self.nb_taxis]) / uf.T) *\
                         ((1 + chromo) / 2)
        chrom_priority = chrom_priority.to_list()

        # End of ideal priority combination

        order_assign = sorted(range(len(chrom_priority)),
                              key=lambda k: chrom_priority[k], reverse=True)
        # print("\n *********** order_assign = ", order_assign[:9],"...")
        for i_gene in order_assign:
            if i_gene not in self.na_i_gene:
                i_req = (i_gene // self.nb_taxis) + 1
                j_taxi = (i_gene % self.nb_taxis) + 1

                # ----------------------------------------------
                req_id = self.av_data.req_id[i_req - 1]
                req_id_data_idx = self.av_data.loc[self.av_data['req_id'] == req_id].index[0]
                # print("\n * req_id  =  ", req_id, " * j_taxi = ", j_taxi)
                # print(" * pick_t: ", self.av_data.pick_t[req_id_data_idx])
                # ----------------------------------------------
                dur_prep = uf.cal_dur_move_time(center_val, self.av_data, req_id_data_idx)
                prev_fini_time = 0
                prev_req = None
                prev_req_data_idx = None
                if len(self.new_m_task["taxi" + str(j_taxi)]) != 0:
                    prev_req = self.new_m_task["taxi" + str(j_taxi)][-1]
                    if prev_req != "b":
                        # ----------------------------------------------
                        # print(" prev_req :", prev_req)
                        prev_req_data_idx = all_data.loc[all_data['req_id'] == prev_req].index[0]
                        # ----------------------------------------------

                        dur_prep = uf.cal_dur_move_time(center_val, all_data, req_id_data_idx, prev_req_data_idx)
                    prev_fini_time = self.new_m_ft["taxi" + str(j_taxi)][-1]

                # Check enough time to pick up
                enough_time_to_pick = False
                if prev_fini_time + dur_prep <= self.av_data["pick_t"][req_id_data_idx]:
                    enough_time_to_pick = True

                dur_t_req = self.av_data["dur_t"][req_id_data_idx]
                dur_back_center = uf.cal_dur_back_to_center(req_id_data_idx, center_val, self.av_data)

                # Check enough battery to service and go back to center if needed
                enough_battery = False
                if self.new_bl[j_taxi - 1] - uf.bcr * (dur_prep + dur_t_req + dur_back_center) >= uf.b_min:
                    enough_battery = True

                # Possible to response the demand?
                if enough_time_to_pick:
                    if enough_battery:
                        if self.av_data["pick_t"][req_id_data_idx] + dur_t_req <= uf.T:
                            self.new_task(j_taxi, req_id, self.av_data["pick_t"][req_id_data_idx],
                                          self.av_data["pick_t"][req_id_data_idx] + dur_t_req,
                                          round(self.new_bl[j_taxi - 1] - uf.bcr * (dur_prep + dur_t_req), 2),
                                          i_req)
                    # ----------------------------------------------
                    # added by Sana for serving requests inside time-windows
                    else:  # charge battery if battery level is not enough
                        dur_prev_to_center = uf.cal_dur_back_to_center(prev_req_data_idx, center_val, all_data)
                        self.new_task(j_taxi, "b",
                                      prev_fini_time + dur_prev_to_center,
                                      prev_fini_time + dur_prev_to_center + uf.R,
                                      100, i_req)

                        dur_prep1 = uf.cal_dur_move_time(center_val, self.av_data, req_id_data_idx)
                        prev_fini_time1 = self.new_m_ft["taxi" + str(j_taxi)][-1]

                        # check if it's possible to serve in the time interval [early_t, late_t]
                        if ((self.av_data["late_t"][req_id_data_idx] >= prev_fini_time1 + dur_prep1 >=
                             self.av_data["early_t"][req_id_data_idx])
                                &
                                (self.new_bl[j_taxi - 1] - uf.bcr * (dur_prep1 + dur_t_req + dur_back_center) >=
                                 uf.b_min)):
                            if self.av_data["pick_t"][req_id_data_idx] + dur_t_req <= self.t_sup:
                                req_id_new_pick_time = round(prev_fini_time1 + dur_prep1, 2)
                                self.new_task(j_taxi, req_id, req_id_new_pick_time,
                                              req_id_new_pick_time + dur_t_req,
                                              round(self.new_bl[j_taxi - 1] - uf.bcr * (dur_prep1 + dur_t_req), 2),
                                              i_req)

                elif ((self.av_data["late_t"][req_id_data_idx] >= prev_fini_time + dur_prep >=
                       self.av_data["early_t"][req_id_data_idx])
                      &
                      (self.new_bl[j_taxi - 1] - uf.bcr * (dur_prep + dur_t_req + dur_back_center) >=
                       uf.b_min)):
                    req_id_new_pick_time = round(prev_fini_time + dur_prep, 2)
                    if req_id_new_pick_time + dur_t_req <= self.t_sup:
                        self.new_task(j_taxi, req_id, req_id_new_pick_time,
                                      req_id_new_pick_time + dur_t_req,
                                      round(self.new_bl[j_taxi - 1] - uf.bcr * (dur_prep + dur_t_req), 2),
                                      i_req)
                    # ----------------------------------------------
                # print(" * new_m_task [taxi",j_taxi,"] =", self.new_m_task["taxi"+str(j_taxi)][-1] )
                # print(" * new_m_st [taxi",j_taxi,"] =", round(self.new_m_st["taxi"+str(j_taxi)][-1], 2) )
                # print(" * new_m_ft [taxi",j_taxi,"] =", round(self.new_m_ft["taxi"+str(j_taxi)][-1], 2) )
        # print(" * new_m_task =", self.new_m_task)

        if detail:
            self.m_task = copy.deepcopy(self.new_m_task)
            self.m_st = copy.deepcopy(self.new_m_st)
            self.m_ft = copy.deepcopy(self.new_m_ft)
            self.av_bl = copy.deepcopy(self.new_bl)
            self.obj_val += self.obj_val_chrom
            return self.serv_req
        else:
            return self.obj_val_chrom

    def new_task(self, j_taxi, req_id, st, ft, bl, i_req):
        """the function, new_task is only called by "decode_ga", and implements the following variables:
        input :
               * j_taxi     : taxi number used
               * req_id     : request ID
               * st         : start time of request
               * ft         : finish time of request
               * bl         : battery level of request
               * i_req      : priority

        output:
               * self.new_m_task     : new matrix task
               * self.new_m_st       : new matrix start time
               * self.new_m_ft       : new matrix finished time
               * self.new_bl         : new battery level
               * self.obj_val_chrom  : new value of the objective-function
               * self.serv_req       : updated list of the served requests
               * self.na_req_id      : updated list of Not Available priority
        """
        self.new_m_task["taxi" + str(j_taxi)].append(req_id)
        self.new_m_st["taxi" + str(j_taxi)].append(round(st, 2))
        self.new_m_ft["taxi" + str(j_taxi)].append(round(ft, 2))
        self.new_bl[j_taxi - 1] = bl
        if req_id != "b":
            # implements the Second objective-function that maximizes the total service time input
            self.obj_val_chrom += ft - st
            # ----------------------------------------------
            # add req_id to served requests
            self.serv_req.append(req_id)
            # ----------------------------------------------
            # remove the assigned demand from the chromosome
            for j_count in range(self.nb_taxis):
                if j_count != j_taxi - 1:
                    elem_remove = self.nb_taxis * (i_req - 1) + j_count
                    self.na_i_gene.append(elem_remove)

    def genetic_algorithm(self, center_val, all_data):
        """the function "genetic_algorithm" the GA main loop, that performs cross-over and mutations
        to evolve the population in order to find the "best" chromosome/solution

        input :
               * center_val  : [x,y]--coordinates (values) of the charging center
               * all_data    : pandas dataframe representing the data for an instance

        return:
               * serv          : list of the served requests
        output:
               * self.obj_val   : new value of the objective-function
               * self.av_bl     : new battery level
               * self.m_task    : new matrix task
               * self.m_st      : new matrix start time of each request
               * self.m_ft      : new matrix finished time of each request
        """
        # ----------------------------------------------
        nb_req = len(self.av_data)
        chrom_size = nb_req * self.nb_taxis
        nb_chrom = 2 * chrom_size
        population = np.random.rand(nb_chrom, chrom_size)

        result_ga = []
        for chromosome in population:
            result_ga_elem = self.decode_ga(chromosome, center_val, all_data, False)
            result_ga.append(result_ga_elem)

        population, result_ga = self.sorting_population(population, result_ga)
        best_res = result_ga[0]
        stop_value = 30
        nb_select = int(0.1 * nb_chrom)
        nb_crossover = int(0.7 * nb_chrom)
        nb_mutation = nb_chrom - nb_select - nb_crossover
        n_not_improve = 0
        n_iteration = 0

        while n_not_improve < stop_value:
            new_pop = []
            if not isinstance(population, np.ndarray):
                population = np.array(population)
            new_pop.extend(list(population[:nb_select]))
            chrom_p1 = population[np.random.randint(nb_select, size=nb_crossover)]
            chrom_p2 = population[np.random.randint(nb_select + 1, high=nb_chrom - 1, size=nb_crossover)]
            crossover_chrom_temp = np.random.randint(10, size=(nb_crossover, chrom_size))
            crossover_chrom = (crossover_chrom_temp <= 6) * chrom_p1 + (crossover_chrom_temp > 6) * chrom_p2
            new_pop.extend(list(crossover_chrom))
            new_pop.extend(list(np.random.rand(nb_mutation, chrom_size)))
            population = new_pop
            result_ga = []
            for chromosome in population:
                result_ga_elem = self.decode_ga(chromosome, center_val, all_data, False)
                result_ga.append(result_ga_elem)

            population, result_ga = self.sorting_population(population, result_ga)
            best_res_iter = result_ga[0]
            if best_res_iter > best_res:
                best_res = best_res_iter
                n_not_improve = 0
            else:
                n_not_improve = n_not_improve + 1
            n_iteration = n_iteration + 1

        final_chrom = population[0]
        serv = self.decode_ga(final_chrom,
                              center_val,
                              all_data,
                              True)

        return serv

    def rh_ga2(self, window_len, all_data, center_val):
        """" "rh_ga2" implements the rolling-horizon approach for the genetic algorithm.
        It schedules the requests inside the first window using
        the GA. then, it moves the unserved requests to the next
        window and schedules them with the new requests ; this process continues
        until reaching the last time window

        input :
           * window_len   : length of the (rolling)  window
           * all_data     : pandas dataframe representing the data for an instance
           * center_val   : [x,y]--coordinates of the center

        return :
           * cpu          : cpu time
           * serv_req     : list of the served requests
           * av_req       : list of the (remaining) available requests (unserved requests)
        output:
           * self.t_inf, self.t_sup : upper and lower bounds on the scheduling horizon
                            [t_inf, t_sup] = [0, 1440](24 hours for example)
           * self.av_data : a pandas dataframe with the available requests data
           * self.m_task, self.m_st and self.m_ft : matrices of tasks as defined in the above function descriptions
           * self.av_bl : list of battery level as defined in the above function descriptions
           """
        av_req = []
        serv_req = []
        # matrix initializations
        self.m_init = {"taxi" + str(j): [] for j in range(1, self.nb_taxis + 1)}
        self.m_task = copy.deepcopy(self.m_init)
        self.m_st = copy.deepcopy(self.m_init)
        self.m_ft = copy.deepcopy(self.m_init)
        self.av_bl = [100] * self.nb_taxis  # battery level (100% for all taxis)
        self.obj_val = 0

        t_i = time.time()  # initial time (to compute the cpu time)

        for t in range(self.t_inf, self.t_sup, window_len):
            print("\n * t =", t)
            # data preparation
            av_req.extend([all_data.req_id[i] for i in all_data.index
                           if t + window_len > all_data.pick_t[i] >= t])
            self.av_data = all_data.loc[all_data['req_id'].isin(av_req)]
            self.av_data.reset_index(drop=True, inplace=True)
            print(" * available requests =", av_req)
            new_serv_req = self.genetic_algorithm(center_val,
                                                  all_data)

            # update requests list: available requests (av_req) and served requests (serv_req)
            av_req = [i for i in av_req if i not in new_serv_req]
            serv_req.extend(new_serv_req)

            # print(self.av_bl, self.m_task, self.m_st, self.m_ft)

        t_f = time.time()  # final time
        cpu = round(t_f - t_i, 4)  # cpu calculation
        print(" * unserved requests : ", av_req)
        return cpu, serv_req, av_req


# # ___________________            main          _______________________________
# if __name__ == "__main__":
#     ga_results = None
#     windows = [90, 180, 240, 360, 480, 540, 720, 840, 1440]
#     # instances_list = ["instances/instance10_2.txt", "instances/instance20_2.txt",
#     #                   "instances/instance20_3.txt", "instances/instance30_2.txt",
#     #                   "instances/instance30_3.txt", "instances/instance30_4.txt",
#     #                   "instances/instance50_2.txt", "instances/instance50_3.txt",
#     #                   "instances/instance50_5.txt", "instances/instance100_3.txt",
#     #                   "instances/instance100_5.txt"]
#
#     instances_list = ["instances/instance50_2.txt", "instances/instance100_3.txt"]
#     windows = windows[4:5]
#     with_result = False  # to save the results in a csv file
#     for window_lengths in windows:
#         try:
#             ga = DynamicGa2(window_lengths, instances_list, with_result)
#             if with_result:
#                 ga = ga.get_test_results()
#         except Exception:
#             print('error with windows_lengths:', window_lengths)
#             if with_result:
#                 ga = pd.DataFrame({'Objective values': [None],
#                                    '% unserved requests': [None],
#                                    'GA CPU time': [None],
#                                    'window lengths': [window_lengths]})
#         if with_result:
#             if isinstance(ga_results, pd.DataFrame):
#                 ga_results = pd.concat([ga_results, ga], ignore_index=True)
#             else:
#                 ga_results = ga
#     if with_result:
#         ga_results.to_csv('ga_results.csv')
