# -*- coding: utf-8 -*-
"""
Main programm.
"""
__date__ = '01/28/2021'

# -----      important packages    ------
import dynamic_ga2
import dynamic_fcfs
import dynamic_nn
import dynamic_nn_zones
import pandas as pd

# ___________________            main          _______________________________
if __name__ == "__main__":
    with_result = True  # to save the results in a csv file
    with_matrix_result = True  # results with matrix task, matrix start time, matrix finish time
    heuristics = ['nn', 'nn_zones']  # possibility: ['ga2', 'fcfs', 'nn', 'nn_zones']
    # windows = [90, 180, 240, 360, 480, 540, 720, 840, 1440]
    instances_list_panwadee = ["instances/instance10_2.txt", "instances/instance20_2.txt",
                               "instances/instance20_3.txt", "instances/instance30_2.txt",
                               "instances/instance30_3.txt", "instances/instance30_4.txt",
                               "instances/instance50_2.txt", "instances/instance50_3.txt",
                               "instances/instance50_5.txt", "instances/instance100_3.txt",
                               "instances/instance100_5.txt"]
    instances_list_sana = ["new_instances/instance50_2.txt", "new_instances/instance100_3.txt",
                           "new_instances/instance100_5.txt", "new_instances/instance250_5.txt",
                           "new_instances/instance250_10.txt", "new_instances/instance500_4.txt",
                           "new_instances/instance500_10.txt", "new_instances/instance1000_9.txt",
                           "new_instances/instance1000_15.txt", "new_instances/instance10000_20.txt"]

    windows = [90, 480]
    df_results = None
    # ---- Algorithm input Panwadee ---
    instance_orig = "panwadee"
    if instance_orig == "sana":
        instances_list = instances_list_sana[0:1]
    else:
        instances_list = instances_list_panwadee[6:7]
    if 'ga2' in heuristics:
        ga = dynamic_ga2.DynamicGa2(windows, instances_list, with_result, with_matrix_result,
                                    instance_orig).get_test_results()
        if with_result:
            if isinstance(df_results, pd.DataFrame):
                df_results = pd.concat([df_results, ga], ignore_index=True)
            else:
                df_results = ga

    # ---- Algorithm input Sana ---
    instance_orig = "sana"
    if instance_orig == "sana":
        instances_list = instances_list_sana[0:1]
    else:
        instances_list = instances_list_panwadee[0:1]
    if 'fcfs' in heuristics:
        fcfs = dynamic_fcfs.DynamicFCFS(windows, instances_list, with_result, with_matrix_result,
                                        instance_orig).get_test_results()
        if with_result:
            if isinstance(df_results, pd.DataFrame):
                df_results = pd.concat([df_results, fcfs], ignore_index=True)
            else:
                df_results = fcfs
    if 'nn' in heuristics:
        nn = dynamic_nn.DynamicNN(windows, instances_list, with_result, with_matrix_result,
                                  instance_orig).get_test_results()
        if with_result:
            if isinstance(df_results, pd.DataFrame):
                df_results = pd.concat([df_results, nn], ignore_index=True)
            else:
                df_results = nn
    if 'nn_zones' in heuristics:
        nn_zones = dynamic_nn_zones.DynamicNNZones(windows, instances_list, with_result, with_matrix_result,
                                                   instance_orig).get_test_results()
        if with_result:
            if isinstance(df_results, pd.DataFrame):
                df_results = pd.concat([df_results, nn_zones], ignore_index=True)
            else:
                df_results = nn_zones

    df_results.to_csv('Results.csv')
    print('end')
