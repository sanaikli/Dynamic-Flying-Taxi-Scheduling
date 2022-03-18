About this project:

This project contains the implementations and the test instances used in our article entitled “A rolling horizon approach for the dynamic scheduling of flying taxis”. 

* All of the material provided in this project is freely available for use.

* Readers may refer to our original paper “A rolling horizon approach for the dynamic scheduling of flying taxis” (details about this paper will be available soon) for further details on the material provided in this page.

There are currently five python files in the directory named “src”:
* dynamic_fcfs.py: is a python file that contains the implementation of the First-Come, First-Served heuristic
* dynamic_ga2.py: contains the implementation of a genetic algorithm
* dynamic_nn.py: implements the Nearest Neighbor heuristic
* dynamic_nn_zones.py: contains the implementation of the improved nearest neighbor heuristic
* utility_functions.py: is a python file containing all the utility functions that are frequently used in the above-mentioned files

The instances used in our original paper are provided in the “.txt” files, from the directory “src”. In each instance, the following information is provided, for each customer request:

* The request identifier (req_id).
* The origin point of the request in the x-axis (ori_x) and y-axis (ori_y).
* The destination point of the request in the x-axis (des_x) and y-axis (des_y).
* The pick-up time (pick_t) of the request.
* The earliest (early_t) and the latest(late_t) acceptable times of the request.
* The distance (dist_t) between the origin and the destination points of the request, and its duration (dur_t).

The above-mentioned information represents the input parameters of the problem. The decision variables and the complete mathematical model are presented in our original paper.




