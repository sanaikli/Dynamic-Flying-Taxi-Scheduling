# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:10:28 2021

@author: saikli
"""

""" This program generates instances for the problem of flying taxi scheduling
the generator is based on Panwadee's code, but augmented with time windows for
each request, and specific peak hours for taxi demands"""

#-----      important packages    ------
import math
import random as rd
#---------------------------------------
# Parameters from Panwadee's
nb_req = 250
nb_taxi = 10
x_min = 0
y_min = 0
x_max = 24000
y_max = 30000
# st_max = 500
pick_t_max = 1440 #(24h)
x_center = round((x_min+x_max)/2)
y_center = round((y_min+y_max)/2)

#-----------------------------------------------------------------------------
# added by Sana
v_fly = 50 #in km/h
s_need = 5

# upper bounds on an interval
ub1, ub2, ub3, ub4 = 15, 30, 45, 60 # 15, 30, 45 and 60 minutes
f = 1 # delay or advance factors

# peak intervals for requests: [6h, 10h] and [17h, 21h]
# i.e., [360, 600] and [1020, 1260]
peak1_lb, peak1_ub = [360, 600] 
peak2_lb, peak2_ub = [1020, 1260]
perc1, perc2, perc3 = 0.5, 0.7, 0.8 # percentage of demands in peak intervals
#-----------------------------------------------------------------------------

print(nb_req)
file_name = "instance"+str(nb_req)+"_"+str(nb_taxi)+".txt"
text_file = open(file_name, "w")
text_file.write(str(nb_req)+' '+str(nb_taxi)+'\n')
text_file.write(str(x_center)+' '+str(y_center)+'\n')

for i in range(nb_req):
    ori_x = rd.randrange(x_min, x_max)
    ori_y = rd.randrange(y_min, y_max)
    dest_x = rd.randrange(x_min, x_max)
    dest_y = rd.randrange(y_min, y_max)
    
    #-------------------------------------------------------------------------
    # added by Sana
    if i >= perc1*nb_req:
        pick_t = rd.choice([rd.randrange(peak1_lb, peak1_ub),
                           rd.randrange(peak2_lb, peak2_ub)])
    else:
        pick_t = rd.choice([rd.randrange(0, peak1_lb),
                            rd.randrange(peak1_ub, peak2_lb),
                            rd.randrange(peak2_ub, pick_t_max)])
    
    dist  = round( math.sqrt((ori_x-dest_x)**2 + (ori_y-dest_y)**2), 2)
    dur_t = round(dist/(v_fly*1000/60)+2*s_need, 2)
    if dur_t <= ub1:
        early_t = int(pick_t - f*ub1)
        late_t  = int(pick_t + f*ub1)
        
    if ub1 < dur_t and dur_t <= ub2:
        early_t = int(pick_t - f*ub2)
        late_t  = int(pick_t + f*ub2)
        
    if ub2 < dur_t and dur_t <= ub3:
        early_t = int(pick_t - f*ub3)
        late_t  = int(pick_t + f*ub3)
        
    else:
        early_t = int(pick_t - f*ub3)
        late_t  = int(pick_t + f*ub3)
    #-------------------------------------------------------------------------
    #To reduce the possibility of demand between 0h-6h and after 20h (30% to accept)
    # if ((st >= 0 and st <= 360) or (st >= 1200)):
    #     print("st_1 = ", st)
    #     getst = rd.randint(1, 10)
    #     print("getst_1 = ", getst)
    #     while (getst > 3 and ((st >= 0 and st <= 360) or (st >= 1200))):
    #         prinst("come_here")
    #         st = rd.randrange(0, st_max)
    #         print("st = ", st)
    #         getst = rd.randint(1, 10)
    #         print("getst = ", getst)
    
    #-------------------------------------------------------------------------
    # added by Sana
    text_file.write(str(i+1)+' '+str(ori_x)+' '+str(ori_y)+' '+str(dest_x)+
                    ' '+str(dest_y)+' '+str(early_t) +' '+str(pick_t) +
                    ' '+str(late_t) +' '+str(dist)+' '+str(dur_t)+'\n')
    #-------------------------------------------------------------------------

text_file.close()