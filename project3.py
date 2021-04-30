#   #######################################################
#   compuational science project 3.1
#   plotting plantes orbit 
#   Author - Girmay Asrat,  April 2021

#   python version -----------   Python 3.8.5
#   numpy   version-----------   numpy.__version__1.19.1
#   To run the code, make sure all the libraries are imported and the word document
#   in the same folder where the python code is

#   #######################################################

import sys
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt 
import matplotlib.animation as animation
from prettytable import PrettyTable
from docx import Document
import time
import warnings

warnings.filterwarnings("ignore")

start_time = time.time()

print("\nprinting output to a txt file ....")
temp = sys.stdout 
sys.stdout = open("output.txt", "w")


#the gravitational constant G
G = 6.67384e-11

# Mass of the sun
Ms = 1.98855e30

# constant Ks in s^2/m^3
Ks = 4* math.pi**2/(G*Ms)

#   reading the file document
if  len (sys.argv ) != 2 :
    print("usage is : %s input[docx file] " % (sys.argv[0])) 
    sys.exit()

doc_result = Document(sys.argv[1])
table = doc_result.tables[0]
data = [[cell.text for cell in row.cells] for row in table.rows]
df = pd.DataFrame(data)
outside_col, inside_col = df.iloc[0], df.iloc[1]
heir_index = pd.MultiIndex.from_tuples(list(zip(outside_col,inside_col)))
df = pd.DataFrame(data, columns=heir_index).drop(df.index[[0,1]]).reset_index(drop = True)
print("\n\nsome of the data as read from the document is \n\n")
partial_data_frame = df[["Planet", "Mass \n(1024 kg)", "Perihelion","Sidereal orbit period (days)"]].iloc[::2]
print(partial_data_frame)



def euler_Method(Ms, ri,vi,period):
    print("\n\n1.Method Used: Euler\n\n")
    print('{:<10s}{:>12s}{:^24s}{:^24s}{:^12s}{:>12s}'.format("step","dt(days)","V(km/s)","dr(km)","dist2sun(KM)","theta"))
    dt = period *24*3600/1e6
    v =vi
    r = ri
    #initializing arrays
    distances = np.zeros(1000000)
    velocities = np.zeros(1000000)
    angles = np.zeros(1000000)
    x = np.zeros(1000000)
    y = np.zeros(1000000)
    
    for n in range(0,1000000):
        r = r + v*dt
        a = -(G * Ms) /(np.linalg.norm(r)**3) * r
        v = v + a*dt
        if r[0] > 0 and r[1] > 0:
            angle = math.degrees(math.atan(r[1]/r[0]))
        elif (r[0] < 0 and r[1] > 0)or (r[0] < 0 and r[1] < 0) :
            angle = math.degrees(math.atan(r[1]/r[0])) + 180
        else:
            angle = math.degrees(math.atan(r[1]/r[0])) + 360
        x[n] = r[0]
        y[n] = r[1]
        distances[n] = math.sqrt(r[0]**2+r[1]**2)
        velocities[n] = math.sqrt(v[0]**2+v[1]**2)
        angles[n] = angle
        if n==0 or n%5000 == 0:
            print('{:<10d}{:>12.4f}{:^24.4f}{:^24.4f}{:^12.4f}{:>12.4f}'.format(n,dt*n/(24*3600),velocities[n]/1e3,\
               (distances[n]-r[0])/10e3 if n==0 else abs(distances[n]-distances[n-1])/1e3,distances[n]/1e3,angles[n])) 
   
    return np.max(distances), np.min(velocities), np.mean(velocities), np.mean(distances),x,y

def cromer_Method(Ms, ri, vi, period):
    print("\n2.Method Used:Cromer's Method\n\n")
    print('{:<10s}{:>12s}{:^24s}{:^24s}{:^12s}{:>12s}'.format("step","dt(days)","V(km/s)","dr(km)","dist2sun(KM)","theta"))
    dt = period *24*3600/1e6
    v =vi
    r = ri
    distances = np.zeros(1000000)
    velocities = np.zeros(1000000)
    angles = np.zeros(1000000)
    x = np.zeros(1000000)
    y = np.zeros(1000000)
    for n in range(0,1000000):
        a = -(G * Ms) /(np.linalg.norm(r)**3) * r
        v = v + a*dt
        r = r + v*dt
        if r[0] > 0 and r[1] > 0:
            angle = math.degrees(math.atan(r[1]/r[0]))
        elif (r[0] < 0 and r[1] > 0)or (r[0] < 0 and r[1] < 0) :
            angle = math.degrees(math.atan(r[1]/r[0])) + 180
        else:
            angle = math.degrees(math.atan(r[1]/r[0])) + 360
        x[n] = r[0]
        y[n] = r[1]
        distances[n] = math.sqrt(r[0]**2+r[1]**2)
        velocities[n] = math.sqrt(v[0]**2+v[1]**2)
        angles[n] = angle
        if n==0 or n%5000 == 0:
           print('{:<10d}{:>12.4f}{:^24.4f}{:^24.4f}{:^12.4f}{:>12.4f}'.format(n,dt*n/(24*3600),velocities[n]/1e3,\
               (distances[n]-r[0])/10e3 if n==0 else abs(distances[n]-distances[n-1])/1e3,distances[n]/1e3,angles[n])) 
    
    return np.max(distances), np.min(velocities), np.mean(velocities), np.mean(distances), x , y


def midPoint(Ms, ri , vi , period):
    print("\n3.Method Used:Mid Point\n\n")
    print('{:<10s}{:>12s}{:^24s}{:^24s}{:^12s}{:>12s}'.format("step","dt(days)","V(km/s)","dr(km)","dist2sun(KM)","theta"))
    dt = period *24*3600/1e6
    v =vi
    r = ri
    distances = np.zeros(1000000)
    velocities = np.zeros(1000000)
    angles = np.zeros(1000000)
    x = np.zeros(1000000)
    y = np.zeros(1000000)
    for n in range(0,1000000):
        a = -(G * Ms) /(np.linalg.norm(r)**3) * r
        r_half = r + v*(dt/2)
        a_half = -(G * Ms) /(np.linalg.norm(r_half)**3) * r_half
        v_half = v + a*(dt/2)

        r = r + v_half*dt
        v = v + a_half*dt

        if r[0] > 0 and r[1] > 0:
            angle = math.degrees(math.atan(r[1]/r[0]))
        elif (r[0] < 0 and r[1] > 0)or (r[0] < 0 and r[1] < 0) :
            angle = math.degrees(math.atan(r[1]/r[0])) + 180
        else:
            angle = math.degrees(math.atan(r[1]/r[0])) + 360
        x[n] = r[0]
        y[n] = r[1]
        distances[n] = math.sqrt(r[0]**2+r[1]**2)
        velocities[n] = math.sqrt(v[0]**2+v[1]**2)
        angles[n] = angle
        if n==0 or n%5000 == 0:
            print('{:<10d}{:>12.4f}{:^24.4f}{:^24.4f}{:^12.4f}{:>12.4f}'.format(n,dt*n/(24*3600),velocities[n]/1e3,\
               (distances[n]-r[0])/10e3 if n==0 else abs(distances[n]-distances[n-1])/1e3,distances[n]/1e3,angles[n])) 
    return np.max(distances), np.min(velocities), np.mean(velocities), np.mean(distances),x,y

def graphing(s):
    plt.title("Runge Kutta for {} planets".format(s))
    plt.show()         

# initializing arrays to store the results
max_distance =np.zeros(len(partial_data_frame))
min_speed = np.zeros(len(partial_data_frame))
semi_major_axis = np.zeros(len(partial_data_frame))
T= np.zeros(len(partial_data_frame))
e = np.zeros(len(partial_data_frame))
mean_v = np.zeros(len(partial_data_frame))
colors= ["green","pink", "blue","red","black","brown","purple","cyan"]
fig, ax = plt.subplots(2)
for i in range(0,len(partial_data_frame)):
    def output(dist, v, v_mean, r_mean):
        max_distance[i] = dist/1e9
        min_speed[i] = v/1e3
        semi_major_axis[i] = (dist + distance_at_perihelion[0])/(2* 1e9)
        T[i] = math.sqrt(Ks*r_mean**3)/(3600 * 24)
        e[i] = (semi_major_axis[i]- distance_at_perihelion[0]/(1e9))/semi_major_axis[i] 
        mean_v[i] = v_mean/1e3
        print("\nplanet =     {}\nspeed at the aphelion = {}\ndistance at the aphelion ={}\
            \nsemi major axis = {}\nperiod = {}\neccentricty= {}\nmean_v=   {}".format(planet, \
                min_speed[i],max_distance[i],semi_major_axis[i],T[i],e[i],mean_v[i]))
        


        
    
    planet= partial_data_frame.values.tolist()[i][0]
    distance_at_perihelion = np.array([float(partial_data_frame.values.tolist()[i][2].replace(',', ''))*1e9,0])
    speed_at_perihelion = np.array([0,float(partial_data_frame.values.tolist()[i][3].replace(',', ''))*1e3])
    period = float(partial_data_frame.values.tolist()[i][4].replace(',', ''))
    print("\n\n*****************processing Planet {} ***************************\n\n".format(planet))

    dist, v, v_mean, r_mean,x,y= euler_Method(Ms,distance_at_perihelion, speed_at_perihelion,period)
    output(dist, v, v_mean, r_mean)
    
    dist, v, v_mean, r_mean, x,y = cromer_Method(Ms,distance_at_perihelion, speed_at_perihelion,period)
    output(dist, v, v_mean, r_mean)

    dist, v, v_mean, r_mean,x, y = midPoint(Ms,distance_at_perihelion, speed_at_perihelion,period)
    output(dist, v, v_mean, r_mean)
    if i< 4:
    
        ax[0].scatter(x, y, color=colors[i], label = planet)
        ax[0].scatter(0,0, color = 'orange',linewidths = 7)
        ax[0].legend(loc = "upper right")
    else:
        ax[1].scatter(x, y, color=colors[i], label = planet)
        ax[1].scatter(0,0, color = 'orange',linewidths = 5)
        ax[1].legend(loc = "upper right")
        
graphing(" inner and outer")    

sys.stdout.close()
sys.stdout = temp
print("output file write completed!")
print("\n--- %s seconds ---" % (time.time() - start_time))

