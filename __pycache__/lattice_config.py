import numpy as np
import matplotlib.pyplot as plt
def lattice_positions(config,size):
    x = np.array([0])
    y = np.array([0])
    x_temp = np.array([])
    y_temp = np.array([])
    if config == "sc":
        for i in range(0,(size**2)):
            x_temp = np.array([])
            y_temp = np.array([])
            x_temp = np.append(x_temp,x[i]) #0
            y_temp = np.append(y_temp,y[i]) #0
            y_temp = np.append(y_temp,y_temp[i]+1) #0,1
            x_temp = np.append(x_temp,x_temp[i]+1) #0,1
            x_temp = np.append(x_temp,x[i]+1)#0,1,1
            y_temp = np.append(y_temp, y_temp[i])#0,1,0
            x_temp = np.append(x_temp, x_temp[i])#0,1,1,0
            y_temp = np.append(y_temp, y_temp[i]+1)#0,1,0,1
            print(x_temp)
            print(y_temp)
            x = np.append(x,x_temp)
            y = np.append(y,y_temp)
            i=i+(size**2)
            print(x,y)
    elif config == "bcc":
        count = 0
        while count < size:
            x = np.append(x,x+1)
            y = np.append(y,y)
            y = np.append(y,y+1)
            x = np.append(x,x)
            x = np.append(x,(0.5+count))
            y = np.append(y,(0.5+count))
            count = count+1
    elif config == "fcc":
        count = 0
        while count < size:
            x = np.append(x,x+0.5)
            y = np.append(y,y)
            y = np.append(y,y+0.5)
            x = np.append(x,x)
            x = np.append(x,(0.5+count))
            y = np.append(y,(0.5+count))

            count = count+0.5

    else:
        print("error, not a recognised lattice structure")

    plt.scatter(x,y)
    print(len(x),len(y))
    return(x,y)
    
config = input("Enter your lattice configuration: ")
size = int(input("Enter the number (squared) of unit cells: "))
lattice_positions(config,size)
plt.show()
    




 