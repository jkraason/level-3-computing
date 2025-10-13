import numpy as np
import matplotlib.pyplot as plt
def lattice_positions(config,size):
    x = np.array([0])
    y = np.array([0])
    if config == "sc":
        count = 0
        while count < size:
            x = np.append(x,x+1)
            y = np.append(y,y)
            y = np.append(y,y+1)
            x = np.append(x,x)
            count = count+1
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
    print(x,y)
    return(x,y)
    
config = input("Enter your lattice configuration: ")
size = int(input("Enter the number (squared) of unit cells: "))
lattice_positions(config,size)
plt.show()
    




 