import matplotlib.pyplot as plt
import numpy as np

# Define the points for x and N
x = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
N_values = [10, 100, 1000]

# Create the plot
plt.figure(figsize=(10, 6))
y[10] = [0.000137329, ] #2.10x10^-9 , 4.88x10^-19, 2.64x10-
y[100] = [0.0015106] # 2.31x10-8, 5.37x10-18, 2.91x10-37
y[1000] =[0.0152435] #

for N in N_values:
    y = 1000000000000*(N - 1) / (2**x)
    print(y)
    plt.plot(x, y, label=f'N={N}', marker='o')

# Set the x-axis to log scale
#plt.xscale('log')

# Add labels and title
plt.xlabel('x (log scale)')
plt.ylabel('y')
plt.title('Plot of y=(N-1)/(2^x) for different N values')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
