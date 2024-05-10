import numpy as np
from horses3d import *
from plot import *

filename = "/mnt/c/Users/Jon/Desktop/MUMI/TFM/horses3d_Fernando/horses3d/Solver/test/NavierStokes/TaylorGreenGANS/MESH/TGV_HO_0000010000.hmesh"

mesh = Q_from_file(filename).transpose(0, 2, 3, 4, 1)

Q_HO_SRCNN = np.load("RESULTS/Q_HO_SRCNN.npy")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Assuming `mesh_data` is your (512, 9, 9, 9, 3) array
# Reshape to (N, 3) for coordinates and (N,) for solution
coordinates    = mesh[:,:, :, :, :]
solution_values = Q_HO_SRCNN[100, :, :, :, :, 2]

# Flatten coordinates and solution arrays
coordinates_flat = coordinates.reshape(-1, 3)
solution_values_flat = solution_values.reshape(-1)

# Create a scatter plot with solution values as color
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = coordinates_flat[:, 0]
y = coordinates_flat[:, 1]
z = coordinates_flat[:, 2]
c = solution_values_flat

# Plot scatter plot with colors based on solution values
img = ax.scatter(x, y, z, c=c, cmap='jet')

# Add color bar
cbar = plt.colorbar(img)
cbar.set_label('Solution Value')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title('Mesh Data Visualization')
plt.show()
