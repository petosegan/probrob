import ogmap
import numpy as np
import matplotlib.pyplot as plt
from mapdef import mapdef, NTHETA

this_map = mapdef()

this_sonar = ogmap.Sonar(NUM_THETA = NTHETA, GAUSS_VAR = 1)

x_traj = np.linspace(0, 80, 20)
y_traj = [60]*len(x_traj)

scans = []

for pose in zip(x_traj, y_traj):
    scans.append(this_sonar.simulate_scan(pose, this_map))
    
xvar = 0.01
yvar = 0.01  

plt.ion()
fig = plt.figure()
xs = []
ys = []
for scan in scans:
    scan = this_sonar.maxmin_filter(scan)
    # x0 = scan.x0 + np.random.normal(0, np.sqrt(xvar))
    # y0 = scan.y0 + np.random.normal(0, np.sqrt(yvar))
    x0 = scan.x0
    y0 = scan.y0
    xs.extend(x0+scan.rs*np.cos(scan.thetas))
    ys.extend(y0+scan.rs*np.sin(scan.thetas))

nbins = this_map.N
H, xedges, yedges = np.histogram2d(xs,ys,bins=nbins)

# H needs to be rotated and flipped
H = np.rot90(H)
H = np.flipud(H)
 
Hmax = np.max(np.max(H))
 
# Mask zeros
Hmasked = np.ma.masked_where(H<=.05*Hmax,H) # Mask pixels with low values

 
# Plot 2D histogram using pcolor
plt.pcolormesh(xedges,yedges,Hmasked)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, this_map.N)
plt.ylim(0, this_map.N)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')
