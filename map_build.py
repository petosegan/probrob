import ogmap
import numpy as np
import matplotlib.pyplot as plt

this_map = ogmap.OGMap(100)
this_map.rect(95, 0, 10, 50)
this_map.rect(5, 0, 5, 50)
this_map.rect(10, 70, 50, 10)

this_sonar = ogmap.Sonar()

x_traj = np.linspace(0, 80, 20)
y_traj = [60]*len(x_traj)

scans = []

for (xpos, ypos) in zip(x_traj, y_traj):
    scans.append(this_sonar.simulate_scan(xpos, ypos, this_map))
    
xvar = 3
yvar = 3    

plt.ion()
fig = plt.figure()
xs = []
ys = []
for scan in scans:
    scan = this_sonar.maxmin_filter(scan)
    x0 = scan.x0 + np.random.normal(0, np.sqrt(xvar))
    y0 = scan.y0 + np.random.normal(0, np.sqrt(yvar))
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
# fig2 = plt.figure()
plt.pcolormesh(xedges,yedges,Hmasked)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, this_map.N)
plt.ylim(0, this_map.N)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')
