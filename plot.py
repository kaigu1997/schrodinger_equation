import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
matplotlib.use('TkAgg')

# read input grid and time
def read_input():
    xfile = open('x.txt', 'r')
    x = np.array(xfile.readlines(), dtype=float)
    xfile.close()
    tfile = open('t.txt', 'r')
    t = np.array(tfile.readlines(), dtype=int)
    tfile.close()
    return x, t

# plot preparation
x, t = read_input()
fig = plt.figure()
ax = fig.add_subplot(111, xlim=(x[0], x[len(x)-1]), ylim=(0, 1))
ax.grid()
plt.xlabel('x')
plt.ylabel('Population')

# set color
lines = []
plotcolor = ['red', 'blue']
for i in range(2):
    lines.append(ax.plot([], [], lw=2, color=plotcolor[i])[0])

# set time text
# warning: position is the place on the screen, not the x-y coordinates shown
time_template = 'time = %ds'
time_text = ax.text(0.02, 0.93, '', fontdict={'size': 16}, transform=ax.transAxes)

# initialization: blank page
def init():
    for line in lines:
        line.set_data([], [])
    time_text.set_text('')
    return lines, time_text

# animation: each line in psi.txt
def ani(i):
    psifile = open('psi.txt', 'r')
    for j in range(i - 1):
        psifile.readline()
    psi = np.array(psifile.readline().split(), dtype=float)
    psifile.close()
    xlist = [x, x]
    ylist = [psi[0:len(x)], psi[len(x):len(psi)]]
    for lnum, line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum])
    time_text.set_text(time_template % t[i])
    return lines, time_text

# make the animation
ani = animation.FuncAnimation(fig, ani, len(t), init, interval=1000//(t[1]-t[0]), repeat=False, blit=False)
# show
ani.save('psi.gif','imagemagick')
# plt.show()
