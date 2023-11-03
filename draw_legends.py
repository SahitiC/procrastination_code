"""
script to make legend independant of any plot
"""

import matplotlib.pyplot as plt

#plt.rcParams['text.usetex'] = True
colors = ["gold",
          "tab:blue",
          "brown"#mpl.colors.to_rgba('tab:blue', alpha=0.5),
          ]
plt.figure(figsize=(0.5,0.5), dpi=300)
# cmap= mpl.colormaps.get_cmap('viridis')
# colors = [cmap(1.0), cmap(0.5), cmap(0.0)]
f = lambda m,c: plt.plot([],[],marker=m, markersize=15, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(3)]
labels = ["check", "work", "shirk"]
legend = plt.legend(handles, labels, loc=3, 
                    framealpha=1, frameon=False,
                    title='actions', title_fontsize=18)
fig  = legend.figure
fig.canvas.draw()
plt.axis('off')
plt.show()