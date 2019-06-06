#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns;  sns.set()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Reading the Input File
df = pd.read_table('Elephant_Killings.txt',\
		delim_whitespace=True , skiprows= 6 ,\
		names=('Year','CentralAfrica','EasternAfrica','WestAfrica','SouthernAfrica'))

Head 	   				='Number of Elephants Killed'
Head_Total 				='Total of Elephants Killed'
Years 					= np.array(df['Year'])
CentralAfrica 			= np.array(df['CentralAfrica'])
EasternAfrica 			= np.array(df['EasternAfrica'])
WestAfrica 				= np.array(df['WestAfrica'])
SouthernAfrica 			= np.array(df['SouthernAfrica'])

TOTAL = []

# Create Totals per Year
for j in range(0,len(Years)):
	TOTAL.append(CentralAfrica[j] + EasternAfrica[j] + WestAfrica[j]+ SouthernAfrica[j])

MAX = np.max([ np.max(CentralAfrica), np.max(EasternAfrica),np.max(WestAfrica),np.max(SouthernAfrica) ])
MIN = np.min([ np.min(CentralAfrica), np.min(EasternAfrica),np.min(WestAfrica),np.min(SouthernAfrica) ])

MAX_Total = np.max(TOTAL)
MIN_Total  = np.min(TOTAL)

CA = pd.DataFrame(CentralAfrica,Years)
CA.columns = {Head}

EA = pd.DataFrame(EasternAfrica,Years)
EA.columns = {Head}

WA = pd.DataFrame(WestAfrica,Years)
WA.columns = {Head}

SA = pd.DataFrame(SouthernAfrica,Years)
SA.columns = {Head}

TOT = pd.DataFrame(TOTAL,Years)
TOT.columns = {Head_Total}

Total_Killed = np.sum(TOTAL)


#fig, axes = plt.subplots(2,1, figsize=(20, 14), sharex=True)
fig = plt.figure(figsize=(20, 14))
ax3 = plt.subplot2grid((3, 1), (0, 0))
ax1 = plt.subplot2grid((3, 1), (1, 0))
ax2 = plt.subplot2grid((3, 1), (2, 0))


ax1.set_title("Total Number of Elephants Killed in Africa per Year : [2003 - 2017] ",fontweight='bold', fontsize=15)
ax2.set_title("Elephant Killings in Africa by Zone - Year : [2003 - 2017] ",fontweight='bold', fontsize=15)
ax3.set_title("Total Number of Elephants Killed in Africa per Year : [2003 - 2017] ",fontweight='bold', fontsize=15)
ax1.set_ylabel("Total # of Killings ",fontweight='bold', fontsize=15)
ax2.set_ylabel("# of Elephant Killings per Zone ",fontweight='bold', fontsize=15)
ax3.set_ylabel("Total # of Killings ",fontweight='bold', fontsize=15)
ax2.set_xlabel("Year ",fontweight='bold', fontsize=15)


Disp_template = 'Year = %d'
TEXT = plt.text(Years[0], MAX-500,'', fontsize=25,fontweight='bold')
Disp_template2 = 'Total Killed over 15 years = %d'
TEXT2 = plt.text(Years[0]-1, MAX-1200,'', fontsize=20,fontweight='bold',color='maroon')

def animate(i):

	CA_data 		= CA.iloc[:int(i+1)]
	EA_data 		= EA.iloc[:int(i+1)] 
	WA_data 		= WA.iloc[:int(i+1)] 
	SA_data 		= SA.iloc[:int(i+1)] 
	TOT_data 		= TOT.iloc[:int(i+1)] 
    
	if (i == 1):

		p = sns.lineplot(x=CA_data.index, y=CA_data[Head], data=CA_data, color="r", marker="D", label='Central Africa', ax=ax2)
		q = sns.lineplot(x=EA_data.index, y=EA_data[Head], data=EA_data, color="b", marker="D", label='Eastern Africa', ax=ax2)
		r = sns.lineplot(x=WA_data.index, y=WA_data[Head], data=WA_data, color="g", marker="D", label='Western Africa', ax=ax2)
		s = sns.lineplot(x=SA_data.index, y=SA_data[Head], data=SA_data, color="y", marker="D", label='Southern Africa', ax=ax2)
		t = sns.lineplot(x=TOT_data.index, y=TOT_data[Head_Total], data=TOT_data, color="maroon", marker="D", label='Total Killings', ax=ax1)
		t1 = sns.barplot(x=TOT_data.index, y=TOT_data[Head_Total], data=TOT_data, label='Total Killings', ax=ax3)
    
	else:
		
		p = sns.lineplot(x=CA_data.index, y=CA_data[Head], data=CA_data, color="r", marker="D", ax=ax2)
		q = sns.lineplot(x=EA_data.index, y=EA_data[Head], data=EA_data, color="b", marker="D", ax=ax2)
		r = sns.lineplot(x=WA_data.index, y=WA_data[Head], data=WA_data, color="g", marker="D", ax=ax2)
		s = sns.lineplot(x=SA_data.index, y=SA_data[Head], data=SA_data, color="y", marker="D", ax=ax2)
		t = sns.lineplot(x=TOT_data.index, y=TOT_data[Head_Total], data=TOT_data, color="maroon", marker="D", ax=ax1)
		t1 = sns.barplot(x=TOT_data.index, y=TOT_data[Head_Total], data=TOT_data, ax=ax3)
    
		
	TEXT.set_text(Disp_template % (Years[i]))
	
	if (i>=13):
		
		TEXT2.set_text(Disp_template2 % (Total_Killed))
	
	p.tick_params(direction='in', length=6, width=2,labelsize=20)
	t.tick_params(direction='in', length=6, width=2,labelsize=20)
	t1.tick_params(direction='in', length=6, width=2,labelsize=20)
	plt.setp(p.lines,linewidth=3)
	plt.setp(t.lines,linewidth=3)
	plt.tight_layout()
	sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=15, repeat=False, interval=2000)

plt.xlim((Years[0]-1), (Years[-1]+1))
plt.ylim(MIN-500,MAX+500)

ax1.set(ylim=(MIN_Total-500, MAX_Total+500))
ax1.set(xlim=((Years[0]-1), (Years[-1]+1)))

ax3.set(ylim=(MIN_Total-500, MAX_Total+500))
ax3.set(xlim=((Years[0]-1), (Years[-1]+1)))

ani.save('Elephant-Sad-Story.mp4', writer='ffmpeg')
plt.show()
