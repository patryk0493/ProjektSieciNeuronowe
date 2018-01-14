import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import time
###

df = pd.read_csv('Skoroszyt1.csv', sep=';')

'''
@relation car
@attribute Buying {vhigh,high,med,low}
@attribute Maint {vhigh,high,med,low}
@attribute Doors {2,3,4,5,more}
@attribute Persons {2,4,more}
@attribute Lug_boot {small,med,big}
@attribute Safety {low,med,high}
@attribute Acceptability {unacc,acc,vgood,good}
@inputs Buying, Maint, Doors, Persons, Lug_boot, Safety
@output Acceptability
'''

dataFrame = pd.DataFrame(df, columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
classFrame = pd.DataFrame(df, columns = ['G'])

t_data = None
v_data = None

#dataFrame.
def fill_missing_data():
	#mode(dataFrame['A'])
	none = 'None'
	dataFrame.fillna(none, inplace=True)
	print('Missing values replaced by ' + none)

def validate_data():
	#dataFrame['A'] = np.where(dataFrame['A'])
	#dataFrame['A'].replace(to_replace=['error'], value=np.NaN, inplace=True)
	dataFrame['A'][dataFrame['A'] != ('vhigh' or 'low')] = 'None'
	pass

def map_data():
	dataFrame['A'] = dataFrame['A'].map({'vhigh': 0, 'high': 1, 'med' : 2, 'low' : 3})
	dataFrame['B'] = dataFrame['B'].map({'vhigh': 0, 'high': 1, 'med' : 2, 'low' : 3})
	dataFrame['C'] = dataFrame['C'].map({'more': 6})
	dataFrame['E'] = dataFrame['D'].map({'more': 6})
	dataFrame['F'] = dataFrame['F'].map({'small': 0, 'med': 1, 'big' : 2})
	dataFrame['G'] = dataFrame['F'].map({'low': 0, 'med': 1, 'big' : 2})
#print(classFrame)

def split_data():
	t_data = dataFrame[:1360]
	v_data = dataFrame[1361:]

fill_missing_data()
validate_data()
map_data()
split_data()
print(t_data)

