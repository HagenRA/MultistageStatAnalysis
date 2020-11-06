# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:39:18 2020

@author: Administrator
"""


#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

#%% Importing set

for i in tqdm(range(100), desc = "Import set"):
	allsites = pd.read_excel('FILE PATH TO SOURCE FILE HERE',
				   sheet_name = 'Sampled Data v2')
	allsites = allsites.drop(['Jar #'], axis =1)
	allsites = allsites.replace(0, np.nan)
	check = allsites['Cavity Depth']< 32
	allsites.loc[check, 'Cavity Depth'] = np.nan

# Categorise by jar site
j1_df = allsites[allsites['Jar Site']==1]
j1_df = j1_df.drop(['Jar Site', 'Rim Thickness'], axis =1)
j2_df = allsites[allsites['Jar Site']==2]
j2_df = j2_df.drop(['Jar Site'], axis =1)
j3_df = allsites[allsites['Jar Site']==3]
j3_df = j3_df.drop(['Jar Site'], axis =1)

#%% Descriptive stats that.... don't say much

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

sites  = [allsites, j1_df, j2_df, j3_df]
for i in sites:
	print(get_df_name(i) + ' has the following number of null values:')
	print(i.info())
	print(i.isnull().sum())
	print('\n')

#%% Jar Site Paired
	
for i in tqdm(range(100), desc = "Site 1"):
	plt_j1_df = sns.pairplot(j1_df)
	plt_j1_df.fig.suptitle("Jar Site 1")
	plt.show()

for i in tqdm(range(100), desc = "Site 2"):
	plt_j2_df = sns.pairplot(j2_df)
	plt_j2_df.fig.suptitle("Jar Site 2")
	plt.show()

for i in tqdm(range(100), desc = "Site 3"):
	plt_j3_df = sns.pairplot(j3_df)
	plt_j3_df.fig.suptitle("Jar Site 3")
	plt.show()

#%% All Sites
for i in tqdm(range(100), desc = "All Sites"):
	sns.set(style='ticks', color_codes=True)
	plt_df = sns.pairplot(allsites, hue='Jar Site', vars =['Jar Height',
												 'Middle Body Circumference',
				  'Base Circumference', 'Cavity Diameter', 'Rim Diameter',
				  'Rim Height', 'Cavity Depth'])
	plt_df.fig.suptitle("All Sites")
	plt.show()


#%% Plotting individually across all sites
x_vars = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Cavity Depth']
y_vars = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Cavity Depth']


sns.set_palette(sns.color_palette("muted"))
plt_df = sns.pairplot(allsites, hue='Jar Site', vars =['Jar Height',
												 'Middle Body Circumference',
				  'Base Circumference', 'Cavity Diameter', 'Rim Diameter',
				  'Rim Height', 'Cavity Depth'])
plt_df.fig.suptitle("All Sites")
plt.savefig(f'C:/Users/Administrator/Downloads/All Sites/All paired.png')


for x, y in [(x,y) for x in x_vars for y in y_vars if x != y]:
	df_plt = sns.pairplot(allsites, height = 3, hue='Jar Site', vars =[x, y])
	df_plt.fig.suptitle(f'{x} x {y}')
	plt.savefig(f'C:/Users/Administrator/Downloads/All Sites/{x} x {y}.png')
#	plt.show()

x_vars = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Cavity Depth']
y_vars = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Cavity Depth']
# Site 1
colors = ["windows blue"]
sns.set_palette(sns.xkcd_palette(colors))
plt_j1_df = sns.pairplot(j1_df)
plt_j1_df.fig.suptitle("Jar Site 1")
plt.savefig(f'C:/Users/Administrator/Downloads/J1/J1 paired.png')

for x, y in [(x,y) for x in x_vars for y in y_vars if x != y]:
	df_plt = sns.pairplot(j1_df, height = 3, vars =[x, y])
	df_plt.fig.suptitle(f'J1 {x} x {y}')
	plt.savefig(f'C:/Users/Administrator/Downloads/J1/J1 - {x} x {y}.png')

x_vars = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Rim Thickness'
		  , 'Cavity Depth']
y_vars = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Rim Thickness'
		  , 'Cavity Depth']

# Site 2
colors = ["orange"]
sns.set_palette(sns.xkcd_palette(colors))
plt_j2_df = sns.pairplot(j2_df)
plt_j2_df.fig.suptitle("Jar Site 2")
plt.savefig(f'C:/Users/Administrator/Downloads/J2/J2 paired.png')


for x, y in [(x,y) for x in x_vars for y in y_vars if x != y]:
	df_plt = sns.pairplot(j2_df, height = 3, vars =[x, y])
	df_plt.fig.suptitle(f'J2 {x} x {y}')
	plt.savefig(f'C:/Users/Administrator/Downloads/J2/J2 - {x} x {y}.png')
	
# Site 3
colors = ["green"]
sns.set_palette(sns.xkcd_palette(colors))
plt_j3_df = sns.pairplot(j3_df)
plt_j3_df.fig.suptitle("Jar Site 3")
plt.savefig(f'C:/Users/Administrator/Downloads/J3/J3 paired.png')

for x, y in [(x,y) for x in x_vars for y in y_vars if x != y]:
	df_plt = sns.pairplot(j3_df, height = 3, vars =[x, y])
	df_plt.fig.suptitle(f'J3 {x} x {y}')
	plt.savefig(f'C:/Users/Administrator/Downloads/J3/J3 - {x} x {y}.png')

#%%	Correlation matrices 
method = 'pearson'
df_corr = allsites.drop(['Jar Site'], axis =1)
corr_df = df_corr.corr(method = method)
corr_j1_df = j1_df.corr(method = method)
corr_j2_df = j2_df.corr(method = method)
corr_j3_df = j3_df.corr(method = method)

corr_list = {'All Sites': corr_df, 'J1': corr_j1_df,
			 'J2': corr_j2_df, 'J3': corr_j3_df}

for i in tqdm(range(100), desc = "Calculating Correlations"):
	for key, frame in corr_list.items():
		writer = pd.ExcelWriter(f'C:/Users/Administrator/Downloads/Correlations.xlsx',
						  mode = 'a', engine='openpyxl')
		frame.to_excel(writer,sheet_name = f'{key}', index = True)
		writer.save()
	
#%% Scipy T-Test all

column_A = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Rim Thickness'
		  , 'Cavity Depth']
column_B = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Rim Thickness'
		  , 'Cavity Depth']

all_t_test_results = {}

for i in tqdm(range(100), desc = "SciPy Test"):
	for x, y in [(x,y) for x in column_A for y in column_B if x != y]:
		all_t_test_results.update({str(f'{x} + {y}'):
							 (scipy.stats.ttest_ind(allsites[x],allsites[y], equal_var = False,
							   nan_policy = 'omit'))})
	all_ttest_output = pd.DataFrame.from_dict(all_t_test_results, orient='index',
										   columns =['T-Statistic', 'p-value'])
	writer = pd.ExcelWriter(f'C:/Users/Administrator/Downloads/T-test.xlsx',
						mode = 'a', engine='openpyxl')
	all_ttest_output.to_excel(writer,sheet_name = 'All sites',
						   float_format = '%.10f', index = True)
	writer.save()

#%% TTest for singular sites

column_A = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Cavity Depth']
column_B = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Cavity Depth']

j1_t_test_results = {}
for x, y in [(x,y) for x in column_A for y in column_B if x != y]:
	j1_t_test_results.update({str(f'{x} + {y}'):
		(scipy.stats.ttest_ind(j1_df[x],j1_df[y], equal_var = False,
						 nan_policy = 'omit'))})

j1_ttest_output = pd.DataFrame.from_dict(j1_t_test_results, orient='index',
										 columns =['T-Statistic', 'p-value'])

writer = pd.ExcelWriter(f'C:/Users/Administrator/Downloads/T-test.xlsx',
						mode = 'a', engine='openpyxl')
j1_ttest_output.to_excel(writer,sheet_name = 'Site 1', float_format = '%.10f',
						 index = True)
writer.save()

column_A = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Rim Thickness'
		  , 'Cavity Depth']
column_B = ['Jar Height', 'Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Rim Thickness'
		  , 'Cavity Depth']

j2_t_test_results = {}
for x, y in [(x,y) for x in column_A for y in column_B if x != y]:
	j2_t_test_results.update({str(f'{x} + {y}'):
		(scipy.stats.ttest_ind(j2_df[x],j2_df[y], equal_var = False,
						 nan_policy = 'omit'))})

j2_ttest_output = pd.DataFrame.from_dict(j2_t_test_results, orient='index',
										 columns =['T-Statistic', 'p-value'])

writer = pd.ExcelWriter(f'C:/Users/Administrator/Downloads/T-test.xlsx',
						mode = 'a', engine='openpyxl')
j2_ttest_output.to_excel(writer,sheet_name = 'Site 2', float_format = '%.10f',
						 index = True)
writer.save()

j3_t_test_results = {}
for x, y in [(x,y) for x in column_A for y in column_B if x != y]:
	j3_t_test_results.update({str(f'{x} + {y}'):
		(scipy.stats.ttest_ind(j3_df[x],j3_df[y], equal_var = False,
						 nan_policy = 'omit'))})

j3_ttest_output = pd.DataFrame.from_dict(j3_t_test_results, orient='index',
										 columns =['T-Statistic', 'p-value'])

writer = pd.ExcelWriter(f'C:/Users/Administrator/Downloads/T-test.xlsx',
						mode = 'a', engine='openpyxl')
j3_ttest_output.to_excel(writer,sheet_name = 'Site 3', float_format = '%.10f',
						 index = True)
writer.save()

#%% Modelling the data
# Personal note: Linear regression isn't exactly the best way to do the
# prediction, instead go for clustering
# Trying using Jar Height x Middle Body Circumference

unknown_df = allsites[allsites['Jar Height'].isnull()].copy()
train_df = allsites[allsites['Jar Height'].notnull()].copy()

#%% Determine which metrics have the most complete data

ml_df = [train_df, unknown_df]

for j in ml_df:
	print(get_df_name(j) + ' has the following number of null values:')
	print()
	print(j.isnull().sum())
	print('\n')
	
# Jar height x Cavity Depth seem most safe to predict jar height in unknown_df

#%% Fixing up the training dataset by filling the empty cells with
# 	the mean of the column it's a part of.
#	Similarly, filling the unkown df with the column that least empty,
#	which is 'Cavity Depth'
	
train_fill = ['Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Rim Thickness',
		  'Cavity Depth']
for i in train_fill:
	train_df[i] = train_df[i].fillna((train_df[i].mean()))

for i in train_fill:
	unknown_df[i] = unknown_df[i].fillna((unknown_df[i].mean()))
	
#%% Defining the independent (x) variables to the dependent variables (y)

train_X = (train_df[['Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Rim Thickness',
		  'Cavity Depth']]).astype(int)
train_y = (train_df['Jar Height']).astype(int)

unknown_X = (unknown_df[['Middle Body Circumference', 'Base Circumference',
		  'Cavity Diameter', 'Rim Diameter', 'Rim Height', 'Rim Thickness',
		  'Cavity Depth']]).astype(int)

#%% Trying to fit the data

regr = LinearRegression()
regr = regr.fit(train_X, train_y)

#%% Prediction

unknown_X['prediction'] = regr.predict(unknown_X)
print(unknown_X)