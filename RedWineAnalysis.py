# -*- coding: utf-8 -*-
"""
Created on Tue May 22 05:28:25 2018

@author: IvanA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab

wine_data = pd.read_csv("winequality-red.csv", header=0)
wine_data_original = wine_data

print(wine_data.head())

#######################
##  File statistics  ##
#######################
print('\n -- Sección File Statistics --')
# Obtain number of columns and number of records
print(wine_data.info())

# Obtain for each column the min, max, mean and std
print(wine_data.agg(['min', 'max', 'mean', 'std']))


# Check if there is any column with zeros
print("\n")
print("Comprueba si hay columnas con ceros")
for i in range(0,len(wine_data.columns)):
    
    df = wine_data[wine_data.iloc[:,i] == 0]
    if(len(df.index) > 0):
        print("La columna {} tiene zeros" .format(df.columns[i]))
    else:
        print("La columna {} NO tiene zeros" .format(df.columns[i]))
        
# Check if there is any column with Nulls

print("\n")
print ("Comprueba si hay columnas con null")

for i in range(0,len(wine_data.columns)):
    if (wine_data.iloc[:,1].isnull().any()):
        print("La columna {} tiene vacios" .format(wine_data.columns[i]))
    else:
        print("La columna {} NO tiene vacios" .format(wine_data.columns[i]))

################# 
## Correlation ##
#################
print('\n-- Sección Correlation --')
print('Correlación de todas las variables del dataset')
corr_matrix = wine_data.corr()

print(corr_matrix)

corr_matrix.to_csv("wine column correlation.csv", sep='\t')
        

##################################
##  Extreme score verification  ##
##################################
print('\n-- Sección Extreme score verification --')
    
def find_outliers (df):
    """Returns a sorted list of extreme values """
    
    # Define boxplot in order to obtain whiskers
    _ , bp = pd.DataFrame.boxplot(df, return_type= 'both', whis =3)
    
    # Get the whiskers
    whiskers = [whiskers.get_ydata() for whiskers in bp["whiskers"]]
    
    
    min_whisker = whiskers[0][1]
    max_whisker = whiskers[1][1]
    
    # Using the obtained whiskers prepare final outliers list
    outliers = df[df>max_whisker]
    outliers.append(df[df<min_whisker])
    
    outliers_list = outliers.values.T.tolist()
    
    outliers_list.sort()
    
    return outliers_list

print('Muestra por cada variable sus gráficas de histograma y boxplot')
# Creation of Histogram and boxplot graphs
for i in range(0,len(wine_data.columns)):
    wine_data.hist(column=wine_data.columns[i])
    plt.title(wine_data.columns[i])
    plt.suptitle("")
    plt.show()
    wine_data.boxplot(column=wine_data.columns[i], whis = 3) 
    plt.title(wine_data.columns[i])
    plt.suptitle("")
    plt.show()  

plt.close()   
# Calculation of outliers   
print('Calcula los valores extremos de cada columna')
for i in range(0,len(wine_data.columns)):
    print("Columna: {}" .format(wine_data.columns[i]))
    outliers_list = find_outliers(wine_data.iloc[:,i])
    print("Valores extremos: {}" 
          .format(outliers_list))  
    print("Porcentaje de valores extremos: {0:.2f} \n" 
          .format(len(outliers_list)*100/len(wine_data)))

##############################
##  Extreme score handling  ##
##############################
print('\n-- Sección Extreme score handling --')   

print('Crea un nuevo dataset, con los valores extremos eliminados')
print('El nuevo dataset es winequality-red-analysis.csv')
# Removal of extreme outliers for Residual Sugar & Chlorides columns
# Truncate all records whose Residual Sugar is >= 4.8
# Truncate all records whose Chlorides is >= 0.152

temp_df = wine_data[wine_data["residual sugar"]<4.8 ]

df = temp_df[temp_df["chlorides"]<0.152 ] 

df.to_csv("winequality-red-analysis.csv", sep='\t')



###################################
##  Data Selection for Analysis  ##
################################### 
print('\n-- Sección Data Selection for Analysis --')
# Creation of two dataframes for analysis
# The first df will contain records having quality > 6
# The second df will contain records having quality <=6

df_high_quality = df[df["quality"]>6]
df_normal_quality = df[df["quality"]<=6]

#######################
##  Normality Test   ##
#######################
print('\n-- Sección Normality Test --')
# The Normality Test will be done using the Anderson-Darling

from scipy import stats


def AD_test(df):
    """Performs the Anderson-Darling test for the columns of the dataframe
    besides the last column which is assumed to be the class"""
    
    matrix_ad = [['Variable', 'DF', 'Test Statistic', 'p-value']]
    
    
    
    for i in range(0,len(df_high_quality.columns)):
        anderson_results = stats.anderson(df[df.columns[i]])
        
        # Important to note that taking anderson_results[1][2] is due to
        # having a significant level of 0.05 (3rd value in the array)
        matrix_ad.append( 
        [df.columns[i], len(df.iloc[:,i]) - 1,
         anderson_results[0], anderson_results[1][2]])
    
    
    print(" {0:33} {1:5} {2:20} {3:7} " .format(matrix_ad[0][0],matrix_ad[0][1],
              matrix_ad[0][2],matrix_ad[0][3]) )
    
    for i in range(1, len(matrix_ad)-1):   
        print(" {0:20} {1:15d} {2:15.3f} {3:15.3f} " .format(matrix_ad[i][0],matrix_ad[i][1],
              matrix_ad[i][2],matrix_ad[i][3]) )
        
    print("\n")


print('Resultado del test de Anderson-Darling')
AD_test(df_high_quality)
AD_test(df_normal_quality)



######################
##  Variance Test   ##
######################
print('\n-- Sección Variance Test --')
# The Variance Homogeneity Test will be done using the Fligner-Killeen test

matrix_fligner = [['Dataset', 'Statistic', 'p-value']]

statistic,p_value = stats.fligner(df_high_quality.iloc[:,0],
                                df_high_quality.iloc[:,1],
                                df_high_quality.iloc[:,2],
                                df_high_quality.iloc[:,3],
                                df_high_quality.iloc[:,4],
                                df_high_quality.iloc[:,5],
                                df_high_quality.iloc[:,6],
                                df_high_quality.iloc[:,7],
                                df_high_quality.iloc[:,8],
                                df_high_quality.iloc[:,9],
                                df_high_quality.iloc[:,10],
                                df_high_quality.iloc[:,11])



matrix_fligner.append(['df_high_quality', statistic, p_value])


statistic, p_value = stats.fligner(df_normal_quality.iloc[:,0],
                                df_normal_quality.iloc[:,1],
                                df_normal_quality.iloc[:,2],
                                df_normal_quality.iloc[:,3],
                                df_normal_quality.iloc[:,4],
                                df_normal_quality.iloc[:,5],
                                df_normal_quality.iloc[:,6],
                                df_normal_quality.iloc[:,7],
                                df_normal_quality.iloc[:,8],
                                df_normal_quality.iloc[:,9],
                                df_normal_quality.iloc[:,10],
                                df_normal_quality.iloc[:,11])



matrix_fligner.append(['df_normal_quality', statistic, p_value])

print('Resultado del test de Fligner-Killeen')
print(" {0:25} {1:20} {2:20}" .format(matrix_fligner[0][0],matrix_fligner[0][1],
      matrix_fligner[0][2]))

for i in range(1, len(matrix_fligner)):   
        print(" {0:20} {1:15f} {2:15.3f} " .format(matrix_fligner[i][0],
              matrix_fligner[i][1],
              matrix_fligner[i][2] ))

#####################
##  Data analysis  ##  
#####################
print('\n-- Sección Data analysis --')      
import seaborn as sns


wine_data = df.copy()
# Compare each feature to the class visually   

print('\nGráficas comparativas de las variables con el valor de la clase')

for i in range(0, len(wine_data.columns)-1):
    g = sns.FacetGrid(wine_data, col="quality", margin_titles=True)
    g.map(plt.hist,  wine_data.columns[i] )

    median = wine_data[wine_data.columns[i]].median()
    for ax in g.axes.flat:
        ax.plot((median, median), (0, 1599), c="r", ls="--")
 
plt.close()   
# Create quality buckets for better observation
# bucket 1 - 3 & 4
# bucket 2 - 5 & 6
# bucket 3 - 7 & 8
print('\nGráficas comparativas de las variables con el valor de la clase con buckets')
# Compare each feature againts the quality buckets      
wine_data_buckets = wine_data
wine_data_buckets["quality_bucket"]=[1 if x <5 else 2 if x < 7 else 3 for x 
         in wine_data_buckets['quality']]
    
plt.gcf().clear()   
for i in range(0, len(wine_data_buckets.columns)-2):
    g = sns.FacetGrid(wine_data_buckets, col="quality_bucket", margin_titles=True)
    g.map(plt.hist,  wine_data_buckets.columns[i] )

    median = wine_data[wine_data_buckets.columns[i]].median()
    for ax in g.axes.flat:
        ax.plot((median, median), (0, 1599), c="r", ls="--")
plt.show()
plt.close()   
# Calculation of alcohol grades in high quality wines

aggregation_by_quality_bucket = wine_data_buckets[['alcohol', 'quality_bucket']].groupby(
['quality_bucket']).agg('count')

median_alcohol = wine_data_buckets['alcohol'].median()

df_high_level_alcohol = wine_data_buckets[wine_data_buckets['alcohol']>median_alcohol]

aggregation = df_high_level_alcohol[['alcohol', 'quality_bucket']].groupby(
['quality_bucket']).agg('count')

print('El porcentaje de vinos de alta calidad con un nivel de alcohol por encima de la mediana es {0:.2f}' 
      .format(aggregation.iloc[2,0] * 100/ aggregation_by_quality_bucket.iloc[2,0]))

pylab.gcf().clear()

print('\nGráficas de correlación')
# Correlation alcohol & quality

pylab.scatter(wine_data_buckets['quality'],wine_data_buckets['alcohol'])
z = np.polyfit(wine_data_buckets['quality'],
    wine_data_buckets['alcohol'],1)
p = np.poly1d(z)
pylab.plot(wine_data_buckets['quality'],p(wine_data_buckets['quality']),'r')
pylab.xlabel('quality')  
pylab.ylabel('alcohol')
pylab.show()

pylab.gcf().clear()
# Correlation Volatile acidity & quality
pylab.scatter(wine_data_buckets['quality'],wine_data_buckets['volatile acidity'])
z = np.polyfit(wine_data_buckets['quality'],
    wine_data_buckets['volatile acidity'],1)
p = np.poly1d(z)
pylab.plot(wine_data_buckets['quality'],p(wine_data_buckets['quality']),'r')
pylab.xlabel('quality')  
pylab.ylabel('volatile acidity')
pylab.show()

pylab.gcf().clear()
# Correlation sulphates & quality
pylab.scatter(wine_data_buckets['quality'],wine_data_buckets['sulphates'])
z = np.polyfit(wine_data_buckets['quality'],
    wine_data_buckets['sulphates'],1)
p = np.poly1d(z)
pylab.plot(wine_data_buckets['quality'],p(wine_data_buckets['quality']),'r')  
pylab.xlabel('quality')  
pylab.ylabel('Sulphates')

pylab.show()

# Compare correlated variables to quality class
plt.gcf().clear()
# Correlation 'total sulfur dioxide' & 'free sulfur dioxide' vs quality
g = sns.lmplot(x = 'total sulfur dioxide', y = 'free sulfur dioxide', data = wine_data_buckets,
           hue = 'quality_bucket')
print('Mediana de total sulfur dioxide: {}' 
      .format(wine_data_buckets['total sulfur dioxide'].median()))
print('Mediana de free sulfur dioxide: {}'
      .format(wine_data_buckets['free sulfur dioxide'].median()))
plt.show()
plt.gcf().clear()
# Correlation 'pH' & 'fixed acidity' vs quality
g = sns.lmplot(x = 'pH', y = 'fixed acidity', data = wine_data_buckets,
           hue = 'quality_bucket')
print('Mediana de pH: {}' 
      .format(wine_data_buckets['pH'].median()))
print('Mediana de fixed acidity: {}'
      .format(wine_data_buckets['fixed acidity'].median()))
plt.show()

plt.gcf().clear()
# Correlation 'pH' & 'citric acid' vs quality
g = sns.lmplot(x = 'pH', y = 'citric acid', data = wine_data_buckets,
           hue = 'quality_bucket')
print('Mediana de pH: {}' 
      .format(wine_data_buckets['pH'].median()))
print('Mediana de citric acid: {}'
      .format(wine_data_buckets['citric acid'].median()))
plt.show()

plt.gcf().clear()
# Correlation 'fixed acidity' & 'density' vs quality
g = sns.lmplot(x = 'fixed acidity', y = 'density', data = wine_data_buckets,
           hue = 'quality_bucket')
print('Mediana de fixed acidity: {}' 
      .format(wine_data_buckets['fixed acidity'].median()))
print('Mediana de density: {}'
      .format(wine_data_buckets['density'].median()))
plt.show

plt.gcf().clear()
# Correlation 'volatile acidity' & 'citric acid' vs quality
g = sns.lmplot(x = 'volatile acidity', y = 'citric acid', data = wine_data_buckets,
           hue = 'quality_bucket')
print('Mediana de volatile acidity: {}' 
      .format(wine_data_buckets['volatile acidity'].median()))
print('Mediana de citric acid: {}'
      .format(wine_data_buckets['citric acid'].median()))
plt.show()
plt.close()
#######################
##  Hypothesis test  ##
#######################

print('\n-- Sección Hipothesis test --')
# The test will be done for the high quality wines vs. normal wines

# Check whether the data is normally distributed
g = pylab.hist(wine_data_buckets['alcohol'])
pylab.xlabel('alcohol level')  
pylab.show()

print('\n- Sub-sección Test Mann-Whitney -')
print("Media de alcohol de los vinos de alta calidad {}" .format(df_high_quality['alcohol'].mean()))
print("Mediana de alcohol de los vinos de alta calidad {}" .format(df_high_quality['alcohol'].median()))

print("Media de alcohol de los vinos normales {}" .format(df_normal_quality['alcohol'].mean()))
print("Mediana de alcohol de los vinos normales {}" .format(df_normal_quality['alcohol'].median()))



# As the length of both datasets is way different, in order to run the test
# a subset of the normal_quality wine dataset is created with the length
# of the high quality dataset
df_normal_quality_ss = df_normal_quality[['alcohol']].sample(len(df_high_quality))



statistic, p_value = stats.mannwhitneyu(df_high_quality[['alcohol']],
                                        df_normal_quality_ss[['alcohol']],
                                        alternative = 'less')


print('Resultado del test Mann-Whitney')
print('Statistic: {}, p-value {}' .format(statistic, p_value))

## show that actually middle quality wines have the same alcohol levels than 
## high quality alcohols

print('\n- Sub-sección Cálculo de Probabilidad -')
print('Muestra que los vinos tienen los mismos niveles de alcohol, indistintamente de la calidad')
plt.gcf().clear()

pylab.scatter(df_high_quality['quality'],df_high_quality['alcohol'], c='green')
pylab.scatter(df_normal_quality['quality'],df_normal_quality['alcohol'], c='red')
pylab.xlabel('Quality')  
pylab.ylabel('Alcohol level')
pylab.show()

# Probability calculation
# Calculate the probability of getting a high quality wine per alcohol level
#wine_data_alcohol_buckets = wine_data.copy()
wine_data_alcohol_buckets = wine_data_buckets.loc[:,['alcohol','quality_bucket']]

alcohol_bucket = ['[0-10)','[10-11)','[11-12)','[12-13)','[13-15)']

wine_data_alcohol_buckets["alcohol_level_bucket"]=[1 if x <10 else 2 if x < 11 else 
                         3 if x < 12 else 4 if x < 13 else 5 
                         for x in wine_data_alcohol_buckets['alcohol']]

alcohol_buckets_stats =  []
alcohol_levels = [1,2,3,4,5]
high_quality = 3


for i in alcohol_levels:   
    recs_alcohol_level = (wine_data_alcohol_buckets[wine_data_alcohol_buckets["alcohol_level_bucket"]==i])['alcohol'].count()
    recs_alcohol_quality = (wine_data_alcohol_buckets[(wine_data_alcohol_buckets["alcohol_level_bucket"]==i) & 
                              (wine_data_alcohol_buckets["quality_bucket"]==high_quality)])['alcohol'].count()
    probability = round(((recs_alcohol_quality/recs_alcohol_level)*100),2)
    
    alcohol_buckets_stats.append([i, recs_alcohol_level, recs_alcohol_quality, probability])

df= pd.DataFrame(alcohol_buckets_stats)   
df.columns=['Nivel de alcohol','Total registros', 'Registros alta calidad','Probabilidad']


print('\nCálculo de probabilidad de obtener un vino de alta calidad por nivel de alcohol')
print(df)

plt.gcf().clear()
plt.plot(alcohol_bucket,df['Probabilidad'])
plt.xlabel('alcohol levels')
plt.ylabel('Probability to get a high quality wine - %')
plt.show()

    