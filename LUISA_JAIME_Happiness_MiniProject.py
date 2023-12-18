#!/usr/bin/env python
# coding: utf-8

# ## **Project Requirements**

# **1.High-Level View [2 pts]**
# 
# 
# This dataset includes a wide range of factors for countries worldwide, including their region, happiness rank, happiness score, low confidence interval, upper confidence interval, economy (GDP per capita), family, health (life expectancy), freedom, trust(corruption), generosity and dystopia residual. This dataset can be used for various tasks such as studying global well-being trends, understanding the impact of economic and social variables on what we understand of "happiness". It also helps us identify specific patterns in key factors that can influence lifestyles.
# 
# **2. Preliminary Exploration [4 pts]**
# 
# 
# 
# • Just by looking at the data, I can tell there is some missing information for certain countries in specific factors. By missing data I am referring to the 0 values in some columns, but that zero could mean so many different things. I will not remove it, I will take it as a value. To report the data is noisy I will need to plot the data, and see the distribution and spreadness of it. Another way to see if there is any noise in our data is to create boxplots for numerical columns and report the outliers.
# • To clean this dataframe I will either choose to drop rows with missing values or impute them with methods like mean, median, etc.
# 
# •In the next steps I will expose how I worked the preliminary exploration
# 
# **3. Defining objectives - [3 pts]**
# 
# *Factors influencing happiness scores*
# 
# 
# •From GDP, freedom, and trust in government exhibit the strongest correlation with the happiness score
# 
# 
# *Continental variations in happiness*
# 
# 
# • Are there significant variations in happiness scores among different continents or regions?
# 
# 
# *Comparing the family  and happiness score*
# 
# 
# •*Comparing the family  and happiness score*
# 
# 
# •Is there any relationship between region, happiness score, and the influence on family?
# 
# • Extra, does the saddest country based on happiness score compared to the happy country (based on happiness score) have a bigger value on family?

# #  First Step

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('https://drive.google.com/uc?id=1PcCyXXs28wvhAcatjBqa48hf9JD8_Xqa&export=download')


# In[4]:


df.shape


# In[5]:


#to look for the first couple of rows
df.head()


# In[6]:


#to look for the last rows
df.tail()


# In[7]:


#conditions you are dealing with
df.columns


# #Preliminary Exploration

# I decided to start by using the function describe(), which will provide me a quick summary of the statistical properties of my data set.  It will help me  understand better the information I am working with

# In[8]:


df.describe()


# df.info() will help me asses the structure and characteristics of my dataframe. If there's a potential issue I will be able to notice it by using this function.

# In[9]:


df.info()


# It seems there are no issues with this data set. It seems there is no missing data. I will still need to check for outliers and data visualization to get a better understanding of it.
# 

# I am pretty sure there are no missing values, but still, to be 100% I will be running the next code: .isna().sum(). This will provide me a series that will show me if there is a missing value in each column of my DataFrame

# In[10]:


print(df.isna().sum())


# I can conclude there are no missing values.
# 
# 
# 
# 

# In[11]:


plt.figure(figsize=(30,20))
df.hist(bins=40, color='blue',grid=True)

plt.tight_layout(rect=[0, 0, 1.5, 0.96])


# I decided to create a histogram plot for all values. This will provide me a quick visual overview of the distribution of values in each column.

# Based on this preliminary exploration I can conclude the data set is pretty clean. Therefore I will continue with the next steps, which are working on my objectives.
# 

# # 3. Defining objectives
# 

# *Factors influencing happiness scores*
# 
# 
# •From GDP, freedom, and trust in government exhibit the strongest correlation with the happiness score
# 

# I am looking for :
# 
#  Factors influencing happiness scores : GDP, freedom, and trust in government exhibit the strongest correlation with happiness score

# In[12]:


df_filtered=df[['Happiness Score','Economy (GDP per Capita)','Freedom','Trust (Government Corruption)']]


# Based on the factors I wanted to work with, my next step was to filter the data.

# In[13]:


df_filtered.head()


# In[14]:


df.columns


# An easy way that allows me to quickly identify patterns and trends in this filtered data is working with heatmaps.

# In[15]:


plt.figure(figsize=(10,8))
plt.title('Correlation plot between  Hapiness score and other factors')
#heatmap will take entire data frame
sns.heatmap(df_filtered.corr(),annot=True,fmt=".4f")
plt.show()


# This is a good way to visualize whether one feature depends on another or not. Correlation goes from -1 to 1
# 
# • -1 is negatively correlated
# 
# • 0 means there is no particular correlation
# 
# • If it lies between ±0.5 and ± 1, then there is a strong correlation
# 
# • If values lies between ±0.30 and ±0.49 then there is a medium correlation
# 
# •  If the values lies below ±0.29 it is a small correlation
# 
# 
# • If the color is dark it means is negatively correlated
# 
# • if it starts to become reddish it starts becoming positively correlated

# In[16]:


sns.pairplot(df_filtered)


# In[17]:


plt.scatter(df_filtered['Happiness Score'],df_filtered['Economy (GDP per Capita)'])
sns.regplot(x='Happiness Score', y='Economy (GDP per Capita)', data=df_filtered, scatter=False, color='red')
plt.xlabel('Happiness Score')
plt.ylabel('Economy (GDP per Capita)')
plt.title(' Happiness Score vs. GDP per Capita')


# • The scatterplot of the Happiness Score vs GDP shows a strong positive correlation. This suggests that, on average, as GDP increases, the Happiness Score increases as well. As we know, correlation does not imply causation, but it gives us an idea of the association between economic prosperity and how happy a country is.

# In[18]:


plt.scatter(df_filtered['Happiness Score'],df_filtered['Freedom'])
sns.regplot(x='Happiness Score', y='Freedom', data=df_filtered, scatter=False, color='red')
plt.xlabel('Happiness Score')
plt.ylabel('Freedom')
plt.title(' Happiness Score vs. Freedom')


# • The scatterplot of Happiness Score vs Freedom shows a  positive correlation. This suggests that, on average, as Freedom increases, the Happiness Score increases as well. As we know, correlation does not imply causation, but it gives us an idea of the association between freedom and how happy a country is.

# In[20]:


plt.scatter(df_filtered['Happiness Score'],df_filtered['Trust (Government Corruption)'])
sns.regplot(x='Happiness Score', y='Trust (Government Corruption)', data=df_filtered, scatter=False, color='red')
plt.xlabel('Happiness Score')
plt.ylabel('Trust (Government Corruption)')
plt.title(' Happiness Score vs. Trust (Government Corruption)')


# • This scatterplot is not as easy to analyze as the ones before. In my opinion, the regression line does not suggest much, and I cannot appreciate a correlation between these two variables. This leads to the conclusion that the average level of trust people have in their government does not necessarily reflect how happy a country is.

# # •  Histogram plots
# I decided to plot all the histograms together differentiating them by colors. This helped me describe the distribution better.

# In[22]:


fig,axes=plt.subplots(1,4, figsize=(20,5))
sns.histplot(df_filtered, x='Happiness Score',color='purple' ,ax=axes[0])
sns.histplot(df_filtered, x='Freedom',color='skyblue', ax=axes[1])
sns.histplot(df_filtered, x='Economy (GDP per Capita)',color='green', ax=axes[2])
sns.histplot(df_filtered, x='Trust (Government Corruption)',color='pink', ax=axes[3])


# *Happiness score*
# 
# •Just by looking at the histogram, it is showing a potential "normal" distribution. By looking at the information described, the mean is approximately 5.38, with a minimum score of 2.905 and a max of 7.526. The standard deviation is 1.14, indicating a moderate level of variability around the mean
# 
# *Freedom*
# 
# •The histogram shows that the distribution is skewed to the left (mean<median).The left skewness suggests that there are more countries with lower levels of freedom than those with higher levels.
# 
# *Economy*
# 
# •The histogram shows that the distribution is skewed to the left.  (mean<median). The left skewness suggests that there are more countries with lower GDP per capita than those with higher GDP per capita.
# 
# *Trust*
# 
# • The histogram shows the distribution is skewed to the right (mean>median). The right skewness suggests that there are more countries with lower levels of trust in government corruption than those with higher levels.
# 

# 
# # Boxplots
# • I wanted to create a boxplot to gain a better idea of the outliers. This will help me understand the noise in my data more effectively. As you can observe, you can actually appreciate the different boxplots in this graph. I will continue to create a boxplot for each column I am interested in

# In[24]:


custom_palette = sns.color_palette("pastel")
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_filtered, palette=custom_palette, showfliers=False)
plt.xticks(rotation=45)


# In[25]:


sns.boxplot( data=df_filtered['Happiness Score'],color='purple')
plt.xticks(rotation=45)
plt.xlabel('Category')
plt.ylabel('Happiness Score')
plt.title('Boxplot of Happiness Score')


# In[26]:


sns.boxplot( data=df_filtered['Freedom'],color='skyblue')
plt.xticks(rotation=45)
plt.xlabel('Category')
plt.ylabel('Freedom')
plt.title('Boxplot of Freedome')


# In[27]:


sns.boxplot( data=df_filtered['Economy (GDP per Capita)'],color='green')
plt.xticks(rotation=45)
plt.xlabel('Category')
plt.ylabel('Economy (GDP per Capita)')
plt.title('Boxplot of Economy (GDP per Capita)')


# In[28]:


sns.boxplot( data=df_filtered['Trust (Government Corruption)'],color='pink')
plt.xticks(rotation=45)
plt.xlabel('Category')
plt.ylabel('Trust (Government Corruption)')
plt.title('Boxplot of Trust (Government Corruption)')


# •Observing the boxplots separately, I can conclude that the only category that is showing some outliers is Trust (government corruption). What I will do next is to calculate the IQR, because the first conclusion that comes to my mind is that there is noisy data in this column.

# In[35]:


trust_column =df_filtered ['Trust (Government Corruption)']
Q1 = trust_column.quantile(0.25)
Q3 = trust_column.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
potential_outliers = trust_column[(trust_column < lower_bound) | (trust_column > upper_bound)]
potential_outliers


# Outliers are defined as values beyond 1.5 times the IQR. This set aims to draw attention to these points and highlight the presence of noise in the Trust column of the DataFrame.

# ## *Continental variations in happiness*
# 
# 
# • Are there significant variations in happiness scores among different continents or regions?

# For the next goal, I filtered the DataFrame only with the specific columns I intend to use.
# The head() is just a way to confirm the data was filtered correctly

# In[39]:


df_happiness_filtered=df[['Happiness Score','Region','Country']]
df_happiness_filtered.head()


# This code df_happiness_filtered['Region'].unique() will show me the specific regions I will be working with

# In[43]:


df_happiness_filtered['Region'].unique()


# In[44]:


region_counts = df_happiness_filtered['Region'].value_counts()
region_counts


# •For this goal, I want to observe if there is any difference in happiness between different regions. I filtered the data by region, happiness score, and country. The country might not be significant since I will just be working with the regional happiness score. To know exactly which regions I will be working with I used the next code: region_counts.

# In[45]:


Sub = df_happiness_filtered[df_happiness_filtered['Region'] == 'Sub-Saharan Africa']
Central = df_happiness_filtered[df_happiness_filtered['Region'] == 'Central and Eastern Europe']
Latin = df_happiness_filtered[df_happiness_filtered['Region'] == 'Latin America and Caribbean']
West = df_happiness_filtered[df_happiness_filtered['Region'] == 'Western Europe']
Middle = df_happiness_filtered[df_happiness_filtered['Region'] == 'Middle East and Northern Africa']
SoutheastA = df_happiness_filtered[df_happiness_filtered['Region'] == 'Southeastern Asia']
SouthA = df_happiness_filtered[df_happiness_filtered['Region'] == 'Southern Asia']
EastA = df_happiness_filtered[df_happiness_filtered['Region'] == 'Eastern Asia']
NorthA = df_happiness_filtered[df_happiness_filtered['Region'] == 'North America']
Aus = df_happiness_filtered[df_happiness_filtered['Region'] == 'Australia and New Zealand']


# • Once I have grouped the regions, I will proceed by comparing their average happiness scores. The mean will help me some sense of the happiness level of each region. Making it easier to compare them.
# 
# 

# In[46]:


mean_Sub = Sub['Happiness Score'].mean()
mean_Cent = Central['Happiness Score'].mean()
mean_Latin = Latin['Happiness Score'].mean()
mean_West = West['Happiness Score'].mean()
mean_Middle = Middle['Happiness Score'].mean()
mean_SoutheastA = SoutheastA['Happiness Score'].mean()
mean_SouthA = SouthA['Happiness Score'].mean()
mean_EastA = EastA['Happiness Score'].mean()
mean_NorthA = NorthA['Happiness Score'].mean()
mean_Aus = Aus['Happiness Score'].mean()


# I decided to create a dictionary that I will later use to create a new data frame using pandas. This will help me have a better organization to understand the mean values of each region better .

# In[47]:


data_region = {'Region': ['Sub', 'Central', 'Latin', 'West', 'Middle', 'SoutheastA', 'SouthA', 'EastA', 'NorthA', 'Aus'],'Mean': [mean_Sub, mean_Cent, mean_Latin, mean_West, mean_Middle, mean_SoutheastA, mean_SouthA, mean_EastA, mean_NorthA, mean_Aus]}


# In[48]:


new_df = pd.DataFrame(data_region)
new_df


# After creating the new DataFrame, I decided to create a histogram. This is a very simple way to compare the mean happiness score across different regions.

# In[49]:


colors = ['red', 'orange', 'skyblue', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'fuchsia']
plt.bar(new_df['Region'], new_df['Mean'], color=colors)
plt.xlabel('Region')
plt.ylabel('Mean Happiness Score')
plt.title('Mean Happiness Score by Region')
plt.xticks(rotation=45, ha='right')


# 
# • The obtained data reveals a diverse range of mean happiness scores across various regions. It is evident, that happiness scores vary significantly from one region to another. North America, Australia, and New Zealand stand out with high mean happiness scores. However, it's essential to note that these regions are based on data from just two countries each.
# 
# Additionally, the region with the lowest mean happiness score is Sub-Saharan Africa, which has the highest number of countries. I recognize the limitation of using the mean, as extreme observations can heavily influence it. To mitigate this, I intend to continue working with the median and interquartile range (IQR) to determine which countries appear to be the happiest.
# 

# In[50]:


new_df.describe


#  I decided to group the data by Region. Then I created a DataFrame with the region information as an index and the happiness score as a column.  I applied the agg function to the happiness score for each region. In this aggregation function, I specified the list of values I was interested in mean, median, and standard deviation and lastly, I used the Lamba function. The lambda function helped me compute the IQR of each region.

# In[ ]:


grouped_df = df_happiness_filtered.groupby('Region')['Happiness Score'].agg(['mean', 'median', np.std, lambda i: i.quantile(0.75) - i.quantile(0.25)]).reset_index()
grouped_df.columns = ['Region', 'Mean', 'Median', 'Standard Deviation', 'IQR']
grouped_df


# I then decided to create a boxplot with these results.
# 
# ---
# 
# 

# In[51]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='Region', y='Happiness Score', data=df_happiness_filtered, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title('Boxplot of Happiness Score for Each Region')
plt.xlabel('Region')
plt.ylabel('Happiness Score')


# • So after all those calculations and graphs, I decided to create a table where I can observe the data that later on will be used for the boxplot. With a quick visualization working with the boxplot, I can conclude that the regions of Western Europe, Australia New Zealand, and North America by looking at their happiness score appear to be "happier" than the rest of the regions. Of course, it would be very premature to establish that they are the happiest regions in the world, but based on this statistical analysis that's what it appears to be.

# # •*Comparing the family  and happiness score*
# 
# 
# •Is there any relationship between region, happiness score, and the influence on family?
# 
# • Extra, does the saddest country based on happiness score compared to the happy country (based on happiness score) have a bigger value on family?

# For the next goal, I filtered the DataFrame only with the specific columns I intend to use.
# The head() is just a way to confirm the data was filtered correctly

# In[52]:


df_filt = df[['Region',	'Country',	 'Happiness Score', 'Family']]


# In[53]:


df_filt.head()


# 
# My last goal is to analyze and understand the relationship between region, happiness score, and family values. I decided to plot a scatterplot and show the regression line

# In[54]:


plt.scatter(df_filt['Happiness Score'], df_filt['Family'])
sns.regplot(x='Happiness Score', y='Family', data=df_filt, scatter=False, color='red')
plt.xlabel('Happiness Score')
plt.ylabel('Family')
plt.title(' Happiness Score vs. Family')


# To analyze the correlation better I created a  heatmap

# In[56]:


plt.figure(figsize=(10,8))
plt.title('Correlation plot between  Hapiness score and Family')
#heatmap will take entire data frame
sns.heatmap(df_filt.corr(),annot=True,fmt=".4f")
plt.show()


# This is a good way to visualize whether one feature depends on another or not. Correlation goes from -1 to 1
# 
# • -1 is negatively correlated
# 
# • 0 means there is no particular correlation
# 
# • If it lies between ±0.5 and ± 1, then there is a strong correlation
# 
# • If values lies between ±0.30 and ±0.49 then there is a medium correlation
# 
# •  If the values lies below ±0.29 it is a small correlation
# 
# 
# • If the color is dark it means is negatively correlated
# 
# • if it starts to become reddish it starts becoming positively correlated

# Based on the previous information Happiness score and the family value appear to have a strong correlation; since the value is 0.79. We could conclude that family does have a positive influence on the happiness score of each region/country.
# 

# Now just because I curious, I decided to compare the least happy country in the world which i named saddest to the max happy country in the world (happiest country)

# In[ ]:


saddest_country=df_filt.loc[df_filt['Happiness Score'].idxmin()]
saddest_country


# In[ ]:


happiest_country=df_filt.loc[df_filt['Happiness Score'].idxmax()]
happiest_country


# Then I asked myself what If i compared the top 10 happiest regions/countries in the world with the top 10 saddest ones

# In[58]:


top_10_happiest = df_filt.sort_values(by=['Happiness Score','Family'], ascending=False).head(10)
top_10_saddest = df_filt.sort_values(by=['Happiness Score','Family']).head(10)


# And because i wanted to plot this information I proceed to combine the data from these two variables into one.

# In[60]:


combined = pd.concat([top_10_happiest, top_10_saddest])
combined


# To finish my analysis, I decided to create an arrangement that would allow me for a side-by-side visual comparison of Happiness scores and Family values for both regions an countries. I believe this will help me identify easily the patterns and differences and even come to a conclusion.

# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(25, 5))
sns.barplot(x='Country', y='Happiness Score', data=combined, palette='Blues', ax=axes[0])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[0].set_title('Top 10 Happiest and Saddest Countries')
sns.barplot(x='Region', y='Happiness Score', data=combined, palette='Purples', ax=axes[1])
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set_title('Top 10 Happiest and Saddest Countries')
sns.barplot(x='Country', y='Family', data=combined, palette='Blues', ax=axes[2])
axes[2].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[2].set_title('Top 10 Happiest and Saddest Countries Family values ')
sns.barplot(x='Region', y='Family', data=combined, palette='Purples', ax=axes[3])
axes[3].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[3].set_title('Top 10 Happiest and Saddest Countries Family values ')
plt.show()


# Combining the graphs with the data obtained on the heatmap. I can conclude that indeed the correlation between happiness score and family is strong. It shows that the family value provokes a positive response in the happiness score. Nonetheless, correlation does not imply causation, it would be very difficult to determine if these top 10 countries are indeed happier because the family values.

# ### Ethics
# 
# 
# While this set helped me work on my skills in Python and it challenged me, I do have some concerns. Firstly, it only provides data for 157 countries out of a total of 195. The exclusion of 38 countries raises questions about representativeness and potential bias. On a personal note, I am not a big fan of how they divided the regions; it seems somewhat arbitrary and potentially biased. Additionally, the lack of clarity on the interpretation of values poses a challenge. If someone examines my results, they may face the same confusion I experienced. Understanding the values is crucial since inaccuracies in interpretation could lead to incorrect conclusions.
# 
# Lastly, there is a notable ethical concern regarding how the happiness score is determined based on specific factors. The rationale behind choosing these factors is not explicitly explained, raising questions about transparency and potential biases in the scoring system. It's essential to address these concerns for a more comprehensive and reliable analysis.
# 

# In[ ]:




