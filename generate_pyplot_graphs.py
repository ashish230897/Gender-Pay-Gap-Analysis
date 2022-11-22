import warnings
import itertools
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_style('whitegrid')

data = pd.read_csv('/../home/pranav/Downloads/multipleChoiceResponses.csv')

question_names = data.iloc[0]
data = data.drop(0, axis=0) # removing the row of questions

data = data[data['Q1'] != "Prefer not to say"]
data = data[data['Q1'] != "Prefer to self-describe"] #keeping only male and female as gender
data = data[data['Q9'].notnull()]
data = data[data['Q9'] != 'I do not wish to disclose my approximate yearly compensation'] #keeping only numerical ranges as salary range

# converting ranges to median values
mapping = {'0-10,000': 5000, '10-20,000': 15000, '20-30,000': 25000, '30-40,000': 35000, 
       '40-50,000': 45000, '50-60,000': 55000, '60-70,000': 65000, '70-80,000': 75000, 
       '80-90,000': 85000, '90-100,000': 95000, '100-125,000': 112500, 
       '125-150,000': 137500, '150-200,000': 175000, '200-250,000': 225000, 
       '250-300,000': 275000, '300-400,000': 350000, '400-500,000': 450000, 
       '500,000+':500000}

data['target'] = data['Q9'].apply(lambda x: mapping[x])


country_dic = {'Morocco': 'Africa',
             'Tunisia': 'Africa',
             'Austria': 'Europe',
             'Hong Kong (S.A.R.)': 'Asia',
             'Republic of Korea': 'Asia',
             'Thailand': 'Asia',
             'Czech Republic': 'Europe',
             'Philippines': 'Asia',
             'Romania': 'Europe',
             'Kenya': 'Africa',
             'Finland': 'Europe',
             'Norway': 'Europe',
             'Peru': 'South America',
             'Iran, Islamic Republic of...': 'Middle East',
             'Bangladesh': 'Asia',
             'New Zealand': 'Oceania',
             'Egypt': 'Africa',
             'Chile': 'South America',
             'Belarus': 'Europe',
             'Hungary': 'Europe',
             'Ireland': 'Europe',
             'Belgium': 'Europe',
             'Malaysia': 'Asia',
             'Denmark': 'Europe',
             'Greece': 'Europe',
             'Pakistan': 'Asia',
             'Viet Nam': 'Asia',
             'Argentina': 'South America',
             'Colombia': 'South America',
             'Indonesia': 'Oceania',
             'Portugal': 'Europe',
             'South Africa': 'Africa',
             'South Korea': 'Asia',
             'Switzerland': 'Europe',
             'Sweden': 'Europe',
             'Israel': 'Middle East',
             'Nigeria': 'Africa',
             'Singapore': 'Asia',
             'I do not wish to disclose my location': 'dna',
             'Mexico': 'North America',
             'Ukraine': 'Europe',
             'Netherlands': 'Europe',
             'Turkey': 'Asia',
             'Poland': 'Europe',
             'Australia': 'Oceania',
             'Italy': 'Europe',
             'Spain': 'Europe',
             'Japan': 'Asia',
             'France': 'Europe',
             'Canada': 'North America', 
             'United Kingdom of Great Britain and Northern Ireland': 'Europe',
             'Germany': 'Europe',
             'Brazil': 'South America',
             'Russia': 'Russia',
             'Other': 'Other',
             'China': 'China',
             'India':'India',
             'United States of America': 'USA'}

data['Q3'] = data['Q3'].apply(lambda x: country_dic[x])

def gen_regionwise_dist(data):
    plt.figure(figsize = (15,10))
    sns.violinplot(cut=0, x = 'Q3', y = 'target', hue = 'Q1', data = data, split=True, order = data['Q3'].value_counts().index)
    plt.ylabel("Yearly Income ($)", fontsize=12)
    plt.xlabel("Nationality", fontsize=12)
    plt.title("Illustration of the Gender Wage Gap for Different Regions", fontsize=15)
    plt.savefig("Regionwise distribution of responses.png")

   
gen_regionwise_dist(data)

data = data[data['Q4'] != "I prefer not to answer"]
data = data[data['Q4'] != "No formal education past high school"]
data = data[data['Q4'] != 'Some college/university study without earning a bachelor’s degree']

degree = []
for i in data["Q4"].unique():
    degree.append(i)
    

def gen_degreewise_dist(data,degree):
    degree_holders = data[data["Q4"] == str(degree)]
    plt.figure(figsize = (15,10))
    sns.violinplot(cut=0, x = 'Q3', y = 'target', hue = 'Q1', data = degree_holders, split=True, order = degree_holders['Q3'].value_counts().index)
    plt.ylabel("Yearly Income ($)", fontsize=12)
    plt.xlabel("Nationality", fontsize=12)
    plt.title("Gender Wage Gap for Different Regions for people having {dname}".format(dname = str(degree)), fontsize=15)
    plt.savefig("{dname}.png".format(dname = str(degree)))


for j,i in enumerate(degree): 
    gen_degreewise_dist(data,degree[j])

undergrad_dict = {'Engineering (non-computer focused)': 'Engineering(non CS) major',
                  'Computer science (software engineering, etc.)': 'Computer science major',
                  'A business discipline (accounting, economics, finance, etc.)': 'Business major',
                  'Medical or life sciences (biology, chemistry, medicine, etc.)': 'Medical science major',
                  'Humanities (history, literature, philosophy, etc.)': 'Humanities major'}

def gen_majorwise_dist(data,major,major_alias):
    degree_holders = data[data["Q5"] == str(major)]
    plt.figure(figsize = (15,10))
    sns.violinplot(cut=0, x = 'Q3', y = 'target', hue = 'Q1', data = degree_holders, split=True, order = degree_holders['Q3'].value_counts().index)
    plt.ylabel("Yearly Income ($)", fontsize=12)
    plt.xlabel("Nationality", fontsize=12)
    plt.title("Gender Wage Gap for Different Regions for people having {dname}".format(dname = str(major_alias)), fontsize=15)
    plt.savefig("{dname}.png".format(dname = str(major_alias)))

    
for i in undergrad_dict.keys(): 
    gen_majorwise_dist(data,i,undergrad_dict[i])
    
def gen_box_plots(data,factor,factor_alias):
    age_order = ['18-21', '22-24', '25-29', '30-34','35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70-79', '80+']
    plt.figure(figsize=(15,10))
    if(factor=="Q2"):
        sns.boxplot(x=str(factor), y='target', data=data, order=age_order, showfliers=False)
    elif(factor=="Q5" or factor=="Q6"):
        plt.xticks(rotation=-70)
        sns.boxplot(x=str(factor), y='target', data=data, showfliers=False)
    else:
        sns.boxplot(x=str(factor), y='target', data=data, showfliers=False)
    plt.ylabel("Yearly Income ($)", fontsize=12)
    plt.xlabel(str(factor_alias), fontsize=12)
    plt.title("Distribution of the Yearly Income according to {fname}".format(fname = str(factor_alias)), fontsize=15)
    plt.savefig("{fname}.png".format(fname = str(factor_alias)))


analysis_dict = {'Q2': 'Age',
                  'Q3': 'Region',
                  'Q5': 'Undergrad major',
                  'Q6': 'Occupation'}

for i in analysis_dict.keys():
    gen_box_plots(data,i,analysis_dict[i])