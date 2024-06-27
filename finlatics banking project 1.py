# FINLATICS DATA ANALYTICS PROJECT
#By SAGAR CHANDAN

#Data Cleansing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'C:\Users\ASUS\Desktop\Banking\banking_data.csv')

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

print(df.info())

print(df.columns)

unique_counts=df.nunique()
print(unique_counts)


print(df.describe())

print(df['age'].describe())

print(df['marital'].value_counts())
print(df['marital_status'].value_counts())

df.drop(columns=['marital_status'],inplace = True)

print(df.columns)

print(df['housing'].value_counts())
print(df['loan'].value_counts())

print(df.isnull().sum())

print(df['education'].value_counts())
print(df['marital'].value_counts())

df['education'] = df['education'].fillna(df['education'].mode()[0])
df['marital'] = df['marital'].fillna(df['marital'].mode()[0])

print(df['education'].value_counts())
print(df['marital'].value_counts())


print(df['default'].value_counts())
print(df['default'].value_counts())


df.rename(columns={'y':'subscription'},inplace=True)


#Questions


print(df['age'].describe())
print(df['age'].median())
print(df['age'].mode())
plt.hist(df['age'],color='turquoise',edgecolor='black', bins=20, alpha=0.9)
plt.xlabel('Age',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Distribution of Age',fontsize=14,fontweight='bold')
plt.show()

sns.boxplot(data=df['age'])
plt.title('Boxplot')
plt.ylabel('Values')
plt.show()


print(df['job'].describe())
print(df['job'].mode())
job_counts = df['job'].value_counts()
print(job_counts)
job_counts_sorted = job_counts.sort_values(ascending=False)

sns.barplot(x=job_counts_sorted.index, y=job_counts_sorted.values)
plt.xlabel('Job Type',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Variation in Job types',fontsize=14,fontweight='bold')
plt.xticks(rotation=315)
plt.tight_layout() 
plt.show()


marital_counts=df['marital'].value_counts()
print(marital_counts)
marital_counts_sorted=marital_counts.sort_values(ascending=False)

sns.barplot(x=marital_counts_sorted.index,y=marital_counts_sorted.values)
plt.xlabel('Marital Status',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Marital Status Distribution',fontsize=14,fontweight='bold')
plt.show()

wedgeprops = {'linewidth': 0.7, 'edgecolor': 'black'}
plt.pie(marital_counts, labels=marital_counts.index, autopct='%1.1f%%',startangle=70,wedgeprops=wedgeprops)
plt.title('Marital Status Distribution',fontsize=14,fontweight='bold')
plt.axis('equal')  
plt.show()


education_counts = df['education'].value_counts()
print(education_counts)
education_counts_sorted = education_counts.sort_values(ascending=False)

sns.barplot(x=education_counts_sorted.index, y=education_counts_sorted.values)
plt.xlabel('Level of Education',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Education Level Distribution',fontsize=14,fontweight='bold')
plt.show()

wedgeprops = {'linewidth': 0.7, 'edgecolor': 'black'}
plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%',startangle=70,wedgeprops=wedgeprops)
plt.title('Level of Education Distribution',fontsize=14,fontweight='bold')
plt.axis('equal')  
plt.show()


default_counts = df['default'].value_counts()
print(default_counts)
explode2 = (0, 0.5)
wedgeprops = {'linewidth': 1, 'edgecolor': 'black'}
plt.pie(default_counts, labels=default_counts.index, autopct='%1.1f%%',startangle=30,wedgeprops=wedgeprops,explode=explode2)
plt.title('Proportion of Clients with Credit in Default',fontsize=12,fontweight='bold')
plt.axis('equal')  
plt.show()

sns.barplot(x=default_counts.index, y=default_counts.values)
plt.xlabel('Default',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Proportion of Clients with Credit in Default ',fontsize=14,fontweight='bold')
plt.show()


print(df['balance'].describe())
print(df['balance'].median())
print(df['balance'].mode())

plt.hist(df['balance'], bins=100, color='lightgreen', edgecolor='black', alpha=0.6)
plt.xlabel('Average Yearly Balance', fontsize=11, fontweight='bold')
plt.ylabel('Frequency',fontsize=11, fontweight='bold')
plt.title('Distribution of Average Yearly Balance')
plt.grid(True)
plt.show()

sns.boxplot(data=df['balance'])
plt.title('Boxplot')
plt.ylabel('Values')
plt.show()

 
housing_counts=df['housing'].value_counts()
print(housing_counts)
housing_counts_sorted = housing_counts.sort_values(ascending=False)

sns.barplot(x=housing_counts_sorted.index, y=housing_counts_sorted.values)
plt.xlabel('Housing Loan',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Housing Loan Distribution',fontsize=14,fontweight='bold')
plt.show()

explode2 = (0, 0.1)
wedgeprops = {'linewidth': 1.5, 'edgecolor': 'black'}
colors=['#F97306','#069AF3']
plt.pie(housing_counts, labels=housing_counts.index, autopct='%1.1f%%',startangle=100,explode=explode2,shadow=False,wedgeprops=wedgeprops,colors=colors)
plt.title('Proportion of Clients with Housing Loan',fontsize=14,fontweight='bold')
plt.axis('equal')  
plt.show()


Ploan_counts=df['loan'].value_counts()
print(Ploan_counts)
Ploan_counts_sorted = Ploan_counts.sort_values(ascending=False)

sns.barplot(x=Ploan_counts_sorted.index, y=Ploan_counts_sorted.values)
plt.xlabel('Personal Loan',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Personal Loan Distribution',fontsize=14,fontweight='bold')
plt.show()

explode2 = (0, 0.1)
wedgeprops = {'linewidth': 1.5, 'edgecolor': 'black'}
plt.pie(Ploan_counts, labels=Ploan_counts.index, autopct='%1.1f%%',startangle=60,explode=explode2,wedgeprops=wedgeprops)
plt.title('Proportion of Clients with Personal Loan',fontsize=12,fontweight='bold')
plt.axis('equal')  
plt.show()


contact_counts=df['contact'].value_counts()
print(contact_counts)
contact_counts_sorted=contact_counts.sort_values(ascending=False)

sns.barplot(x=contact_counts_sorted.index, y=contact_counts_sorted.values)
plt.xlabel('Mode of Communication',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Mode of Communication Distribution',fontsize=14,fontweight='bold')
plt.show()

wedgeprops = {'linewidth': 1.5, 'edgecolor': 'black'}
plt.pie(contact_counts, labels=contact_counts.index, autopct='%1.1f%%',startangle=60,wedgeprops=wedgeprops)
plt.title('Proportion of Communication Methods',fontsize=12,fontweight='bold')
plt.axis('equal')  
plt.show()


print(df['day'].describe())
mode=df['day'].mode()[0]
print(df['day'].mode())
modal_freq=(df['day']==mode).sum()
print(modal_freq)

plt.hist(df['day'], bins=31, color='purple', edgecolor='black', alpha=0.7)
plt.xlabel('Day of the month', fontsize=11, fontweight='bold')
plt.ylabel('Frequency',fontsize=11, fontweight='bold')
plt.title('Distribution of Last Contact Day',fontsize=14, fontweight='bold')
plt.grid(True)
plt.show()

sns.boxplot(data=df['day'])
plt.title('Boxplot')
plt.ylabel('Values')
plt.show()


desired_order = ['jan', 'feb', 'mar','apr', 'may', 'jun','jul', 'aug', 'sep','oct', 'nov', 'dec']
df['month'] = df['month'].astype(pd.CategoricalDtype(categories=desired_order, ordered=True))
df_sorted = df.sort_values('month')

print(df['month'].value_counts())
sns.countplot(x='month',data=df)
plt.xlabel('Month',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Distribution of Last Contact Month',fontsize=14,fontweight='bold')
plt.show()


print(df['duration'].describe())
mode=df['duration'].mode()[0]
print(df['duration'].mode())
modal_freq1=(df['duration']==mode).sum()
print(modal_freq1)

plt.hist(df['duration'], bins=100, color='orange', edgecolor='black', alpha=0.7)
plt.xlabel('Duration of Last Contact (seconds)',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Distribution of Duration of Last Contact',fontsize=14,fontweight='bold')
plt.grid(True)
plt.tight_layout() 
plt.show()

sns.boxplot(data=df['duration'])
plt.title('Boxplot')
plt.ylabel('Values')
plt.show()


print(df['campaign'].describe())
mode=df['campaign'].mode()[0]
print(df['campaign'].mode())
modal_freq=(df['campaign']==mode).sum()
print(modal_freq)

plt.hist(df['campaign'], bins=50, color='orange', edgecolor='black', alpha=0.7)
plt.xlabel('Number of Contacts during the Campaign',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Distribution of Number of Contacts during the Campaign',fontsize=14,fontweight='bold')
plt.grid(True)
plt.tight_layout() 
plt.show()

sns.boxplot(data=df['campaign'])
plt.title('Boxplot')
plt.ylabel('Values')
plt.show()


print(df['pdays'].describe())
mode=df['pdays'].mode()[0]
print(df['pdays'].mode())
modal_freq=(df['pdays']==mode).sum()
print(modal_freq)

plt.hist(df['pdays'], bins=20, color='orange', edgecolor='black', alpha=0.7)
plt.xlabel('Number of Days Passed since Last Contact',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Distribution of Number of Days Passed since Last Contact',fontsize=14,fontweight='bold')
plt.grid(True)
plt.tight_layout() 
plt.show()

sns.boxplot(data=df['pdays'])
plt.title('Boxplot')
plt.ylabel('Values')
plt.show()


print(df['previous'].describe())
mode=df['previous'].mode()[0]
print(df['previous'].mode())
modal_freq=(df['previous']==mode).sum()
print(modal_freq)

plt.hist(df['previous'],bins=50,color='orange', edgecolor='black', alpha=0.7)
plt.xlabel('Number of Contacts performed before the current campaign',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Distribution of Number of contacts performed before the current campaign',fontsize=14,fontweight='bold')
plt.grid(True)
plt.tight_layout() 
plt.show()

sns.boxplot(data=df['previous'])
plt.title('Boxplot')
plt.ylabel('Values')
plt.show()


Poutcome_counts=df['poutcome'].value_counts()
print(Poutcome_counts)
Poutcome_counts_sorted = Poutcome_counts.sort_values(ascending=False)

sns.barplot(x=Poutcome_counts_sorted.index, y=Poutcome_counts_sorted.values)
plt.xlabel('Outcome of previous Campaign',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Campaign Outcome Distribution',fontsize=14,fontweight='bold')
plt.show()

wedgeprops = {'linewidth': 0.7, 'edgecolor': 'black'}
plt.pie(Poutcome_counts, labels=Poutcome_counts.index, autopct='%1.1f%%',startangle=60,wedgeprops=wedgeprops)
plt.title('Outcome of Marketing Campaign ',fontsize=12,fontweight='bold')
plt.axis('equal')  
plt.show()


subscription_counts = df['subscription'].value_counts()
print(subscription_counts)
explode2 = (0, 0.1)
wedgeprops = {'linewidth': 1.5, 'edgecolor': 'black'}
plt.pie(subscription_counts, labels=subscription_counts.index, autopct='%1.1f%%',startangle=30,explode=explode2,wedgeprops=wedgeprops)
plt.title('Proportion of Clients who subscribed to the term deposit ',fontsize=12,fontweight='bold')
plt.axis('equal')  
plt.show()

sns.barplot(x=subscription_counts.index, y=subscription_counts.values)#,palette='Set2' )
plt.xlabel('Subscribed',fontsize=11,fontweight='bold')
plt.ylabel('Frequency',fontsize=11,fontweight='bold')
plt.title('Proportion of Clients who subscribed to the term deposit ',fontsize=14,fontweight='bold')
plt.show()


print(df['subscription'].value_counts())
df['subscription'] = df['subscription'].replace({'yes': 1, 'no': 0})
print(df['subscription'].value_counts())

print(df['loan'].value_counts())
df['loan'] = df['loan'].replace({'yes': 1, 'no': 0})
print(df['loan'].value_counts())

print(df['housing'].value_counts())
df['housing'] = df['housing'].replace({'yes': 1, 'no': 0})
print(df['housing'].value_counts())

print(df['default'].value_counts())
df['default'] = df['default'].replace({'yes': 1, 'no': 0})
print(df['default'].value_counts())


numeric_df=df.select_dtypes(include=['int64','float64',])
corr_matrix=numeric_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True, cmap='YlOrRd', fmt=".2f",)
plt.title('Correlation Matrix of Subscription Factors', fontsize=14,fontweight='bold')
plt.show()


sns.countplot(x="job", hue="subscription", data=df)
plt.xlabel("Job Type ",fontsize=11,fontweight='bold')
plt.ylabel("Promotion success",fontsize=11,fontweight='bold')
plt.title("Job Type Success Rate",fontsize=14,fontweight='bold')
plt.xticks(rotation=315)
plt.show()

sns.countplot(x="marital", hue="subscription", data=df)
plt.xlabel("Marital Status",fontsize=11,fontweight='bold')
plt.ylabel("Promotion success",fontsize=11,fontweight='bold')
plt.title("Marital Status Success Rate",fontsize=14,fontweight='bold')
plt.show()

sns.countplot(x="education", hue="subscription", data=df)
plt.xlabel("Education Level",fontsize=11,fontweight='bold')
plt.ylabel("Promotion success",fontsize=11,fontweight='bold')
plt.title("Education level Success Rate",fontsize=14,fontweight='bold')
plt.show()

sns.countplot(x="contact", hue="subscription", data=df)
plt.xlabel("Mode of Communication",fontsize=11,fontweight='bold')
plt.ylabel("Promotion success",fontsize=11,fontweight='bold')
plt.title("Mode of Communication Success Rate",fontsize=14,fontweight='bold')
plt.show()

sns.countplot(x="month", hue="subscription", data=df)
plt.xlabel("Month of Contact",fontsize=11,fontweight='bold')
plt.ylabel("Promotion success",fontsize=11,fontweight='bold')
plt.title("Month of contact Success Rate",fontsize=14,fontweight='bold')
plt.show()

sns.countplot(x="poutcome", hue="subscription", data=df)
plt.xlabel("Previous Campaign Outcome ",fontsize=11,fontweight='bold')
plt.ylabel("Promotion success",fontsize=11,fontweight='bold')
plt.title("Previous Campaign Outcome Success Rate",fontsize=14,fontweight='bold')
plt.show()




