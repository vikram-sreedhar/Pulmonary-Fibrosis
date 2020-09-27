#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[2]:


# Visualisation libraries
import matplotlib.pyplot as plt


# In[3]:


import seaborn as sns
sns.set()
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
import pycountry
py.init_notebook_mode(connected=True)
import folium 
from folium import plugins


# Graphics in retina format 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
#plt.rcParams['image.cmap'] = 'viridis'

# palette  colors to be used for plots
colors = ["steelblue","dodgerblue","lightskyblue","powderblue","cyan","deepskyblue","cyan","darkturquoise","paleturquoise","turquoise"]


# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')


# In[4]:


from pathlib import Path


# In[5]:


from IPython.display import YouTubeVideo
YouTubeVideo('1Kyo9Hcyiq0', width=800, height=300)


# In[6]:


get_ipython().run_line_magic('pwd', '')


# In[7]:


os.chdir('D:\Kaggle\Pulmonary Fibrosis')


# In[8]:


get_ipython().run_line_magic('pwd', '')


# In[9]:


## Reading input and directory path

train = pd.read_csv('train.csv')
dataset_dir = 'D:\\Kaggle\\Pulmonary Fibrosis\\train'


# In[10]:


train


# In[95]:


test = pd.read_csv('test.csv')


# In[96]:


test


# In[13]:


## Reading test and train data

print('Train:\n',train.head(5),'\n')
print(train.isna().sum())
print('\n---------------------------------------------------------------------------\n')
print('Test:\n',test.head(5),'\n')
print(test.isna().sum())


# In[14]:


train.info()


# In[15]:


train.describe()


# In[16]:


dataset_dir


# In[17]:


train.shape[0]


# In[18]:


test.shape[0]


# In[19]:


INPUT = Path("D:/Kaggle/Pulmonary Fibrosis/train")


# In[20]:


INPUT


# In[21]:


train.Patient.agg(['nunique','count'])


# In[22]:


test.Patient.agg(['nunique','count'])


# In[23]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.countplot(train.Sex, palette="Reds_r", ax=ax[0]);
ax[0].set_xlabel("")
ax[0].set_title("Gender counts");

sns.countplot(test.Sex, palette="Blues_r", ax=ax[1]);
ax[1].set_xlabel("")
ax[1].set_title("Gender counts");


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


fig, axs = plt.subplots(ncols=3)
fig.set_size_inches(19,6)
sns.countplot(train['SmokingStatus'],ax=axs[0])
sns.countplot(train['SmokingStatus'][train['Sex']=="Male"],ax=axs[1])
sns.countplot(train['SmokingStatus'][train['Sex']=="Female"],ax=axs[2])
fig.savefig("output2.jpeg")


# In[26]:


# Select unique bio info for the patients
agg_train = train.groupby(by="Patient")[["Patient", "Age", "Sex", "SmokingStatus"]].first().reset_index(drop=True)

# Figure
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16, 6))

a = sns.distplot(agg_train["Age"], ax=ax1, hist=False, kde_kws=dict(lw=6, ls="--"))
b = sns.countplot(agg_train["Sex"], ax=ax2)
c = sns.countplot(agg_train["SmokingStatus"], ax=ax3)

a.set_title("Patient Age Distribution", fontsize=16)
b.set_title("Sex Frequency", fontsize=16)
c.set_title("Smoking Status", fontsize=16);


# In[27]:


fig, axs = plt.subplots(ncols=3)
fig.set_size_inches(19,6)
sns.countplot(test['SmokingStatus'],ax=axs[0])
fig.savefig("output3.jpeg")


# In[28]:


fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.distplot(train.Age,kde=False,bins=80,color="k")
fig.savefig("output4.jpeg")


# In[29]:


fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.distplot(test.Age,kde=False,bins=80,color="k")
fig.savefig("output5.jpeg")


# In[30]:


print("Min FVC value: {:,}".format(train["FVC"].min()), "\n" +
      "Max FVC value: {:,}".format(train["FVC"].max()), "\n" +
      "\n" +
      "Min Percent value: {:.4}%".format(train["Percent"].min()), "\n" +
      "Max Percent value: {:.4}%".format(train["Percent"].max()))

# Figure
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))

a = sns.distplot(train["FVC"], ax=ax1, hist=False, kde_kws=dict(lw=6, ls="--"))
b = sns.distplot(train["Percent"], ax=ax2, hist=False, kde_kws=dict(lw=6, ls="-."))

a.set_title("FVC Distribution", fontsize=16)
b.set_title("Percent Distribution", fontsize=16);


# In[31]:


print("Minimum no. weeks before CT: {}".format(train['Weeks'].min()), "\n" +
      "Maximum no. weeks after CT: {}".format(train['Weeks'].max()))

plt.figure(figsize = (16, 6))

a = sns.distplot(train['Weeks'], hist=False, kde_kws=dict(lw=8, ls="--"))
plt.title("Number of weeks before/after the CT scan", fontsize = 16)
plt.xlabel("Weeks", fontsize=14);


# In[32]:


def create_baseline():
    first_scan=pd.DataFrame()    
    for i in train.Patient.unique():
        first_scan=first_scan.append((train[train['Patient']=="{}".format(i)][:1]))
    first_scan=first_scan.drop("Patient",axis=1)
    first_scan=first_scan.drop("Weeks",axis=1)
    return first_scan
fc=create_baseline()
fc=fc.reset_index(drop=True)
fc.head()


# In[33]:


fc


# In[34]:


(sns.pairplot(train,hue="SmokingStatus",height=4)).savefig("output5.jpeg")


# In[35]:


sns.pairplot(fc,hue="SmokingStatus",height=4).savefig("output6.jpeg")


# In[36]:


fig, ax = plt.subplots(nrows=2)
fig.set_size_inches(22, 8.27)
sns.lineplot(x='Weeks',y='Percent',data=train,ax=ax[0]).set_title("All Patients Percent trend",fontsize=15,y=0.85)
sns.lineplot(x='Weeks',y='FVC',data=train,ax=ax[1]).set_title("All Patients FVC trend",fontsize=15,y=0.85)
fig.savefig("weeksfvccomp.jpeg")


# In[37]:


# FVC and Percent trend Males vs Females

males=train[train["Sex"]=="Male"]
females=train[train["Sex"]=="Female"]


# In[38]:


fig, ax = plt.subplots(nrows=4)
fig.set_size_inches(22, 22)
sns.lineplot(x='Weeks',y='FVC',data=males,ax=ax[0]).set_title("MALES FVC TREND", fontsize=15,y=0.85)
sns.lineplot(x='Weeks',y='FVC',data=females,ax=ax[1]).set_title("FEMALES FVC TREND", fontsize=15,y=0.85)
sns.lineplot(x='Weeks',y='Percent',data=males,ax=ax[2]).set_title("MALES PERCENT TREND", fontsize=15,y=0.85)
sns.lineplot(x='Weeks',y='Percent',data=females,ax=ax[3]).set_title("FEMALES PERCENT TREND", fontsize=15,y=0.85)
fig.savefig("malevsfemalesfvc_percenttrend.jpeg")


# In[39]:


# FVC and Percent trend Smokers vs nonsmokers for all patients

smoker=train[train["SmokingStatus"]=="Ex-smoker"]
never_smoked=train[train["SmokingStatus"]=="Never smoked"]
current_smoker=train[train["SmokingStatus"]=="Currently smokes"]


# In[40]:


fig, ax = plt.subplots(nrows=6)
fig.set_size_inches(22, 35)
sns.lineplot(x='Weeks',y='FVC',data=smoker,ax=ax[0]).set_title("EX SMOKER FVC TREND",fontsize=15,y=0.90)
sns.lineplot(x='Weeks',y='FVC',data=never_smoked,ax=ax[1]).set_title("NON SMOKER FVC TREND",fontsize=15,y=0.90)
sns.lineplot(x='Weeks',y='FVC',data=current_smoker,ax=ax[2]).set_title("SMOKER FVC TREND",fontsize=15,y=0.90)
sns.lineplot(x='Weeks',y='Percent',data=smoker,ax=ax[3]).set_title("EX SMOKER PERCENT  TREND",fontsize=15,y=0.90)
sns.lineplot(x='Weeks',y='Percent',data=never_smoked,ax=ax[4]).set_title("NON SMOKER PERCENT TREND",fontsize=15,y=0.90)
sns.lineplot(x='Weeks',y='Percent',data=current_smoker,ax=ax[5]).set_title("SMOKER PERCENT TREND",fontsize=15,y=0.90)
fig.savefig("weeksvpercent_smokervsnonsmoker.jpeg")


# In[41]:


# creating Age-Bins in train data

category = pd.cut(train.Age,bins = [49,55,65,75,85,120],labels=['<=55','56-65','66-75','76-85','85+'])
train.insert(5,'Age_Bins',category)


# In[42]:



f, (ax1, ax2) = plt.subplots(1,2, figsize = (16, 6))

a = sns.barplot(x = train["SmokingStatus"], y = train["FVC"], ax=ax1)
b = sns.barplot(x = train["SmokingStatus"], y = train["Percent"], ax=ax2)

a.set_title("Mean FVC per Smoking Status", fontsize=16)
b.set_title("Mean Perc per Smoking Status", fontsize=16);


# In[43]:


f, (ax1, ax2) = plt.subplots(1,2, figsize = (16, 6))

a = sns.barplot(x = train["Age_Bins"], y = train["FVC"], hue = train["Sex"], ax=ax1)
b = sns.barplot(x = train["Age_Bins"], y = train["Percent"], hue = train["Sex"], ax=ax2)

a.set_title("Mean FVC per Gender per Age category", fontsize=16)
b.set_title("Mean Perc per Gender per Age Category", fontsize=16);


# In[44]:


f, (ax1, ax2) = plt.subplots(1,2, figsize = (16, 6))

a = sns.barplot(x = train["Age_Bins"], y = train["FVC"], hue = train["SmokingStatus"], ax=ax1)
b = sns.barplot(x = train["Age_Bins"], y = train["Percent"], hue = train["SmokingStatus"], ax=ax2)

a.set_title("Mean FVC per Smoking_status per Age category", fontsize=16)
b.set_title("Mean Perc per Smoking_status per Age Category", fontsize=16);


# In[45]:


plt.figure(figsize=(16,10))
sns.heatmap(train.corr(),annot=True)


# In[46]:


import scipy


# In[47]:


# Compute Correlation
corr1, _ = scipy.stats.pearsonr(train["FVC"], train["Percent"])
corr2, _ = scipy.stats.pearsonr(train["FVC"], train["Age"])
corr3, _ = scipy.stats.pearsonr(train["Percent"], train["Age"])
print("Pearson Corr FVC x Percent: {:.4}".format(corr1), "\n" +
      "Pearson Corr FVC x Age: {:.0}".format(corr2), "\n" +
      "Pearson Corr Percent x Age: {:.2}".format(corr3))


# In[48]:


train.describe()


# In[49]:


train.info()


# In[50]:


# creating Age-Bins in fc data

category = pd.cut(fc.Age,bins = [49,55,65,75,85,120],labels=['<=55','56-65','66-75','76-85','85+'])
fc.insert(5,'Age_Bins',category)


# In[51]:


fc.info()


# In[52]:


fc.describe()


# In[53]:


f, (ax1, ax2) = plt.subplots(1,2, figsize = (16, 6))

a = sns.barplot(x = fc["Age_Bins"], y = fc["FVC"], hue = fc["SmokingStatus"], ax=ax1)
b = sns.barplot(x = fc["Age_Bins"], y = fc["Percent"], hue = fc["SmokingStatus"], ax=ax2)

a.set_title("Patient FVC per Smoking_status per Age category", fontsize=16)
b.set_title("Patinet Perc per Smoking_status per Age Category", fontsize=16);


# In[54]:


import pydicom


# In[55]:


import os
import json
from pathlib import Path
from glob import glob


# In[56]:


from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *
from fastai.medical.imaging import *

import pydicom,kornia,skimage


# In[57]:


try:
    import cv2
    cv2.setNumThreads(0)
except: pass

import seaborn as sns
sns.set(style="whitegrid")
sns.set_context("paper")


# In[58]:


#Visulising Dicom Files

files = Path('D:/Kaggle/Pulmonary Fibrosis/train')


# In[59]:


train_files = get_dicom_files(files)


# In[60]:


train_files


# In[61]:


info_view = train_files[33025]
dimg = dcmread(info_view)
dimg


# In[62]:


#There are some 'key' aspects within the header:

#(0018, 0015) Body Part Examined CS: Chest: images are from the chest area

#(0020, 0013) Instance Number IS: "99": this is the same as the .dcm image file

#(0020, 0032) Image Position (Patient) DS: [-191, -29, -241.200012]: represents the x, y and z positions

#(0020, 0037) Image Orientation (Patient) DS: [1, 0, 0, 0, 1, 0]: This is 6 values that represent two 
#normalized 3D vectors(in this case directions) where the first vector [1,0,0] represents Xx, Xy, Xz and the 
#second vector [0,1,0] that represents Yx, Yy, Yz.

#(0028, 0004) Photometric Interpretation CS: MONOCHROME2: aka the colorspace, images are being stored 
#as low values=dark, high values=bright. If the colorspace was MONOCHROME then the low values=bright and high values=dark.

#(0028, 0100) Bits Allocated US: 16: each image is 16 bits

#(0028, 1050) Window Center DS: "-500.0" : aka Brightness

#(0028, 1051) Window Width DS: "-1500.0" : aka Contrast

#(0028, 1052) Rescale Intercept DS: "-1024.0" and (0028, 1053) Rescale Slope DS: "1.0": 
#The Rescale Intercept and Rescale Slope are applied to transform the pixel values of the image into values that 
#are meaningful to the application. It's importance is explained further in the kernel.

#(7fe0, 0010) Pixel Data OW: Array of 524288 elements: the image pixel data that pydicom uses to convert the pixel data 
#into an image.

#This can be calculated by this formula:
#Array of elements = Rows X Columns X Number of frames X Samples per pixel X (bits_allocated/8) 
#so in this example it would be 512 X 512 X 1 X 1 X (16/8) = 524288


# In[63]:


dimg.PixelData[:33025]


# In[218]:


dimg.pixel_array


# In[64]:


dimg.pixel_array.shape


# In[65]:


dimg.show()


# In[66]:


import pydicom as dicom
import PIL # optional
import pandas as pd
import matplotlib.pyplot as plt


# In[67]:


# Metdata of dicomfiles extracted as dataframe

df_dicom = pd.DataFrame.from_dicoms(train_files)


# In[68]:


df_dicom


# In[69]:


df_dicom.describe()


# In[70]:


df_dicom.info()


# In[71]:


df_dicom.head()


# In[72]:


get_ipython().run_line_magic('pwd', '')


# In[73]:


df_dicom.to_csv('df_dicom.csv')


# In[74]:


unique_patient_df = train.drop(['Weeks', 'FVC', 'Percent'], axis=1).drop_duplicates().reset_index(drop=True)
unique_patient_df['# visits'] = [train['Patient'].value_counts().loc[pid] for pid in unique_patient_df['Patient']]

print('Number of data points: ' + str(len(unique_patient_df)))
print('----------------------')

for col in unique_patient_df.columns:
    print('{} : {} unique values, {} missing.'.format(col, 
                                                          str(len(unique_patient_df[col].unique())), 
                                                          str(unique_patient_df[col].isna().sum())))
unique_patient_df.head()


# In[75]:


#Convert to JPG and extracting all information in one go..

import pydicom as dicom
import matplotlib.pyplot as plt
import os
import cv2
import PIL # optional
import pandas as pd
import csv
# make it True if you want in PNG format
PNG = False
# Specify the .dcm folder path
folder_path = 'D:/Kaggle/Pulmonary Fibrosis/train/ID00007637202177411956430/'
# Specify the .jpg/.png folder path
jpg_folder_path = 'D:\Kaggle\Pulmonary Fibrosis\Train_wkg'
images_path = os.listdir(folder_path)


# In[76]:


arr=dimg.pixel_array


# In[77]:


arr


# In[78]:


df_arr = pd.DataFrame(arr)


# In[79]:


df_arr


# In[80]:


from glob import glob


# In[81]:




PATH_dicom = os.path.abspath(os.path.join('D:/Kaggle/Pulmonary Fibrosis', 'Train_jpg'))


# In[82]:


images_dicom = glob(os.path.join(PATH_dicom, "*.jpg"))


# In[83]:


images_dicom[0:5]


# In[84]:


images_dicom[0:5]


# In[85]:


r = random.sample(images_dicom, 3)
r


# In[86]:


plt.figure(figsize=(16,16))
plt.subplot(131)
plt.imshow(cv2.imread(r[0]))

plt.subplot(132)
plt.imshow(cv2.imread(r[1]))

plt.subplot(133)
plt.imshow(cv2.imread(r[2]));


# In[87]:


get_ipython().run_line_magic('pwd', '')


# In[88]:



submission = pd.read_csv('sample_submission.csv')


# In[89]:


train.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])


# In[90]:


train


# In[91]:


submission


# In[92]:


submission['Patient'] = (
    submission['Patient_Week']
    .apply(
        lambda x:x.split('_')[0]
    )
)

submission['Weeks'] = (
    submission['Patient_Week']
    .apply(
        lambda x: int(x.split('_')[-1])
    )
)

submission = submission[['Patient','Weeks','FVC', 'Confidence','Patient_Week']]

submission = submission.merge(test.drop('Weeks', axis=1), on="Patient")


# In[93]:


submission


# In[97]:


test


# In[98]:


train['Dataset'] = 'train'
test['Dataset'] = 'test'
submission['Dataset'] = 'submission'


# In[99]:


submission


# In[100]:


all_data = train.append([test, submission])

all_data = all_data.reset_index()
all_data = all_data.drop(columns=['index'])


# In[101]:


all_data.head()


# In[102]:


all_data['FirstWeek'] = all_data['Weeks']
all_data.loc[all_data.Dataset=='submission','FirstWeek'] = np.nan
all_data['FirstWeek'] = all_data.groupby('Patient')['FirstWeek'].transform('min')


# In[103]:


first_fvc = (
    all_data
    .loc[all_data.Weeks == all_data.FirstWeek][['Patient','FVC']]
    .rename({'FVC': 'FirstFVC'}, axis=1)
    .groupby('Patient')
    .first()
    .reset_index()
)

all_data = all_data.merge(first_fvc, on='Patient', how='left')


# In[104]:


all_data.head()


# In[105]:


all_data


# In[106]:


all_data['WeeksPassed'] = all_data['Weeks'] - all_data['FirstWeek']


# In[107]:


all_data


# In[108]:


#Calculating derived field of height from First FVC value
# Reference - https://en.wikipedia.org/wiki/Vital_capacity#:~:text=It%20is%20equal%20to%20the,a%20wet%20or%20regular%20spirometer

def calculate_height(row):
    if row['Sex'] == 'Male':
        return row['FirstFVC'] / (27.63 - 0.112 * row['Age'])
    else:
        return row['FirstFVC'] / (21.78 - 0.101 * row['Age'])

all_data['Height'] = all_data.apply(calculate_height, axis=1)


# In[109]:


all_data.head()


# In[110]:


all_data = pd.concat([
    all_data,
    pd.get_dummies(all_data.Sex),
    pd.get_dummies(all_data.SmokingStatus)
], axis=1)

all_data = all_data.drop(columns=['Sex', 'SmokingStatus'])


# In[111]:


all_data.head()


# In[112]:


def scale_feature(series):
    return (series - series.min()) / (series.max() - series.min())

all_data['Percent'] = scale_feature(all_data['Percent'])
all_data['Age'] = scale_feature(all_data['Age'])
all_data['FirstWeek'] = scale_feature(all_data['FirstWeek'])
all_data['FirstFVC'] = scale_feature(all_data['FirstFVC'])
all_data['WeeksPassed'] = scale_feature(all_data['WeeksPassed'])
all_data['Height'] = scale_feature(all_data['Height'])


# In[113]:


feature_columns = [
   'Percent',
   'Age',
   'FirstWeek',
   'FirstFVC',
   'WeeksPassed',
   'Height',
   'Female',
   'Male', 
   'Currently smokes',
   'Ex-smoker',
   'Never smoked',
]


# In[114]:


train_new = all_data.loc[all_data.Dataset == 'train']
test_new = all_data.loc[all_data.Dataset == 'test']
submission_new = all_data.loc[all_data.Dataset == 'submission']


# In[115]:


train_new[feature_columns].head()


# In[116]:


train_new


# In[117]:


import sklearn
from sklearn import linear_model


# In[118]:


model = linear_model.LinearRegression()


# In[119]:


model.fit(train_new[feature_columns], train_new['FVC'])


# In[120]:


plt.bar(train_new[feature_columns].columns.values, model.coef_)
plt.xticks(rotation=90)
plt.show()


# In[121]:


from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[122]:


predictions = model.predict(train_new[feature_columns])

mse = mean_squared_error(
    train['FVC'],
    predictions,
    squared=False
)

mae = mean_absolute_error(
    train['FVC'],
    predictions
)

print('MSE Loss: {0:.2f}'.format(mse))
print('MAE Loss: {0:.2f}'.format(mae))


# In[123]:


print (model.coef_)


# In[124]:


print (model.intercept_)


# In[125]:


# Rsquare value for the model

model.score(train_new[feature_columns], train_new['FVC'])


# In[126]:


X = train_new[feature_columns]


# In[127]:


Y = train_new['FVC']


# In[128]:


X


# In[129]:


import statsmodels.formula.api as smf


# In[130]:


model_smf_1 = smf.ols(formula='Y~X',data = train_new).fit()


# In[131]:


train_new['prediction'] = predictions


# In[132]:


predictions


# In[133]:


model_smf_1.params


# In[134]:


prediction_smf = model_smf_1.predict(train_new[feature_columns])


# In[135]:


model_smf_1.summary()


# In[136]:


prediction_smf


# In[137]:


predictions


# In[138]:


prds_1_sklearn = pd.DataFrame(predictions)


# In[139]:


prds_1_sklearn


# In[140]:


prds_2_statstools = pd.DataFrame(prediction_smf)


# In[141]:


prds_2_statstools


# In[142]:


plt.scatter(predictions, train_new['FVC'])

plt.xlabel('predictions')
plt.ylabel('FVC (labels)')
plt.show()


# In[143]:


delta = predictions - train_new['FVC']
plt.hist(delta, bins=20)
plt.show()


# In[144]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In[145]:


train_patients = train_new.Patient.unique()


# In[146]:


fig, ax = plt.subplots(10, 1, figsize=(10, 20))

for i in range(10):
    patient_log = train_new[train_new['Patient'] == train_patients[i]]

    ax[i].set_title(train_patients[i])
    ax[i].plot(patient_log['WeeksPassed'], patient_log['FVC'], label='truth')
    ax[i].plot(patient_log['WeeksPassed'], patient_log['prediction'], label='prediction')
    ax[i].legend()


# In[149]:


submission_new

train_new
# In[152]:


from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[153]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# In[154]:


#Create a Gaussian Classifier
regr=RandomForestRegressor(random_state=0)
#Train the model using the training sets Y_pred=clf.predict(X_test)
regr.fit(X_train,Y_train)


# In[155]:


regr.n_estimators


# In[156]:


regr.estimators_[5]


# In[157]:


regr.get_params()


# In[158]:


regr.feature_importances_


# In[162]:


Y_pred = regr.predict(X_test)


# In[163]:


df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
df


# In[164]:


df1 = df.head(50)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[165]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


# In[166]:


from sklearn.tree import export_graphviz
import pydot


# In[167]:


tree = regr.estimators_[5]


# In[168]:


export_graphviz(tree,out_file = 'tree.dot',
                feature_names = X.columns,
                filled = True,
                rounded = True,precision = 1)


# In[169]:


(graph, ) = pydot.graph_from_dot_file('tree.dot')


# In[170]:


graph


# In[171]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


# In[172]:


graph.write_png('tree_graph.png')


# In[173]:


errors_test = abs(Y_pred - Y_test)


# In[174]:


# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors_test), 2), 'degrees.')
mape = np.mean(100 * (errors_test / Y_test))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


# In[175]:


Y_pred_train=regr.predict(X_train)


# In[176]:


df2 = pd.DataFrame({'Actual': Y_train, 'Predicted': Y_pred_train})
df2


# In[177]:


df3 = df2.head(50)
df3.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[178]:


errors_train = abs(Y_pred_train - Y_train)


# In[179]:


# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors_train), 2), 'degrees.')
mape_train = np.mean(100 * (errors_train / Y_train))
accuracy_train = 100 - mape_train
print('Accuracy:', round(accuracy_train, 2), '%.')


# In[234]:


import cv2
import os
import random
import matplotlib.pylab as plt
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[248]:



files_jpg = Path('D:/Kaggle/Pulmonary Fibrosis/Train_jpg1')


# In[250]:



images_jpg = glob(os.path.join(files_jpg, "*.jpg"))


# In[249]:


files_jpg


# In[252]:


images_jpg


# In[253]:


r_jpg = random.sample(images_jpg, 3)
r_jpg

# Matplotlib black magic
plt.figure(figsize=(16,16))
plt.subplot(131)
plt.imshow(cv2.imread(r_jpg[0]))

plt.subplot(132)
plt.imshow(cv2.imread(r_jpg[1]))

plt.subplot(133)
plt.imshow(cv2.imread(r_jpg[2]));    


# In[255]:


def proc_images():
    """
    Returns two arrays: 
        x is an array of resized images
    """
    
    
    x = [] # images as arrays
    WIDTH = 64
    HEIGHT = 64

    for img in images_jpg:
        base = os.path.basename(images_jpg)
        
        # Read and resize image
        full_size_image = cv2.imread(images_jpg)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))

        
    return x


# In[261]:


from PIL import Image


# In[267]:


IMG_DIR = 'D:/Kaggle/Pulmonary Fibrosis/Train_jpg1'

for img in os.listdir(IMG_DIR):
        img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)

        img_array = (img_array.flatten())

        img_array  = img_array.reshape(-1, 1).T

        print(img_array)
        with open('output.csv', 'ab') as f:

            np.savetxt(f, img_array, delimiter=",")
 


# In[281]:


img_array.shape


# In[279]:


os.listdir(IMG_DIR)


# In[278]:


img


# In[ ]:




