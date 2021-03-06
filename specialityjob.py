# -*- coding: utf-8 -*-
"""specialityjob.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1csr7b0o4zE9BIcbfasDS3A6gW4HDf3Uv
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import random
from mlxtend.frequent_patterns import apriori, association_rules
from tqdm import tqdm
from gensim.models import Word2Vec 
import warnings;
warnings.filterwarnings('ignore')

ds = pd.read_excel("data.xlsx",index_col = 0)
pd.options.display.max_columns = None
display(ds)

jobs = ds["jobTitle"].unique().tolist()
len(jobs)

jobs

ds['location']= ds['location'].astype(str)
ds['company']= ds['company'].astype(str)
ds['speciality']= ds['speciality'].astype(str)
ds['fullName']= ds['fullName'].astype(str)
ds['jobTitle']= ds['jobTitle'].astype(str)
ds['skill1']= ds['skill1'].astype(str)
ds['skill2']= ds['skill2'].astype(str)
ds['skill3']= ds['skill3'].astype(str)
ds['skill4']= ds['skill4'].astype(str)
ds['skill5']= ds['skill5'].astype(str)
ds['skill6']= ds['skill6'].astype(str)

jobs = ds["jobTitle"].unique().tolist()
len(jobs)

jobs

# shuffle companies 
random.shuffle(jobs)

# extract 80% of companies names in a list to train our model with
jobs_list = [jobs[i] for i in range(round(0.8*len(jobs)))]

# split data into train and test set
train_job = ds[ds['jobTitle'].isin(jobs_list)]
test_job = ds[~ds['jobTitle'].isin(jobs_list)]

train_job

# list to capture job locations associated to the companies
jobs_train_list = []

# populate the list with the location codes
for i in tqdm(jobs_list):
    temp = train_job[train_job["jobTitle"] == i]["speciality"].tolist()
    jobs_train_list.append(temp)

jobs_train_list

# list to capture job locations associated to the companies
jobs_test_list = []

# populate the list with the location codes
for i in tqdm(jobs_list):
    temp = test_job[test_job["jobTitle"] == i]["speciality"].tolist()
    jobs_test_list.append(temp)

jobs_test_list

# train word2vec model
model3 = Word2Vec(window = 10, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model3.build_vocab(jobs_train_list, progress_per=200)

model3.train(jobs_train_list, total_examples = model3.corpus_count, 
            epochs=10, report_delay=1)

model3.init_sims(replace=True)

# extract all vectors
X = model3.wv.index2entity
X

jobsp = test_job[["speciality","jobTitle"]]

# remove duplicates
jobsp.drop_duplicates(inplace=True, subset='speciality', keep="last")

# jobs.dropna(inplace=True, subset='jobLocation')


# create job titles associated to location and comapnies associated to location dictionary
job_dict = jobsp.groupby('speciality')['jobTitle'].apply(list).to_dict()

job_dict['DS']

job_dict

def similar_jobs(v, n = 6):
    
    # extract most similar locations for the input vector
    ms = model3.wv.similar_by_vector(v, topn= n+1)[1:]
    
    # extract name and similarity score of the similar locations
    new_ms = []
    for j in ms:
        pair = (job_dict[j[0]][0], j[1])
        new_ms.append(pair)
        
    return new_ms




