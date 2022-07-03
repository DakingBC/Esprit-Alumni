#-*- coding: utf-8 -*-
"""
Created on Wed May 11 23:02:41 2022

@author: kenza
"""

from flask  import Flask , render_template ,request
import pickle
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import random
#from mlxtend.frequent_patterns import apriori, association_rules
from tqdm import tqdm
from gensim.models import Word2Vec 
import warnings;
warnings.filterwarnings('ignore')

app = Flask(__name__)



@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')




@app.route('/intern', methods=['GET','POST'])
def intern():
    if request.method == 'POST':

        ds = pd.read_excel("data1.xlsx",index_col = 0)
        pd.options.display.max_columns = None
        
        ds['speciality']= ds['speciality'].astype(str)
        ds['company']= ds['company'].astype(str)
        
        companies = ds["company"].unique().tolist()

        random.shuffle(companies)

        companies_list = [companies[i] for i in range(round(0.8*len(companies)))]

        train_company = ds[ds['company'].isin(companies_list)]
        test_company = ds[~ds['company'].isin(companies_list)]

        companies_train_list = []

        for i in tqdm(companies_list):
            temp = train_company[train_company["company"] == i]["speciality"].tolist()
            companies_train_list.append(temp)


            companies_test_list = []

        for i in tqdm(test_company['company'].unique()):
            temp = test_company[test_company["company"] == i]["speciality"].tolist()
            companies_test_list.append(temp)

            model3 = Word2Vec(window = 10, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

            model3.build_vocab(companies_train_list, progress_per=200)

            model3.train(companies_train_list, total_examples = model3.corpus_count, 
                        epochs=10, report_delay=1)

            model3.init_sims(replace=True)
            
            X = model3.wv.index_to_key

            interns = test_company[["speciality","company"]]

            interns.drop_duplicates(inplace=True, subset='speciality', keep="last")

            company_dict = interns.groupby('speciality')['company'].apply(list).to_dict()

            def similar_intern(v, n = 3):   
    
                ms = model3.wv.similar_by_vector(v, topn= n+1)[1:]    
   
                new_ms = []
                for j in ms:
                    pair = company_dict[j[0]]
                    new_ms.append(pair)
        
                return new_ms        

            spc = request.form.get('spc')

            app.logger.warning(spc)

            app.logger.warning(request.form.values())  
            
            recommandation3 = similar_intern(spc)
    
            app.logger.info(recommandation3)

            return render_template('intern.html', recommandation_text='You should have your internship at : {}'.format(recommandation3))
        
            return redirect(url_for('index'))

    return render_template('intern.html')



@app.route('/job', methods=['GET', 'POST'])
def job():
    if request.method == 'POST':

        ds = pd.read_excel("data1.xlsx",index_col = 0)
        
        ds['location']= ds['location'].astype(str)
        ds['company']= ds['company'].astype(str)
        
        companies = ds["company"].unique().tolist()

        random.shuffle(companies)

        companies_list = [companies[i] for i in range(round(0.8*len(companies)))]

        train_company = ds[ds['company'].isin(companies_list)]
        test_company = ds[~ds['company'].isin(companies_list)]

        companies_train_list = []

        for i in tqdm(companies_list):
            temp = train_company[train_company["company"] == i]["location"].tolist()
            companies_train_list.append(temp)


        companies_test_list = []

        for i in tqdm(test_company['company'].unique()):
            temp = test_company[test_company["company"] == i]["location"].tolist()
            companies_test_list.append(temp)

        model1 = Word2Vec(window = 10, sg = 1, hs = 0,
             negative = 10, # for negative sampling
             alpha=0.03, min_alpha=0.0007,
             seed = 14)

        model1.build_vocab(companies_train_list, progress_per=200)

        model1.train(companies_train_list, total_examples = model1.corpus_count, 
                        epochs=10, report_delay=1)

        model1.init_sims(replace=True)
            
        X = model1.wv.index_to_key

        interns = test_company[["location","company"]]

        interns.drop_duplicates(inplace=True, subset='location', keep="last")

        company_dict = interns.groupby('location')['company'].apply(list).to_dict()

        def similar_companies(v, n = 3):   
    
            ms = model1.wv.similar_by_vector(v, topn= n+1)[1:]    
   
            new_ms = []
            for j in ms:
                pair = company_dict[j[0]]
                new_ms.append(pair)
        
            return new_ms        

        location = request.form.get('location')

        app.logger.warning(location)

        app.logger.warning(request.form.values())  
            
        recommandation1 = similar_companies(location)
    
        app.logger.info(recommandation1)

        return render_template('job.html', recommandation_text='You should work at : \n{}'.format(recommandation1))
        
        return redirect(url_for('index'))

    return render_template('job.html')


    

@app.route('/alumni', methods=['GET','POST'])
def alumni():
    if request.method == 'POST':

        ds = pd.read_excel("data1.xlsx",index_col = 0)
        pd.options.display.max_columns = None
        
        ds['speciality']= ds['speciality'].astype(str)
        ds['fullName']= ds['fullName'].astype(str)
        
        pr = ds["fullName"].unique().tolist()

        random.shuffle(pr)

        pr_list = [pr[i] for i in range(round(0.8*len(pr)))]

        train_pro = ds[ds['fullName'].isin(pr_list)]
        test_pro = ds[~ds['fullName'].isin(pr_list)]   

        prof_train = []

        for i in tqdm(pr_list):
            temp = train_pro[train_pro["fullName"] == i]["speciality"].tolist()
            prof_train.append(temp)  

        prof_test = []

        for i in tqdm(pr_list):
            temp = test_pro[test_pro["fullName"] == i]["speciality"].tolist()
            prof_test.append(temp)   

        model4 = Word2Vec(window = 10, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

        model4.build_vocab(prof_train, progress_per=200)

        model4.train(prof_train, total_examples = model4.corpus_count, 
                        epochs=10, report_delay=1)

        model4.init_sims(replace=True)

        X = model4.wv.index_to_key

        prospe = test_pro[["speciality","fullName"]]
           
        prospe.drop_duplicates(inplace=True, subset='speciality', keep="last")

        prof_dict = prospe.groupby('speciality')['fullName'].apply(list).to_dict()

        def similar_profiles(v, n = 3):
            
            ms = model4.wv.similar_by_vector(v, topn= n+1)[1:]
                
            new_ms = []
            for j in ms:
                pair = (prof_dict[j[0]][0])
                new_ms.append(pair)
                
            return new_ms       

        sp = request.form.get('sp')

        app.logger.warning(sp)

        app.logger.warning(request.form.values())  
            
        recommandation2 = similar_profiles(sp)
    
        app.logger.info(recommandation2)

        return render_template('alumni.html', recommandation_text='You should match with : {}'.format(recommandation2))
        
        return redirect(url_for('index'))

    
    return render_template('alumni.html')


@app.route('/profile', methods = ['GET', 'POST'])
def profile():
    if request.method == 'POST':

        ds = pd.read_excel("data1.xlsx",index_col = 0)
        pd.options.display.max_columns = None
        
        ds['speciality']= ds['speciality'].astype(str)
        ds['fullName']= ds['fullName'].astype(str)
        
        pr = ds["fullName"].unique().tolist()

        random.shuffle(pr)

        pr_list = [pr[i] for i in range(round(0.8*len(pr)))]

        train_pro = ds[ds['fullName'].isin(pr_list)]
        test_pro = ds[~ds['fullName'].isin(pr_list)]   

        prof_train = []

        for i in tqdm(pr_list):
            temp = train_pro[train_pro["fullName"] == i]["speciality"].tolist()
            prof_train.append(temp)  

        prof_test = []

        for i in tqdm(pr_list):
            temp = test_pro[test_pro["fullName"] == i]["speciality"].tolist()
            prof_test.append(temp)   

        model4 = Word2Vec(window = 10, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

        model4.build_vocab(prof_train, progress_per=200)

        model4.train(prof_train, total_examples = model4.corpus_count, 
                        epochs=10, report_delay=1)

        model4.init_sims(replace=True)

        X = model4.wv.index_to_key

        prospe = test_pro[["speciality","fullName"]]
           
        prospe.drop_duplicates(inplace=True, subset='speciality', keep="last")

        prof_dict = prospe.groupby('speciality')['fullName'].apply(list).to_dict()

        def similar_profiles(v, n = 3):
            
            ms = model4.wv.similar_by_vector(v, topn= n+1)[1:]
                
            new_ms = []
            for j in ms:
                pair = (prof_dict[j[0]][0])
                new_ms.append(pair)
                
            return new_ms       

        pr = request.form.get('pr')

        app.logger.warning(pr)

        app.logger.warning(request.form.values())  
            
        recommandation2 = similar_profiles(pr)
    
        app.logger.info(recommandation2)

        return render_template('profile.html', recommandation_text='You should consider hiring : {}'.format(recommandation2))
        
    return render_template('profile.html')



@app.route('/prof')
def prof():
    return render_template('prof.html')

if __name__ == '__main__':
    app.run(port=8080,debug=True)
    
