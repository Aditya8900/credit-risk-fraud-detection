
# coding: utf-8

# #                                              Introduction
# 
# This is a project demonstrating various classification techniques on imbalanced dataset, which is quite different from regular classfication problems. The goal is to perform a binary classification of Class0 (Not fraud) and Class1 (fraud). The approach in this notebook will be :
# 
# 1. Randomly split the dataset into train, validation, and test set.
# 2. Do basic EDA and feature engineering.
# 3. Predict and evaluate with validation set.
# 4. Resample the dataset. 
# 5. Train on resampled train set then predict and evaluate with validation set.
# 6. Try other different models.
# 7. Compare the difference between the predictions and choose the best model.
# 8. Find the optimised threshold of the chosen model.
# 9. Predict on test set to report final result.
# 
# Some points worth noting are :
# 
# 1. The features are mostly PC components(28 out of 30) which we can neither name nor engineer with. 
# 2. Since the dataset is highly imbalanced, using accuracy as evaluation metric is not a clever idea, the performance metrics should be Confusion Matrix and PR(precision-recall) curve, or maybe Area under PR curve if a single-number metric is needed.
# 3. We should not predict hard labels(0 or 1) at first place but probabilities instead, since the default 0.5 cutoff for labeling might not be suitable for imbalanced dataset.
# 4. As a brain refresher, Precision evaluates how many selected items are relevant, while Recall evaluates how many relevant items are selected.
# 5. The train set is used to train model and model tuning, the validation set is used to evaluate on PR curve and choosing threshold, while the test set is only used for final prediction.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("C:/Users/Oliver/Credit Card Fraud detection/creditcard.csv")


# In[3]:


data.head()


# In[4]:


count = pd.value_counts(data['Class'], sort = True).sort_index()
count.plot(kind = 'bar')
count
#The two classes are really imbalanced!


# In[5]:


data.hist(figsize = (20,20))
plt.show()


# In[6]:


#correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat,vmax = .8,square = True)
plt.show()


# #                                                 Data Split
# 
# We will first split the data randomly into train and test sets to do exploratory analysis, while preserving the ratio of classes in each of them.

# In[7]:


from sklearn.model_selection import train_test_split
y = data['Class']
x = data.drop('Class',axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,stratify=y,random_state=81825)
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size=0.333,stratify=y_train,random_state=8125)
train = pd.concat([x_train,y_train],axis = 1)
validation = pd.concat([x_validate,y_validate],axis=1)
test = pd.concat([x_test,y_test],axis =1)
print("Percentage of fraud detection in train is:",round(train.Class.mean(),4))
print("Percentage of fraud detection in test is:",round(test.Class.mean(),4))


# #                                    EDA and Feature Engeneering

# In[8]:


sns.boxplot(x="Class",y = "Time",data=train)
train.groupby("Class").Time.describe()


# Time itself doesn't seem to determine class well. We will feature a new variable which is the time difference between transaction and the precede transaction. In order to that, we will combine train and test first, do the feature engineering, then split the dataset again.

# In[9]:


data = pd.concat([train,validation,test],axis=0)
#The column names are auto reordered, lets turn it back to what it should be
data = data[list(train.columns.values)]
data.sort_index(inplace = True)
#create the time difference feature
data['Time_Difference'] = data['Time']-data['Time'].shift()
#remove the NA row caused by lag
data.dropna(axis=0,inplace=True)

#split the data set again
y = data['Class']
x = data.drop('Class',axis = 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,stratify=y,random_state=8125)
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size=0.333,stratify=y_train,random_state=8125)
train = pd.concat([x_train,y_train],axis=1)


# In[10]:


train.groupby('Class').Time_Difference.describe()


# In[11]:


sns.boxplot(x='Class',y="Time_Difference",data = train)


# Most of the time difference are similiar, however, some large difference might indicates fraud, as fraud often happens at time that has few transactions.

# In[12]:


sns.catplot(x='Class',y='Amount',data=train,kind='bar',ci=None,aspect=.6)


# In[13]:


sns.boxplot(x="Class",y="Amount",data =train)
train.groupby('Class').Amount.describe()


# Even though average amount of non-fraud is greater than fraud, transaction of large amount however are mostly fraud, which follows our instinct about fraud. We will create a binary factor indicating transaction with large amount, with a threshold of 2000.

# In[14]:


pd.options.mode.chained_assignment = None
def amount_threshold(x):
    if x > 2000: return 1
    else: return 0
x_train['large_Amount'] = x_train['Amount'].apply(amount_threshold)
x_validate['large_Amount'] = x_validate['Amount'].apply(amount_threshold)
x_test['large_Amount'] = x_test['Amount'].apply(amount_threshold)


# #                                    Predict and Evaluate - Original set
# We will first use logistic regression, with parameter tuned by stratified cross validation, to predict the probability of labels. We use area under Precision-Recall curve, aka average precision score to evaluate the performance. The reason to choose PR AUC over ROC AUC is that PR does not account for true negatives, therefore more suitable for imbalanced classification.

# In[15]:


from sklearn.linear_model import LogisticRegressionCV
logCV = LogisticRegressionCV(Cs=[0.01,0.1,1,10,100],scoring='average_precision',verbose = 0,max_iter=200,random_state=5001)
logCV.fit(x_train,y_train)
print("The best parameter C is",logCV.C_[0])


# We will use the fitted model to predict our validation (test) set. The output would be a set of probabilities, which we will analyze it with respect to PR-curve.

# In[18]:


from sklearn.metrics import average_precision_score,precision_recall_curve
pred_prob = logCV.predict_proba(x_validate)
y_score = pred_prob[:,1]
average_precision = average_precision_score(y_validate,y_score)
original_precision,original_recall,orignal_thresholds = precision_recall_curve(y_validate,y_score)
plt.step(original_recall,original_precision,color='red',alpha=0.5,linewidth=1.5,label='original logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize='large')


# From the PR curve above we can see how our model performed, and we can choose optimised cutoff to yield the precision and recall rate we want.

# #                                               Data Resampling
# 
# It's time to do some resampling to our train set! We will oversample our minority class with SMOTE(Synthetic Minority Over-sampling Technique), which generates new samples similiar but not typically identical of minority classes, since the algorithm takes nearby instances of same class into account. One important thing to mention is that we should NEVER resample our validation or test set, since it might lead to an over-optimistic performance that is untrustable.

# In[21]:


#conda install -c glemaitre imbalanced-learn
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=4001,kind = 'regular')
x_res,y_res = sm.fit_sample(x_train,y_train)
pd.value_counts(y_res)


# #                                        Predict and Evaluate - Resampled set
# Next we will use logistic regression with the same parameter C=10 to test the performance on our same validation(test) set.

# In[27]:


from sklearn.linear_model import LogisticRegression
log_res = LogisticRegression(C=10,verbose=0,random_state=5001)
log_res.fit(x_res,y_res)


# In[28]:


pred_prob = log_res.predict_proba(x_validate)
resampled_score = pred_prob[:,1]
average_precision = average_precision_score(y_validate,y_score)
resampled_precision,resampled_recall,resampled_thresholds = precision_recall_curve(y_validate,resampled_score)
plt.step(original_recall,original_precision,color='red',alpha=0.5,linewidth=1.5,label='original logistic')
plt.step(resampled_recall,resampled_precision,color='green',alpha=0.5,linewidth=1.5,label='Resampled to logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize='large')


# We can see from the right of the plot that resampling does bring improvement to our model. Similiar precision rate now gives us a higher recall rate! Next we will approach with different models to see if we can make further improvement.

# #                                 Random Forest - Resampled set
# 
# We will now train the well-known random forest algorithm on the resampled train set. Random forest usually have satisfying performance on classification problem, even without complicated model tuning

# In[30]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=4,n_estimators=100,bootstrap=True,random_state=7017)
rf.fit(x_res,y_res)


# In[31]:


pred_prob = rf.predict_proba(x_validate)
rf_score = pred_prob[:,1]
average_precision = average_precision_score(y_validate,y_score)
rf_precision,rf_recall,rf_thresholds = precision_recall_curve(y_validate,rf_score)
plt.step(original_recall,original_precision,color="red",alpha=0.5,linewidth=1.5,label='Original logistic')
plt.step(resampled_recall,resampled_precision,color='green',alpha=0.5,linewidth=1.5,label='Resampled logistic')
plt.step(rf_recall,rf_precision,color='blue',alpha=0.5,linewidth=1.5,label='Resampled random forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Random Forest Precision-Recall curve')
plt.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize='large')


# Comparing to logistic regression, random forest seems to be more suitable for our data, with another big improvement on validation set.

# #                                Cluster Before Prediction - Original set
# 
# We will aproach the problem in a different but intuitive way. The method is called cluster before prediction, with three consecutive steps. First separate the majority class into clusters, say 20 clusters. Then we will combine each segment with all minority samples to train 20 models. Finally, we predict the test set with these models, averaging all of them to get a final prediction. We will train random forest classifiers since it generates better result comparing to logistic regression.

# In[33]:


from sklearn.cluster import KMeans
x_train_minority,x_train_majority = x_train[y_train==1],x_train[y_train==0]
kmeans = KMeans(n_clusters=20,random_state=5001).fit(x_train_majority)


# In[36]:


prediction = [0]*len(x_validate)
for i in range (0,20):
    majority_segment = x_train_majority[kmeans.labels_==i]
    x_train_segment = pd.concat([majority_segment,x_train_minority])
    y_train_segment = [0]*len(majority_segment)+[1]*len(x_train_minority)
    rf = RandomForestClassifier(max_depth=4,n_estimators=100,bootstrap=True,random_state=7001)
    rf.fit(x_train_segment,y_train_segment)
    pred_prob = rf.predict_proba(x_validate)
    prediction = np.add(prediction,pred_prob[:,1])
prediction=prediction/20


# In[37]:


y_score = prediction
average_precision = average_precision_score(y_validate, y_score)
cls_precision, cls_recall, cls_thresholds = precision_recall_curve(y_validate, y_score)
plt.step(original_recall, original_precision, color='red', alpha=0.5,linewidth=1.5,label='Original logistic')
plt.step(resampled_recall, resampled_precision, color='green', alpha=0.5,linewidth=1.5,label='Resampled logistic')
plt.step(rf_recall, rf_precision, color='blue', alpha=0.5,linewidth=1.5,label='Resampled random forest')
plt.step(cls_recall, cls_precision, color='orange', alpha=0.5,linewidth=1.5,label='Clustered random forest')
plt.plot([0.81],[0.84],color="black",marker='o', markersize=5)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize = 'large')


# The algorithm yields similiar result to training random forest on resampled set. However, the left part of the graph indicates big improvement on keeping high precision rate when recall rate is about 0.2 - 0.4. We will choose resampled random forest as our final model, since it takes less time to train than clustered random forest. The black point indicates the precision and recall rate we want, and we are going to find the optimised cut for that.

# #                                    Threshold and Confusion Matrix

# In[38]:


error = []
for recall, precision in zip(rf_recall, rf_precision):
    err = (recall-0.81)**2 + (precision-0.84)**2
    error.append(err)
error = np.asarray(error)
min_index = np.where(error==min(error))
opt_threshold = rf_thresholds[min_index]
"The optimised threshold is: {0:0.3f}".format(opt_threshold[0])


# In[39]:


from sklearn.metrics import confusion_matrix
y_pred = [1 if x > opt_threshold else 0 for x in rf_score]
conf = confusion_matrix(y_validate,y_pred)
conf = pd.DataFrame(conf, range(2),range(2))
sns.heatmap(conf, annot=True, fmt='g', annot_kws={"size": 16})
plt.title('Confusion matrix for validation set')


# #                                            Predict on Test set

# In[41]:


pred_prob = rf.predict_proba(x_test)
y_score = pred_prob[:,1]
y_pred = [1 if x > opt_threshold else 0 for x in y_score]
conf = confusion_matrix(y_test,y_pred)
conf = pd.DataFrame(conf, range(2),range(2))
sns.heatmap(conf, annot=True, fmt='g', annot_kws={"size": 16})
plt.title('Confusion matrix for prediction on test set')


# Using our chosen model and threshold on test set, we yield similiar result comparing to the validation set.

# #                                                 Conclusion
# 
# For this project we investigated whether resampling technique can bring improvement to model of imbalanced classification. From what I see, I believe that the answer is yes, however there are something I would like to mention additional to the analyis.
# 
# 1. Tuning parameters of resampled model with cross validation can possibly improve our model, however, we should not do cv after oversampling or else the oversampled validation slice of each fold won't be a good estimate of test set. The proper steps to do cv for each fold is to first select a validation set and leave it unchanged, oversample the remain data, and finally test the trained model on unchanged validation set. 
# 2. There are other oversample techniques such as ROSE that we didn't try but can possibly improve our model. Undersampling techniques might be a suitable choice too. 
# 3. In real world situation, the optimised point depends. Sometimes the cost of allowing more false positive is a lot cheaper than the loss of not identifying the positive ones. In such case, we can lower our precision to gain more recall.
