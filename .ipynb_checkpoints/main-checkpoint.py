#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor


# In[3]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# ## Reading Data

# In[4]:


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# In[5]:


train_data.head(5)


# ## Train Data

# ### Exploring and Analysing Train Data

# In[6]:


train_data.columns.values


# In[7]:


train_data.shape


# In[8]:


train_data.info()


# In[9]:


train_data.describe()


# ### Viusally Analysing Train Data

# In[10]:


train_data["Survived"].value_counts().plot(kind="bar", color=["salmon", "lightblue"])

plt.title('Survival Distribution in Training Data', fontsize=16)
plt.xlabel('Survived', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# In[11]:


pd.crosstab(train_data.Survived, train_data.Sex)


# In[12]:


sns.set(style="white")

survived_label = 'Survived'
not_survived_label = 'Not Survived'

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

male = train_data[train_data['Sex'] == 'male']
female = train_data[train_data['Sex'] == 'female']

ax = sns.histplot(male[male['Survived'] == 1].Age.dropna(), bins=18, label=survived_label, 
                  ax=axes[0], kde=False)
ax = sns.histplot(male[male['Survived'] == 0].Age.dropna(), bins=40, label=not_survived_label, 
                  ax=axes[0], kde=False)
ax.legend()
ax.set_title('Male')

ax = sns.histplot(female[female['Survived'] == 1].Age.dropna(), bins=18, label=survived_label, 
                  ax=axes[1], kde=False)
ax = sns.histplot(female[female['Survived'] == 0].Age.dropna(), bins=40, label=not_survived_label, 
                  ax=axes[1], kde=False)
ax.legend()
ax.set_title('Female')

for ax in axes:
    ax.set_xlabel('Age')

plt.tight_layout()
plt.show()


# In[13]:


sns.set(style="white")

g = sns.catplot(x='Pclass', y='Survived', hue='Embarked', col='Sex', kind='bar', data=train_data, palette='YlGnBu_d', errorbar=None)

g.fig.suptitle('Survival by Passenger\'s Class, Embarking Point, and Gender', fontsize=16)
g.set_axis_labels('Passenger Class', 'Survival Rate')

g.fig.subplots_adjust(top=0.85)

plt.show()


# In[14]:


sns.set(style="white")

counts = train_data.groupby(['Pclass', 'Survived']).size().unstack()

ax = counts.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], figsize=(10, 6))

plt.title('Survival by Passenger\'s Class', fontsize=16)
plt.xlabel('Passenger Class', fontsize=14)
plt.ylabel('Count', fontsize=14)

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height}', (x + width/2, y + height/2), ha='center', va='center', fontsize=12, color='black')

plt.legend(title='Survived', labels=['Not Survived', 'Survived'])

sns.despine()

plt.show()


# ## Pre-processing Train Data

# ### Dealing with missing values in train data

# In[15]:


train_df = train_data.copy()


# In[16]:


train_df.isnull().sum()


# In[17]:


train_df['Embarked'].describe()


# In[18]:


train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)


# ### Dropping Passenger ID and Cabin Column from train data

# In[19]:


train_df = train_df.drop('Cabin', axis=1)


# In[20]:


train_df = train_df.drop('PassengerId', axis=1)


# ### Filling Age Column missing values in train data

# In[21]:


features = ['Pclass', 'SibSp', 'Parch', 'Fare']

known_age = train_df[train_df['Age'].notna()]
unknown_age = train_df[train_df['Age'].isna()]

model = RandomForestRegressor()
model.fit(known_age[features], known_age['Age'])

predicted_ages = model.predict(unknown_age[features])

train_df.loc[train_df['Age'].isna(), 'Age'] = predicted_ages


# ### Converting dtype float to int in train data

# In[22]:


train_df['Age'] = train_df['Age'].astype(int)
train_df['Fare'] = train_df['Fare'].astype(int)


# In[23]:


train_df.isnull().sum()


# ### Dealing with dtype object in train data

# In[24]:


embarked_mapping = {'C': 0, 'S': 1, 'Q': 2}
train_df['Embarked'] = train_df['Embarked'].map(embarked_mapping)

sex_mapping = {'male': 0, 'female': 1}
train_df['Sex'] = train_df['Sex'].map(sex_mapping)


# In[25]:


train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[26]:


train_df['Title'].unique()


# In[27]:


title_mapping = {
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Miss': 'Miss',
    'Master': 'Master',
    'Don': 'Rare',
    'Rev': 'Rare',
    'Dr': 'Rare',
    'Mme': 'Mrs',
    'Ms': 'Miss',
    'Major': 'Rare',
    'Lady': 'Rare',
    'Sir': 'Rare',
    'Mlle': 'Miss',
    'Col': 'Rare',
    'Capt': 'Rare',
    'the Countess': 'Rare',
    'Jonkheer': 'Rare'
}
train_df['Title'] = train_df['Title'].map(title_mapping)


# In[28]:


title_mapping_numerical = {
    'Mr': 1,
    'Mrs': 2,
    'Miss': 3,
    'Master': 4,
    'Rare': 5
}
train_df['Title'] = train_df['Title'].map(title_mapping_numerical)


# ### Dropping dtype object in train data

# In[29]:


train_df = train_df.drop('Name', axis=1)


# In[30]:


train_df = train_df.drop('Ticket', axis=1)


# In[31]:


train_df.info()


# ### New Featrures in Train Data

# In[32]:


train_df['Age_Group'] = pd.cut(train_df['Age'], bins=[0, 12, 18, 35, 60, np.inf], 
                               labels=[0, 1, 2, 3, 4], right=False)


# In[33]:


train_df['Family_Size'] = train_df['SibSp'] + train_df['Parch'] + 1


# In[34]:


train_df['Fare_Category'] = pd.qcut(train_df['Fare'], q=4, labels=False)


# In[35]:


train_df['Age_Pclass'] = train_df['Age'] * train_df['Pclass']


# In[36]:


train_df.info()


# In[37]:


train_df.head()


# ### Test Data

# In[38]:


test_df = test_data.copy()


# In[39]:


test_df.head(5)


# In[40]:


test_df.shape


# In[41]:


test_df.info()


# In[42]:


test_df.isnull().sum()


# ## Pre-processing Test Data

# ### Dealing with missing values

# In[43]:


test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)


# ### Dropping PassengerID and Cabin in test data

# In[44]:


test_df = test_df.drop('PassengerId', axis=1)


# In[45]:


test_df = test_df.drop('Cabin', axis=1)


# ### Filling Age Column missing values in test data

# In[46]:


features = ['Pclass', 'SibSp', 'Parch', 'Fare']

known_age_test = test_df[test_df['Age'].notna()]
unknown_age_test = test_df[test_df['Age'].isna()]

model = RandomForestRegressor()
model.fit(known_age_test[features], known_age_test['Age'])

predicted_ages = model.predict(unknown_age_test[features])

test_df.loc[test_df['Age'].isna(), 'Age'] = predicted_ages


# ### Converting dtype float to int in test data

# In[47]:


test_df['Age'] = test_df['Age'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)


# ### Dealing with dtype object in test data

# In[48]:


embarked_mapping = {'C': 0, 'S': 1, 'Q': 2}
test_df['Embarked'] = test_df['Embarked'].map(embarked_mapping)


# In[49]:


sex_mapping = {'male': 0, 'female': 1}
test_df['Sex'] = test_df['Sex'].map(sex_mapping)


# In[50]:


test_df['Title'] = test_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[51]:


test_df['Title'].unique()


# In[52]:


title_mapping = {
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Miss': 'Miss',
    'Master': 'Master',  
    'Ms': 'Miss',
    'Col': 'Rare',
    'Rev': 'Rare',
    'Dr': 'Rare',
    'Dona': 'Rare'
}
test_df['Title'] = test_df['Title'].map(title_mapping)


# In[53]:


title_mapping_numerical = {
    'Mr': 1,
    'Mrs': 2,
    'Miss': 3,
    'Master': 4,
    'Rare': 5
}
test_df['Title'] = test_df['Title'].map(title_mapping_numerical)


# ### Dropping dtype object in test data

# In[54]:


test_df = test_df.drop('Name', axis=1)


# In[55]:


test_df = test_df.drop('Ticket', axis=1)


# ### New Featrures in test Data

# In[56]:


test_df['Age_Group'] = pd.cut(test_df['Age'], bins=[0, 12, 18, 35, 60, np.inf], 
                               labels=[0, 1, 2, 3, 4], right=False)


# In[57]:


test_df['Family_Size'] = test_df['SibSp'] + test_df['Parch'] + 1


# In[58]:


test_df['Fare_Category'] = pd.qcut(test_df['Fare'], q=4, labels=False)


# In[59]:


test_df['Age_Pclass'] = test_df['Age'] * test_df['Pclass']


# In[60]:


test_df.info()


# In[61]:


test_df.head()


# In[63]:


train_df.info()


# In[62]:


train_df.head()


# In[64]:


len(train_df.columns), len(test_df.columns)


# ## Model Building

# ### Train Test Split

# In[65]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.copy()


# In[66]:


X_train.shape, Y_train.shape


# ### Training and evaluating

# In[67]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, Y_train)

rf_train_preds = rf_model.predict(X_train)

rf_accuracy = rf_model.score(X_train, Y_train)
print("RandomForest Accuracy (Training):", rf_accuracy)


# In[68]:


knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, Y_train)

knn_train_preds = knn_model.predict(X_train)

knn_accuracy = knn_model.score(X_train, Y_train)
print("KNN Accuracy (Training):", knn_accuracy)


# In[69]:


dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, Y_train)

dt_train_preds = dt_model.predict(X_train)

dt_accuracy = dt_model.score(X_train, Y_train)
print("Decision Tree Accuracy (Training):", dt_accuracy)


# In[70]:


lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, Y_train)

lr_train_preds = lr_model.predict(X_train)

lr_accuracy = lr_model.score(X_train, Y_train)
print("Logistic Regression Accuracy (Training):", lr_accuracy)


# ### Best Model

# In[71]:


models = {
    "RandomForest": RandomForestClassifier(),
    "KNeighbors": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

model_scores = {name: cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy').mean() 
                for name, model in models.items()}

for name, score in model_scores.items():
    print(f"{name}: {score}")


# ### Hyper parameter tuning

# In[72]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

grid_search.fit(X_train, Y_train)

print("Best Parameters:", grid_search.best_params_)


# ### Final Model

# In[73]:


final_model = grid_search.best_estimator_
final_scores = cross_val_score(final_model, X_train, Y_train, cv=5, scoring='accuracy')

print("Final Model Accuracy:", final_scores.mean())


# ### Predictions on Test set

# In[74]:


test_predictions = final_model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_predictions
})
submission.to_csv('titanic_predictions.csv', index=False)


# In[ ]:




