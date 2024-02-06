# Titanic Survival Prediction Project

## Project Overview
- This project aims to predict which passengers survived the Titanic shipwreck using machine learning. The goal is to build a predictive model that answers the question: “What type of passengers were more likely to survive?” using the provided data.

## Setting Up the Environment
- To replicate the analysis and run the code in your local environment, you will need to install the necessary Python packages. A requirements.txt file is provided for easy setup.
- Using Conda
- If you're using Conda, you can create an environment with all the required packages using the following command:

```bash
conda create --name <env_name> --file requirements.txt
```

- Replace <env_name> with your desired environment name.
- After creating the environment, activate it using:

```bash
conda activate <env_name>
```

## Exploratory Data Analysis (EDA)
- Survival Distribution: An analysis of the balance between the two classes, survivors and non-survivors, was conducted, and the findings were visualized using a bar graph plot. Among the passengers, 342 belong to the class "survivors," while 549 belong to the class "non-survivors." It is worth noting that there is a slight imbalance between these two classes, which is a common occurrence in many datasets. However, it is reassuring to see that the model performed admirably despite this slight class imbalance, effectively handling the mismatch to yield promising results.
- Gender vs. Survival: A cross-tabulation analysis conducted between the target classes "survived" and "non-survived" alongside gender reveals a significant trend: females exhibit a notably higher likelihood of survival compared to males. This finding emphasizes the gender-based disparities in survival rates during the given event.
- Age Distribution: A comprehensive analysis to explore the impact of age and gender on the likelihood of survival. Visualizing this relationship through a histogram plot, some intriguing patterns were uncovered:
- Males Aged 20 to 35: The data strongly suggests that males in the age range of approximately 20 to 35 had a higher probability of surviving the disaster. This could be attributed to their physical strength and resilience during the crisis.
- Females Aged 15 to 40: Conversely, females between the ages of 15 and 40 exhibited a significantly greater likelihood of survival. This finding aligns with the historical practice of prioritizing women during rescue operations, reflecting societal values of chivalry and protection.
- Infants: Notably, infants, regardless of gender, showed a slightly increased chance of survival. This observation underscores the universal instinct to prioritize the safety of the youngest passengers, who are the most vulnerable in such situations.
- In conclusion, the analysis of age and gender dynamics within the dataset provides valuable insights into the factors influencing survival rates. These findings shed light on the complex interplay between age, gender, and survival, which can inform future research and emergency response strategies.
- Class, Embarkation, Gender: A comprehensive analysis of survival rates by passengers' class, embarkation points, and gender, uncovered noteworthy correlations and trends:
- Survival Analysis by Gender and Embarkation Points:
- Females: Women exhibited a significantly higher rate of survival compared to men across all embarkation points. However, the impact of embarkation points on female survival rates varied. Females embarking at Port C and Port Q had a notably higher chance of survival. Conversely, female passengers embarking at Port S had a lower probability of survival.
- Males: For males, the survival probability was influenced by their embarkation point as well. Men embarking at Port C had a higher survival probability. However, men boarding at Port S or Port Q had a lower chance of survival.
- Survival Analysis by Passenger Class: Passenger class was strongly correlated with survival. Passengers in 1st class had a significantly higher rate of survival. Conversely, passengers in 3rd class (Class 3) had a lower rate of survival.
- These findings underscore the complex interplay of gender, embarkation points, and passenger class in determining survival rates during the event. It is clear that multiple factors contributed to the passengers' likelihood of surviving, making this analysis valuable in understanding the dynamics of survival aboard the ship.

## Data Preprocessing
### Missing Data Handling
- Embarked: Filled missing values with the most frequent embarkation point.
- Age: Used RandomForestRegressor to predict and fill in missing ages.
- Fare: Filled missing values with the mean fare.

### Feature Engineering
- Converted 'Age' and 'Fare' to integer data types.
- Mapped categorical variables ('Sex', 'Embarked') to numerical values.
- Created new features: 'Age_Group', 'Family_Size', 'Fare_Category', and 'Age_Pclass'.

### Hypotheses from Visualizations
- Higher survival rates observed in females and younger passengers.
- Passengers in higher classes had better survival chances.

## Model Building
Implemented multiple models:
- RandomForestClassifier
- KNeighborsClassifier
- DecisionTreeClassifier
- LogisticRegression

## Model Evaluation
- Initial evaluation based on training accuracy.
- Cross-validation used to assess generalization capability.
- Classification report for detailed evaluation metrics (precision, recall, F1-score).

## Selecting the Best Model
- Compared cross-validated scores to select the best performing model.
- RandomForestClassifier emerged as the best model based on accuracy.

## Hyperparameter Tuning
- Conducted GridSearchCV on RandomForestClassifier to find optimal parameters.

## Final Model
- Re-trained RandomForestClassifier with the best parameters from hyperparameter tuning.
- Final model accuracy assessed using cross-validation.

## Feature Importance Analysis
- Analyzed and visualized the importance of different features in the final model. A comprehensive analysis to assess the importance of various features in the final model. The visualization highlights the significance of these features in influencing the model's predictions. The key influencing features, as depicted in the visual, are:
- Sex: Gender emerged as the most influential feature in our model, indicating that it strongly contributes to predicting survival outcomes.
- Age_Pclass: The combination of age and passenger class (Age_Pclass) is another crucial factor in predicting survival. It underscores the interaction between age and socio-economic status.
- Fare: Fare paid by passengers is a significant determinant of survival, likely reflecting the passengers' class and associated privileges.
- Age: Age alone also plays a notable role in influencing survival predictions, with younger and older passengers having varying chances of survival.
- Pclass: Passenger class (Pclass) independently contributes to the model's predictions, indicating its importance in assessing survival probabilities.
- Family_Size: The size of the passenger's family onboard is an additional relevant factor affecting survival, as it could impact the ability to find and secure safety during the event.
- This visualization provides valuable insights into the relative importance of different features in the final model, helping us better understand the key drivers of survival predictions.

## Generating Predictions
- Predictions made on the test dataset using the final model.
- Generated a submission file (`titanic_predictions.csv`) for evaluation.

## Evaluation and Conclusion
- The final model demonstrated high accuracy and a good balance between precision and recall.
- The classification report indicated the model's robustness in predicting both survivor and non-survivor classes.
