#Hotel Booking Status Prediction Using Machine Learning

1. Introduction
In this project, we aimed to predict the status of hotel bookings (whether a booking is
canceled or not canceled) using machine learning models. The dataset provided information on
different attributes like the number of guests, the type of meal plan, the lead time, and more. We
utilized these features to train our models and made predictions on whether bookings would be
canceled based on historical data. The dataset was chosen for its wide variety of features that
influence booking status, making it a rich resource for experimenting with different machine
learning algorithms.
Our approach was inspired by the Kaggle work of Anurag Verma, who achieved an
89.08% accuracy with a Random Forest model. We sought to replicate this result and explore
further ways to enhance predictive performance. The primary challenge we faced was addressing
the class imbalance in the target variable, as most bookings were not canceled. We adopted
various techniques like hyperparameter tuning, cross-validation, and the use of SHAP values
to interpret the model and ensure we had a transparent, accurate prediction.
2. Objective
The objectives of this project were multifaceted. First, we focused on feature refinement
to improve the accuracy of our models. This involved utilizing correlation matrices to identify
redundant or highly correlated features that could lead to overfitting. By removing highly
correlated features, we ensured that our models could generalize better to unseen data,
preventing them from memorizing the training set.
Next, we aimed to compare models across different machine learning techniques, such
as Logistic Regression, Decision Trees, Random Forests, and Neural Networks
(MLPClassifier). Each model was evaluated based on its F1-Score, AUC-ROC, and overall
accuracy to determine which algorithm was best suited for the given dataset. Finally, we sought
to minimize Type I and Type II errors by focusing on improving both precision and recall.
This would ensure our models were both highly accurate and reliable when deployed in real-
world scenarios, especially in business applications like hotel management.
3. Contributions and Responsibilities
Each team member played a critical role in ensuring the success of the project. Sai
Krupesh Goud Algani led the Exploratory Data Analysis (EDA) phase, providing crucial
insights into the data. By analyzing the distribution of each feature, as well as understanding the
target variable's imbalance, Sai developed a solid foundation for the preprocessing steps that
followed. Sai also helped in identifying the most important features for model training, enabling
us to build effective models.
Dhanush Reddy Kuppireddy was instrumental in developing and tuning the machine-
learning models. He explored a range of classifiers, including Logistic Regression, Decision
Trees, and Random Forests, as well as deep learning techniques like MLPClassifier. Dhanush
focused on hyperparameter tuning using GridSearchCV to optimize the models and ensure
they performed at their best. His contributions were key to finding the optimal configurations for
each algorithm.
Venkat Sujal Valeti applied advanced techniques for model interpretability. Using
SHAP (Shapley Additive Explanations), Venkat analyzed the decision-making process of the
models. This was crucial for understanding the importance of various features, such as lead_time
and avg_price_per_room, in predicting booking cancellations. Venkat also helped interpret the
results of Logistic Regression by analyzing its coefficients to shed light on how each feature
contributed to the decision-making process.
Mounisha Etteri played a pivotal role in the preprocessing pipeline, ensuring that all
features were properly scaled and encoded. She also worked on handling missing values in the
dataset, using SimpleImputer to fill in missing categorical data with the most frequent value.
Mounisha’s focus on data cleanliness and consistency laid the groundwork for successful model
training, enabling the models to learn from the data more effectively.
4. Data Pre-Processing
Handling Missing Values:
One of the key steps in preparing the data for machine learning models was handling
missing values. Since the dataset contained some missing entries, especially in categorical
columns, we used the SimpleImputer class from sklearn. impute to fill in missing data. The
strategy chosen was most_frequent, which replaces missing values with the most common value
in each column. This method ensures that no data is lost, which is crucial for maintaining the
integrity of the model training process.
After addressing missing values, we checked the imputation process by verifying that no
column contained missing entries. Additionally, we applied this imputation technique to
categorical columns, such as meal_plan and market_segment_type, as these columns are
essential for model prediction but contain some empty cells. By ensuring that all values were
filled in, we prevented errors during model training.
Feature Transformation (One-Hot Encoding and Standard Scaling):
Once the data was cleaned, the next step was transforming the features to prepare them
for machine learning models. One-Hot Encoding was applied to categorical features like
meal_plan, market_segment_type, and room_type_reserved. This process converts
categorical variables into a format that can be directly fed into machine learning models. For
numerical features, we applied Standard Scaling to ensure they had the same scale, preventing
models from being biased towards features with larger ranges, like lead_time and
avg_price_per_room.
Scaling the numerical features was essential for ensuring that models such as Logistic
Regression and Neural Networks, which are sensitive to the scale of the data, performed
optimally. Standardizing the features helped avoid issues with the model’s convergence during
training and made sure that each feature contributed equally to the model’s learning process.
5. Exploratory Data Analysis (EDA)
Feature Distribution:
In this stage, we examined the distribution of each feature, which provided valuable
insights into the nature of the data. By plotting histograms of the numerical features such as
lead_time, no_of_adults, and avg_price_per_room, we gained a better understanding of the
data’s spread. This allowed us to identify any potential outliers or skewness in the data, which
could affect model performance. Features like lead_time showed a right-skewed distribution,
indicating that most bookings were made well in advance.
The target variable, booking_status, was also explored using a count plot to
understand the balance between the two classes. With a higher number of Not Canceled
bookings, we observed the class imbalance, which led us to explore techniques such as
class_weight='balanced' in our Random Forest model. This step helped us prepare for
addressing the imbalance during model training.
Correlation Analysis:
To improve model performance, we calculated the correlation matrix of the numerical
features. This matrix helped us identify which features were strongly correlated with each other.
For example, we found that no_of_weekend_nights and no_of_week_nights were highly
correlated. We decided to drop one of the highly correlated features to avoid multicollinearity,
which could otherwise reduce the model’s ability to generalize.
We also calculated how each feature correlated with the target variable (i.e.,
booking_status) to determine which features had the most predictive power. This helped us
focus on important variables like lead_time and avg_price_per_room, which had higher
correlations with the target. By removing or transforming features based on this analysis, we
ensured that the models were trained with the most relevant data.
6. Addressing Class Imbalance
Class Distribution:
One of the significant challenges with this dataset was the class imbalance between Not
Canceled and Canceled bookings. The distribution showed that most bookings were not
canceled, with only 11,885 canceled bookings compared to 24,390 non-canceled bookings. This
imbalance can lead to biased model predictions, where the model favors the majority class.
To address this, we used Random Forests with the parameter class_weight='balanced',
which adjusts the weights of the classes inversely proportional to their frequencies. This helps
the model pay more attention to the minority class, thus improving the prediction for canceled
bookings. We also considered SMOTE as a future improvement to generate synthetic samples of
the minority class and further balance the dataset.
7. Model Development
Logistic Regression:
Logistic Regression served as our baseline model. It is a linear model that works well for
binary classification tasks. We trained the model using the preprocessed features and evaluated
its performance. While the model performed well with an accuracy of 80%, it had a relatively
low recall (0.60) for predicting canceled bookings. This suggests that the model was
conservative and missed some of the positive instances.
Decision Tree:
The Decision Tree classifier was chosen for its interpretability and ability to handle both
categorical and numerical features. After training, it achieved an accuracy of 85%, which was a
significant improvement over Logistic Regression. The Decision Tree model was also evaluated
based on precision and recall, which showed a balanced performance across both classes. This
model is easy to interpret and can be visualized, making it valuable for understanding feature
importance.
Random Forest:
The Random Forest model, consisting of multiple decision trees, demonstrated the best
performance among the models tested. By using GridSearchCV for hyperparameter tuning, we
optimized the number of estimators and the tree depth to achieve an accuracy of 88% and an
AUC-ROC of 0.87. This model outperformed others in both accuracy and recall, especially in
predicting canceled bookings.
Neural Networks (MLPClassifier):
The MLPClassifier (Multi-layer Perceptron) was trained to capture complex, non-linear
relationships in the data. The model was more computationally expensive but showed promising
results with an F1-Score of 0.77 and an AUC-ROC of 0.83. While it was not the best performer,
it provided valuable insights into the potential of neural networks for future improvements.
8. Model Evaluation
Each model was evaluated using several performance metrics, including precision,
recall, F1-Score, and AUC-ROC. The Random Forest model emerged as the top performer,
achieving an F1-Score of 0.82 and an AUC-ROC of 0.87. This indicates that Random Forests
are highly effective for this dataset, balancing both recall and precision.
Logistic Regression and Decision Trees also performed well, but they did not reach the
performance levels of the Random Forest model. The MLPClassifier showed potential but was
slower to train and required more computational resources.
9. SHAP Values for Model Interpretability
Using SHAP values, we gained deep insights into how the Random Forest model
arrived at its decisions. Features such as lead_time and avg_price_per_room were identified as
the most important factors influencing booking cancellations. The SHAP summary plot
provided a visual representation of how these features contributed to the model's output.
Interpreting the model with SHAP is crucial for understanding the business implications.
For instance, lead_time is highly influential, meaning that bookings made well in advance are
more likely to be canceled. This insight can be used to optimize hotel management strategies.
10. Conclusion and Future Work
Key Contributions:
• We improved model performance by removing highly correlated features and
addressing class imbalance.
• We implemented SHAP values to interpret the model’s decisions, providing transparency
into which features had the most impact.
• Hyperparameter tuning using GridSearchCV significantly enhanced the model’s
performance.
Future Work:
• SMOTE could be applied to generate synthetic samples of canceled bookings, further
improving model performance in the minority class.
• We could explore other advanced models such as XGBoost or LightGBM, which have
shown strong performance in similar tasks.
References:
• SHAP Documentation: https://shap.readthedocs.io/en/latest/
• Scikit-learn Documentation: https://scikit-learn.org/stable/
• Anurag Verma’s Kaggle Work: https://www.kaggle.com/code/anurag629/hotel-
reservations-dataset-best-machine-learning/notebook
Code, Dataset and Presentation Links:
• Dataset Link: https://www.kaggle.com/code/anurag629/hotel-reservations-dataset-best-
machine-learning/input
• Python Notebook Code: https://drive.google.com/file/d/1li0c0rtj7sz2zA0InKhjESIrHd-
MjrOP/view?usp=sharing
• Presentation Link:
https://drive.google.com/file/d/1on3zv6rEva6rxLfmFbuQg7wPTSzrJWgw/view?usp=sha
ring
