import pandas as pd
import numpy as np
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from flask import Flask, render_template, request

app = Flask(__name__)

# --------------------------------------------------------------------------------------------------------------------------
#                                             DATA   CLEANING

# # Function to convert 'Household Income' to value ranges
# def convert_income_to_range(income_str):
#     # Extract numbers from the string
#     numbers = [int(s) for s in income_str.split() if s.isdigit()]
#     # Return the range if two numbers are found, else return None
#     return (min(numbers), max(numbers)) if len(numbers) == 2 else None

# # Apply the conversion function to the 'Household Income' column
# df['Household Income'] = df['Household Income'].apply(convert_income_to_range)

# # Drop rows where 'Household Income' is None
# df = df.dropna(subset=['Household Income'])


# # Function to convert 'Percentage of investment' to value ranges
# def convert_percentage_to_numeric_range(percentage_str):
#     if isinstance(percentage_str, tuple):
#         return percentage_str
#     if percentage_str == 'Don\'t Want to Reveal':
#         return [None, None]
#     elif 'Above' in percentage_str:
#         # Remove '%' before conversion
#         num = int(percentage_str.split()[1].replace('%', ''))
#         return [num, 100]
#     elif 'Upon' in percentage_str:
#         # Remove '%' before conversion
#         num = int(percentage_str.split()[1].replace('%', ''))
#         return [0, num]
#     else:
#         # Extract numbers and remove '%' before conversion
#         numbers = [int(s.replace('%', '')) for s in percentage_str.split() if s.replace('%', '').isdigit()]
#         return [min(numbers), max(numbers)] if numbers else [None, None]

# # Apply the conversion function to the 'Percentage of Investment' column
# df['Percentage of Investment'] = df['Percentage of Investment'].apply(convert_percentage_to_numeric_range)

# # ---------------------------------
# # Function to convert 'investment experience' to value ranges
# def convert_experience_to_range(experience):
#     if experience.startswith('Less Than'):
#         num = int(experience.split(' ')[2])
#         return [0, num]
#     elif experience.startswith('Above'):
#         num = int(experience.split(' ')[1])
#         return [num, float('inf')]
#     elif 'to' in experience:
#         parts = experience.split(' ')
#         start, end = int(parts[0]), int(parts[3])
#         return [start, end]
#     else:
#         return [None, None]

# # Apply the conversion function to the 'Investment Experience' column
# df['Investment Experience'] = df['Investment Experience'].apply(convert_experience_to_range)

# # --------------------------------------
# # Function to convert 'Return Earned' to value ranges
# def convert_return_to_range(return_earned):
#     if return_earned == 'Negative Return':
#         return [-float('inf'), -float('inf')]
#     elif return_earned.startswith('More than'):
#         num = int(return_earned.split(' ')[2])
#         return [num, float('inf')]
#     elif 'to' in return_earned:
#         parts = return_earned.split(' ')
#         start, end = int(parts[0]), int(parts[2])
#         return [start, end]
#     else:
#         return [None, None]

# # Apply the conversion function to the 'Return Earned' column
# df['Return Earned'] = df['Return Earned'].apply(convert_return_to_range)

# # Display the first few converted ranges to verify the conversion
# pd.set_option('display.max_columns', None)
# print(df.head())

# -----------------------------------------------------------------------------------------------------------------------------------


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Load the dataset
        df = pd.read_csv('Copy of Sample Data for shortlisting.xlsx - Sheet1.csv')
        df.drop(['S. No.'], axis=1, inplace=True) #removing serial no. column

        # -------------------------------------------------------------------------------------------------------------------

        # Encode categorical variables
        le = LabelEncoder()
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = le.fit_transform(df[column])

        # Define the target variable and features
        X = df.drop(['Risk Level'], axis=1)
        y = df['Risk Level']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')

        # Get feature importances
        feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        print('Feature Importances:')
        print(feature_importances)
        
        # -----------------------------------------------------------------------------------------------------------------

        print("BEST PARAMS...")
        # Define the parameter grid to search
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        # Initialize the GridSearchCV
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)

        # Perform the grid search
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = grid_search.best_params_
        print("best params:", best_params)

        # Select the most important factors
        selected_factors = feature_importances.head(5).index

        # Create a subset of the dataset with the selected factors
        subset_df = df[selected_factors]

        # Calculate the correlation matrix
        correlation_matrix = subset_df.corr()

        # ------------------------------------------------------------------------------------------------------------------

        # Define the features and target variable
        print(" x: ",selected_factors)
        X = df[selected_factors]
        y = df['Risk Level']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the Random Forest Classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        # # Train the model
        # rf_classifier.fit(X_train, y_train)

        # Train the model
        rf_classifier.fit(X_train, y_train)

        print("Model Trained and Ready for Testing...")

        # Assign the trained model to 'model'
        model = rf_classifier

        # Get user input from the form
        knowledge_share_market = int(request.form['knowledge_share_market'])
        knowledge_investment_prod = int(request.form['knowledge_investment_prod'])
        household_income = int(request.form['household_income'])
        investor_influencer = int(request.form['investor_influencer'])
        investment_percentage = int(request.form['investment_percentage'])

        # Predict risk levels using the trained model
        test_data = [knowledge_share_market, knowledge_investment_prod, household_income, investor_influencer, investment_percentage]
        predicted_risk_levels = rf_classifier.predict([test_data])

        if predicted_risk_levels == [2]:
            risk_level = "HIGH RISK INVOLVED"
        elif predicted_risk_levels == [1]:
            risk_level = "MEDIUM RISK INVOLVED"
        else:
            risk_level = "LOW RISK INVOLVED"

        # Generate the correlation matrix plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Selected Factors')

        # Save the plot to a buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()

        # Generate the feature importances plot
        plt.figure(figsize=(10, 6))
        feature_importances.plot(kind='bar')
        plt.title('Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot to a buffer
        feature_importances_buffer = io.BytesIO()
        plt.savefig(feature_importances_buffer, format='png')
        feature_importances_buffer.seek(0)
        feature_importances_plot_data = base64.b64encode(feature_importances_buffer.getvalue()).decode()

        # Pass the base64 encoded image data to the template
        return render_template('result.html', risk_level=risk_level, correlation_plot=f'data:image/png;base64,{plot_data}', feature_importances_plot=f'data:image/png;base64,{feature_importances_plot_data}')



if __name__ == '__main__':
    app.run(debug=True)

