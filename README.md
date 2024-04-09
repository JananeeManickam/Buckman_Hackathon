# Buckman_Hackathon

Investment Decision Recommendation System Project
Introduction
This project aims to build a recommendation system that can accurately predict investment risks based on individual attributes, aiding in informed decision-making.

Understanding the Problem Statement
The goal is to build a recommendation system that can predict the investment risk level for new data, based on important factors, to help users make the best decisions.

Technical Details
Language: Python 3.12
Editor: VS Code
Model Selection
The project utilizes the Random Forest algorithm due to its robustness against overfitting, ability to handle large datasets with high dimensionality, and capability to provide feature importance.

Process
1. Data Preprocessing
Data preprocessing involves encoding categorical variables and removing unwanted fields.

2. Factor Selection Process
The next step is to identify factors that contribute to making the best investment decision.

3. Model Training
After selecting the relevant factors, the Random Forest algorithm is used to train the model. This algorithm is chosen for its robustness against overfitting and its ability to handle large datasets with high dimensionality.

4. Deployment and User Interface
The final step is to deploy the trained model into a Flask web application. This application allows users to input their characteristics and receive personalized risk predictions. The predictions are displayed along with a correlation matrix visualization to provide users with a clear understanding of the factors influencing their investment risk.

These steps collectively contribute to the development of a recommendation system that assists users in making informed investment decisions based on their individual attributes.

Key Outcomes
The system accurately predicts investment risks based on individual attributes, aiding in informed decision-making.
The trained model is capable of predicting investment risks accurately, assisting users in decision-making.
The Flask web application allows users to input their characteristics and receive personalized risk predictions, displayed along with a correlation matrix visualization.
Further Development
Future developments may focus on enhancing model performance and expanding the application's features for broader financial analysis.
