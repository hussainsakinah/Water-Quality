import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
import joblib
import os

st.title("Water Quality Monitoring System")
st.subheader("Analyze water quality and predict potability using machine learning.")

page = option_menu(
    menu_title= None,
    options =  ['Dataset Overview','Preprocessing','Prediction'],
    orientation='horizontal'
)

# File Upload Section
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your water quality CSV file:", type=["csv"])


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data_cl = data.dropna()  # Drop missing values
    output = data_cl['Potability']  # Target variable
    corrs = data_cl.corrwith(output)[:-1]  # Correlation with Potability

    # Select the top 5 correlated features
    best_features = list(corrs.sort_values(ascending=False)[1:].index[:5])
    X = data_cl[best_features]  # Selected features

    # Apply SMOTETomek
    smt = SMOTETomek(random_state=42)
    X_smt, y_smt = smt.fit_resample(X, output)

    # Apply MinMaxScaler to the resampled data
    mm = MinMaxScaler()
    X_smt_scaled = mm.fit_transform(X_smt)

    # Ensure the scaled data is a DataFrame
    X_smt_scaled_df = pd.DataFrame(X_smt_scaled, columns=best_features)


    if page == 'Dataset Overview':
        st.header("Dataset Preview")
        n = st.number_input(label='Enter the first n rows to display', value=5, min_value=1,max_value=len(data))
        st.write(f'#### First {n} columns of the dataset')
        st.table(data.head(n))

        st.write('#### The properties of each column')
        st.write(f"Shape of the DataFrame : *{data.shape}*")
        st.write("##### Columns and Data Types:")
        st.write(data.dtypes)
        st.write("##### Memory Usage:")
        st.write(data.memory_usage(deep=True))

        st.write('#### The distribution of values of each column')
        st.write(data.describe())

    if page == 'Preprocessing' :
        st.subheader("Missing Values Summary")
        st.write(data.isna().sum())
    
        st.write("### Feature Correlations with Potability")
        st.bar_chart(corrs)

        st.write("### Selected Features for Modeling")
        st.table(best_features)

        st.write("### Feature Distributions")
        graphs = st.selectbox(label="Select the feature for which you want to observe the distribution", options=['Hardness' , 'Turbidity' ,'Chloramines','ph','Trihalomethanes'])
        st.write(f"#### {graphs} Distribution")
    
        bins = st.slider("Select Number of Bins:", min_value=5, max_value=len(data_cl[graphs]), value=100)
        fig = px.histogram(data_cl[graphs], nbins=bins, title= f"Distribution of data of {graphs}", labels={"value": graphs})
        fig.update_layout(xaxis_title=graphs, yaxis_title="Frequency",width = 800,height = 525)
        st.plotly_chart(fig)

        st.write('*')

        st.write('### Initial Potability Distribution')
        st.bar_chart(output.value_counts())

        st.write("### Resampled Potability Distribution")
        st.bar_chart(y_smt.value_counts())

    if page == 'Prediction':
        st.header("Water Quality Prediction")

        # Model Training and Loading
        with st.spinner(text='Training the data...'):
            model_name = 'model.pkl'
            feature_names_file = 'feature_names.pkl'
            Xtrain, Xtest, ytrain, ytest = train_test_split(X_smt, y_smt, test_size=0.3, random_state=42)

            # Check if model exists
            if os.path.exists(model_name) and os.path.exists(feature_names_file):
                grid_rfc = joblib.load(model_name)
                saved_features = joblib.load(feature_names_file)
                st.success("Model and feature names loaded successfully!")
            else:
                # Random Forest parameters for GridSearchCV
                params = {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_features': ['sqrt', 'log2'],
                    'n_estimators': range(50, 250, 50),
                }
                kf = KFold(shuffle=True, n_splits=5)
                rfc = RandomForestClassifier(random_state=42)

                # Train the model using GridSearchCV
                grid_rfc = GridSearchCV(rfc, params, cv=kf, scoring='accuracy')
                scaled_Xtrain = mm.fit_transform(Xtrain)
                scaled_Xtrain_df = pd.DataFrame(scaled_Xtrain, columns=[f"{col} mm" for col in Xtrain.columns])
                grid_rfc.fit(scaled_Xtrain_df, ytrain)

                # Save the model and feature names
                joblib.dump(grid_rfc, model_name)
                joblib.dump(scaled_Xtrain_df.columns, feature_names_file)
                saved_features = scaled_Xtrain_df.columns
                st.success("Model trained and saved successfully!")

            # Model Evaluation
            scaled_Xtest = mm.transform(Xtest)
            scaled_Xtest_df = pd.DataFrame(scaled_Xtest, columns=[f"{col} mm" for col in Xtest.columns])
            scaled_Xtest_df = scaled_Xtest_df.reindex(columns=saved_features, fill_value=0)

            predicts = grid_rfc.predict(scaled_Xtest_df)
            cm = confusion_matrix(ytest, predicts)
            accuracy = accuracy_score(ytest, predicts)

            st.write("### Model Performance")
            st.write(f"##### Accuracy of the best Random Forest model: **{accuracy:.2f}**")

            st.write("#### Confusion Matrix")
            fig = px.imshow(cm, color_continuous_scale="viridis", text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig)

        # Prediction Input
        st.header("Predict Water Quality")
        st.write("Enter the feature values below to predict water quality.")

        new_vals = {}
        for feature in data.drop(columns=['Potability']).columns:
            val = st.number_input(f"Enter the value for {feature}", value=float(data[feature].mean()))
            new_vals[feature] = val

        if st.button("Predict Potability"):
            # Ensure `new_vals_list` matches `best_features`
            new_vals_list = [new_vals[feature] for feature in best_features]

            # Debug: Check feature alignment
            st.write("Input feature count:", len(new_vals_list))
            st.write("Scaler was fitted on features:", len(best_features))

            # Scale the input values
            try:
                scaled_new_vals = mm.transform([new_vals_list])[0]
            except ValueError as e:
                st.error(f"Feature mismatch error: {e}")
                st.stop()

            # Make prediction
            prediction = grid_rfc.predict([scaled_new_vals])

            # Display Prediction
            st.write('***')
            st.subheader('Prediction:')
            st.write(f"#### For the given set of values, the predicted category is => **{'Potable' if prediction[0] == 1 else 'Not Potable'}**")


        else:
            st.write("### Please upload a dataset to start.")
