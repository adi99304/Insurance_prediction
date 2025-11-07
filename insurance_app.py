
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px


st.set_page_config(page_title="📊 Insurance Data Analysis & Model Comparison", layout="wide")
st.title("Hello Hegde!")


st.sidebar.header("Upload Data & Options")
file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

plot_type = st.sidebar.selectbox("🎨 Choose Plot Type:", ["Distribution Plot", "Scatter Plot", "Box Plot", "Violin Plot", "Correlation Heatmap"])
model_type = st.sidebar.selectbox("🤖 Choose Model to Predict:", ["Linear Regression", "Decision Tree", "Random Forest"])


data = None
if file is not None:
    data = pd.read_csv(file)
    if 'sex' in data.columns: data['sex'] = data['sex'].astype('category')
    if 'smoker' in data.columns: data['smoker'] = data['smoker'].astype('category')
    if 'region' in data.columns: data['region'] = data['region'].astype('category')
    st.subheader("📄 Data Preview")
    st.dataframe(data.head())

    st.subheader("📊 Data Summary")
    st.write(data.describe(include='all'))

   
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    all_cols = data.columns.tolist()

    x_var = st.sidebar.selectbox("📊 Select X Variable:", all_cols, index=all_cols.index('age') if 'age' in all_cols else 0)
    y_var = st.sidebar.selectbox("📊 Select Y Variable (Target):", all_cols, index=all_cols.index('charges') if 'charges' in all_cols else -1)
    color_var = st.sidebar.selectbox("🎨 Optional Color Variable:", ['None'] + all_cols)
    plot_btn = st.sidebar.button("📊 Generate Plot")


    if plot_btn:
        st.subheader("📈 Plot")
        plt.figure(figsize=(10, 5))
        if plot_type == "Distribution Plot":
            fig = px.histogram(data, x=x_var, nbins=30, title=f"Distribution of {x_var}")
        elif plot_type == "Scatter Plot":
            fig = px.scatter(data, x=x_var, y=y_var, color=color_var if color_var != 'None' else None, title=f"{x_var} vs {y_var}")
        elif plot_type == "Box Plot":
            fig = px.box(data, x=x_var, y=y_var, color=color_var if color_var != 'None' else None, title=f"Box Plot of {y_var} by {x_var}")
        elif plot_type == "Violin Plot":
            fig = px.violin(data, x=x_var, y=y_var, color=color_var if color_var != 'None' else None, box=True, points="all", title=f"Violin Plot of {y_var} by {x_var}")
        elif plot_type == "Correlation Heatmap":
            corr = data.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

 
    if st.sidebar.button("⚡ Train Model"):
        st.subheader("🏋️ Model Training & Evaluation")
        X = data.drop(columns=[y_var])
        y = data[y_var]
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = None
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Decision Tree":
            model = DecisionTreeRegressor()
        elif model_type == "Random Forest":
            model = RandomForestRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**R²:** {r2:.2f}")

   
        st.session_state['trained_model'] = model
        st.session_state['X_columns'] = X.columns

    

if 'trained_model' in st.session_state:
    st.subheader("🔮 Test Model with Custom Input")
    
   
    col1, col2 = st.columns(2)
    
    inputs = {}
    
   
    with col1:
        for col in st.session_state['X_columns']:
            if 'sex_' in col or 'smoker_' in col or 'region_' in col:
                continue
            inputs[col] = st.number_input(f"{col}", value=0.0)
    
  
    with col2:
       
        sex_option = st.selectbox("Sex", ["male", "female"])
        
     
        smoker_option = st.selectbox("Smoker", ["yes", "no"])
        
       
        region_option = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    if st.button("💡 Predict"):
        
        test_input_df = pd.DataFrame([inputs])
        
        
        for col in st.session_state['X_columns']:
            if col not in test_input_df.columns:
                test_input_df[col] = 0
        
       
        if 'sex_male' in st.session_state['X_columns']:
            test_input_df['sex_male'] = 1 if sex_option == "male" else 0
        
        if 'smoker_yes' in st.session_state['X_columns']:
            test_input_df['smoker_yes'] = 1 if smoker_option == "yes" else 0
        
        for region in ["northeast", "northwest", "southeast", "southwest"]:
            col_name = f'region_{region}'
            if col_name in st.session_state['X_columns']:
                test_input_df[col_name] = 1 if region_option == region else 0
        
       
        test_input_df = test_input_df[st.session_state['X_columns']]
        
      
        prediction = st.session_state['trained_model'].predict(test_input_df)
        st.success(f"🔮 Predicted Value: {prediction[0]:.2f}")

else:
    st.info("📂 Please upload a CSV file to begin.")
