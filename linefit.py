import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import mahalanobis
import statsmodels.api as sm

st.set_page_config(layout="wide")
st.title("LineFitLab: Streamlined Linear Regression Builder")

# Initialize session state
if 'section' not in st.session_state:
    st.session_state.section = 1
if 'encoding_df' not in st.session_state:
    st.session_state.encoding_df = None

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 1. Data Preprocessing
    if st.session_state.section >= 1:
        st.header("1. Data Preprocessing")
        with st.spinner("Loading content..."):
            # 1.1 Column Selection
            st.subheader("1.1 Column Selection")
            st.write("First 10 rows of the dataset:")
            st.write(df.head(10))
            st.write("Column datatypes:")
            buffer = pd.DataFrame(df.dtypes, columns=['dtype'])
            st.write(buffer)
            
            columns_to_remove = st.multiselect("Select columns to remove", df.columns, key="columns_to_remove")
            if columns_to_remove:
                df = df.drop(columns=columns_to_remove)
            
            # 1.2 Target Selection
            st.subheader("1.2 Target Selection")
            target_col = st.selectbox("Select target column", df.columns, key="target_col")
            
            if st.button("Next", key="next_preprocessing"):
                st.session_state.section = 2
    
    # 2. Exploratory Data Analysis
    if st.session_state.section >= 2:
        st.header("2. Exploratory Data Analysis")
        with st.spinner("Loading content..."):
            # 2.1 Null and Duplicate Handling
            st.subheader("2.1 Null and Duplicate Handling")
            null_count = df.isnull().sum().sum()
            duplicate_count = df.duplicated().sum()
            df = df.dropna().drop_duplicates()
            st.write(f"Removed {null_count} null values and {duplicate_count} duplicate rows")
            
            # 2.2 Categorical Encoding
            st.subheader("2.2 Categorical Encoding")
            
            def analyze_dataframe(df, target_col):
                result_df = pd.DataFrame({
                    'column_name': df.columns,
                    'dtype': df.dtypes
                }).reset_index(drop=True)
                
                result_df['status'] = ''
                result_df['unique_values'] = ''
                
                for idx, row in result_df.iterrows():
                    col_name = row['column_name']
                    col_type = row['dtype']
                    
                    if col_name == target_col:
                        result_df.at[idx, 'status'] = 'target'
                        unique_vals = df[col_name].dropna().unique()
                        if len(unique_vals) > 10:
                            unique_vals = list(unique_vals[:10]) + ['...']
                        result_df.at[idx, 'unique_values'] = ', '.join(map(str, unique_vals))
                        continue
                    
                    if col_type in [np.int64, np.int32, np.float64, np.float32]:
                        result_df.at[idx, 'status'] = 'approved'
                    else:
                        key = f"encoding_{col_name}_{idx}"
                        encoding_choice = st.selectbox(
                            f"Select encoding for {col_name} ({col_type})",
                            ["label encoding", "one-hot encoding"],
                            key=key
                        )
                        result_df.at[idx, 'status'] = encoding_choice
                    
                    unique_vals = df[col_name].dropna().unique()
                    if len(unique_vals) > 10:
                        unique_vals = list(unique_vals[:10]) + ['...']
                    result_df.at[idx, 'unique_values'] = ', '.join(map(str, unique_vals))
                
                return result_df
            
            def encode_data(df, target_col, encoding_df):
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                label_cols = encoding_df[encoding_df['status'] == 'label encoding']['column_name']
                onehot_cols = encoding_df[encoding_df['status'] == 'one-hot encoding']['column_name']
                approved_cols = encoding_df[encoding_df['status'] == 'approved']['column_name']
                
                X_encoded = X[approved_cols].copy() if not approved_cols.empty else pd.DataFrame(index=X.index)
                
                for col in label_cols:
                    if col in X.columns:
                        try:
                            le = LabelEncoder()
                            X_encoded[col] = le.fit_transform(X[col].astype(str))
                        except Exception as e:
                            st.warning(f"Failed to label encode {col}: {str(e)}")
                
                if onehot_cols.any():
                    try:
                        X_onehot = pd.get_dummies(X[onehot_cols], columns=onehot_cols, dtype=float)
                        X_encoded = pd.concat([X_encoded, X_onehot], axis=1)
                    except Exception as e:
                        st.warning(f"Failed to one-hot encode {onehot_cols.tolist()}: {str(e)}")
                
                for col in X_encoded.columns:
                    try:
                        X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
                    except:
                        st.warning(f"Column {col} could not be converted to numeric, dropping it.")
                        X_encoded = X_encoded.drop(columns=[col])
                
                X_encoded = X_encoded.dropna()
                y = y.loc[X_encoded.index]
                
                return X_encoded, y
            
            try:
                encoding_df = analyze_dataframe(df, target_col)
                st.session_state.encoding_df = encoding_df  # Store for pipeline visualization
                st.write("Column analysis and encoding options:")
                st.dataframe(encoding_df, use_container_width=True)
                
                X_encoded, y_encoded = encode_data(df, target_col, encoding_df)
                df_encoded = pd.concat([X_encoded, y_encoded.rename(target_col)], axis=1)
            except Exception as e:
                st.error(f"Error in encoding: {str(e)}")
                st.session_state.section = 2
                st.stop()
            
            # 2.3 Outlier Detection
            st.subheader("2.3 Outlier Detection")
            outlier_methods = [
                "No outlier removal", "IQR", "Z-score", "Modified Z-score",
                "Cook's Distance", "Leverage", "Residuals", 
                "Mahalanobis Distance", "Isolation Forest"
            ]
            r2_scores = {}
            
            def get_cleaned_data(method, df_encoded, target_col):
                df_temp = df_encoded.copy()
                X = df_temp.drop(columns=[target_col])
                y = df_temp[target_col]
                
                if method == "No outlier removal":
                    return X, y
                
                X_numeric = X.select_dtypes(include=['float64', 'int64'])
                if X_numeric.empty:
                    st.warning("No numeric columns available for outlier detection.")
                    return X, y
                
                if method == "IQR":
                    Q1 = X_numeric.quantile(0.25)
                    Q3 = X_numeric.quantile(0.75)
                    IQR = Q3 - Q1
                    mask = ~((X_numeric < (Q1 - 1.5 * IQR)) | (X_numeric > (Q3 + 1.5 * IQR))).any(axis=1)
                    return X[mask], y[mask]
                
                elif method == "Z-score":
                    z_scores = np.abs(stats.zscore(X_numeric))
                    mask = (z_scores < 3).all(axis=1)
                    return X[mask], y[mask]
                
                elif method == "Modified Z-score":
                    MAD = np.median(np.abs(X_numeric - np.median(X_numeric, axis=0)), axis=0)
                    modified_z_scores = 0.6745 * (X_numeric - np.median(X_numeric, axis=0)) / MAD
                    mask = (np.abs(modified_z_scores) < 3.5).all(axis=1)
                    return X[mask], y[mask]
                
                elif method == "Cook's Distance":
                    X_sm = sm.add_constant(X_numeric)
                    model = sm.OLS(y, X_sm).fit()
                    influence = model.get_influence()
                    cooks_d = influence.cooks_distance[0]
                    mask = cooks_d < 4/len(X_numeric)
                    return X[mask], y[mask]
                
                elif method == "Leverage":
                    X_sm = sm.add_constant(X_numeric)
                    model = sm.OLS(y, X_sm).fit()
                    influence = model.get_influence()
                    leverage = influence.hat_matrix_diag
                    mask = leverage < 2*len(X_numeric.columns)/len(X_numeric)
                    return X[mask], y[mask]
                
                elif method == "Residuals":
                    model = LinearRegression().fit(X, y)
                    predictions = model.predict(X)
                    residuals = np.abs(y - predictions)
                    threshold = np.std(residuals) * 3
                    mask = residuals < threshold
                    return X[mask], y[mask]
                
                elif method == "Mahalanobis Distance":
                    cov = np.cov(X_numeric.T)
                    inv_cov = np.linalg.pinv(cov)
                    mean = X_numeric.mean()
                    mahal_dist = np.array([mahalanobis(x, mean, inv_cov) for x in X_numeric.values])
                    threshold = np.sqrt(stats.chi2.ppf(0.95, df=len(X_numeric.columns)))
                    mask = mahal_dist < threshold
                    return X[mask], y[mask]
                
                elif method == "Isolation Forest":
                    iso = IsolationForest(contamination=0.1, random_state=42)
                    mask = iso.fit_predict(X_numeric) == 1
                    return X[mask], y[mask]
            
            for method in outlier_methods:
                try:
                    X_clean, y_clean = get_cleaned_data(method, df_encoded, target_col)
                    if len(X_clean) > 0:
                        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                        model = LinearRegression().fit(X_train, y_train)
                        r2_scores[method] = r2_score(y_test, model.predict(X_test))
                    else:
                        r2_scores[method] = np.nan
                except Exception as e:
                    st.warning(f"Error computing R2 for {method}: {str(e)}")
                    r2_scores[method] = np.nan
            
            r2_df = pd.DataFrame.from_dict(r2_scores, orient='index', columns=['R2 Score'])
            r2_df = r2_df.dropna()
            if not r2_df.empty:
                fig = px.bar(r2_df, x=r2_df.index, y='R2 Score', 
                             title='R2 Scores by Outlier Detection Method',
                             color=['red' if x == r2_df['R2 Score'].min() else 'blue' for x in r2_df['R2 Score']])
                st.plotly_chart(fig)
            
            selected_outlier_method = st.selectbox("Select outlier removal method", r2_df.index, key="outlier_method")
            
            # 2.4 Multicollinearity Removal
            st.subheader("2.4 Multicollinearity Removal")
            vif_options = ["No VIF removal", "VIF-based removal"]
            
            def calculate_vif(X):
                X_numeric = X.select_dtypes(include=['float64', 'int64']).dropna()
                if X_numeric.empty:
                    st.warning("No numeric columns available for VIF calculation.")
                    return pd.DataFrame({"feature": [], "VIF": []})
                
                if not np.all(np.isfinite(X_numeric.values)):
                    st.warning("NaN or infinite values detected in numeric columns, attempting to drop them.")
                    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).dropna()
                    if X_numeric.empty:
                        st.warning("No valid data left after removing NaN/infinite values.")
                        return pd.DataFrame({"feature": [], "VIF": []})
                
                try:
                    vif_data = pd.DataFrame()
                    vif_data["feature"] = X_numeric.columns
                    vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
                    return vif_data
                except Exception as e:
                    st.warning(f"VIF calculation failed: {str(e)}")
                    st.write("Data types in X_numeric:", X_numeric.dtypes)
                    return pd.DataFrame({"feature": [], "VIF": []})
            
            def remove_high_vif(X, threshold=5):
                if X.empty:
                    return X
                
                X_numeric = X.select_dtypes(include=['float64', 'int64']).dropna()
                if X_numeric.empty:
                    st.warning("No numeric columns available for VIF-based removal, skipping.")
                    return X
                
                while True:
                    vif_data = calculate_vif(X_numeric)
                    if vif_data.empty:
                        return X
                    
                    if (vif_data["VIF"] > threshold).any():
                        max_vif = vif_data.loc[vif_data["VIF"].idxmax()]
                        X_numeric = X_numeric.drop(columns=[max_vif["feature"]])
                        X = X.drop(columns=[max_vif["feature"]])
                    else:
                        break
                
                return X
            
            vif_r2_scores = {}
            for vif_option in vif_options:
                try:
                    X_clean, y_clean = get_cleaned_data(selected_outlier_method, df_encoded, target_col)
                    
                    if vif_option == "VIF-based removal":
                        X_clean = remove_high_vif(X_clean)
                    
                    if X_clean.empty or len(X_clean) < 2:
                        st.warning(f"No valid features left for {vif_option}, skipping R2 calculation.")
                        vif_r2_scores[vif_option] = np.nan
                        continue
                    
                    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                    model = LinearRegression().fit(X_train, y_train)
                    vif_r2_scores[vif_option] = r2_score(y_test, model.predict(X_test))
                except Exception as e:
                    st.warning(f"Error computing R2 for {vif_option}: {str(e)}")
                    st.write("X_clean dtypes:", X_clean.dtypes)
                    vif_r2_scores[vif_option] = np.nan
            
            vif_r2_df = pd.DataFrame.from_dict(vif_r2_scores, orient='index', columns=['R2 Score'])
            vif_r2_df = vif_r2_df.dropna()
            if not vif_r2_df.empty:
                fig_vif = px.bar(vif_r2_df, x=vif_r2_df.index, y='R2 Score', 
                                title='R2 Scores by Multicollinearity Removal Method',
                                color=['red' if x == vif_r2_df['R2 Score'].min() else 'blue' for x in vif_r2_df['R2 Score']])
                st.plotly_chart(fig_vif)
            
            selected_vif_method = st.selectbox("Select multicollinearity removal method", vif_r2_df.index, key="vif_method")
            
            if st.button("Next", key="next_eda"):
                st.session_state.section = 3
    
    # 3. Model Building and Evaluation
    if st.session_state.section >= 3:
        st.header("3. Model Building and Evaluation")
        with st.spinner("Loading content..."):
            # 3.1 Pipeline Visualization
            st.subheader("3.1 Pipeline Visualization")
            
            # Compute encoding step dynamically
            encoding_step = "No encoding (all numeric columns)"
            if st.session_state.encoding_df is not None:
                encoded_cols = st.session_state.encoding_df[
                    st.session_state.encoding_df['status'].isin(['label encoding', 'one-hot encoding'])
                ]
                if not encoded_cols.empty:
                    encoding_parts = [
                        f"{row['column_name']} ({row['status'].replace(' encoding', '')})"
                        for _, row in encoded_cols.iterrows()
                    ]
                    encoding_step = f"Encoding: {', '.join(encoding_parts)}"
            
            pipeline_steps = [
                f"Remove columns: {columns_to_remove}",
                f"Target: {target_col}",
                encoding_step,
                f"Outlier removal: {selected_outlier_method}",
                f"Multicollinearity: {selected_vif_method}",
                "Linear Regression"
            ]
            
            # Enhanced pipeline visualization
            pipeline_html = """
            <style>
                .pipeline-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    align-items: center;
                    background-color: #F0F2F6;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
                .pipeline-step {
                    font-size: 18px;
                    font-weight: bold;
                    color: #31333F;
                    background-color: #FFFFFF;
                    padding: 10px 20px;
                    border-radius: 5px;
                    border: 1px solid #FF4B4B;
                }
                .pipeline-arrow {
                    font-size: 18px;
                    color: #FF4B4B;
                    margin: 0 10px;
                }
            </style>
            <div class="pipeline-container">
            """
            for i, step in enumerate(pipeline_steps):
                pipeline_html += f'<span class="pipeline-step">{step}</span>'
                if i < len(pipeline_steps) - 1:
                    pipeline_html += '<span class="pipeline-arrow">➔</span>'
            pipeline_html += "</div>"
            
            st.markdown(pipeline_html, unsafe_allow_html=True)
            
            # 3.2 Linear Regression
            st.subheader("3.2 Linear Regression Results")
            try:
                X_final, y_final = encode_data(df, target_col, st.session_state.encoding_df)
                X_final, y_final = get_cleaned_data(selected_outlier_method, pd.concat([X_final, y_final.rename(target_col)], axis=1), target_col)
                if selected_vif_method == "VIF-based removal":
                    X_final = remove_high_vif(X_final)
                
                if X_final.empty:
                    st.error("No valid features left after preprocessing.")
                    st.stop()
                
                X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
                lr_model = LinearRegression().fit(X_train, y_train)
                
                train_r2 = r2_score(y_train, lr_model.predict(X_train))
                test_r2 = r2_score(y_test, lr_model.predict(X_test))
                train_rmse = np.sqrt(mean_squared_error(y_train, lr_model.predict(X_train)))
                test_rmse = np.sqrt(mean_squared_error(y_test, lr_model.predict(X_test)))
                
                st.write(f"Training R2 Score: {train_r2:.4f}")
                st.write(f"Testing R2 Score: {test_r2:.4f}")
                st.write(f"Training RMSE: {train_rmse:.4f}")
                st.write(f"Testing RMSE: {test_rmse:.4f}")
                
                # 3.3 Model Fit Assessment
                st.subheader("3.3 Model Fit Assessment")
                r2_diff = abs(train_r2 - test_r2)
                if r2_diff > 0.1:
                    fit_status = "Overfitting" if train_r2 > test_r2 else "Underfitting"
                else:
                    fit_status = "Good fit"
                st.write(f"Model Fit: {fit_status} (R2 difference: {r2_diff:.4f})")
            except Exception as e:
                st.error(f"Error in linear regression: {str(e)}")
            
            if st.button("Next", key="next_model"):
                st.session_state.section = 4
    
    # 4. Advanced Models
    if st.session_state.section >= 4:
        st.header("4. Advanced Models")
        with st.spinner("Loading content..."):
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Elastic Net": ElasticNet()
            }
            
            model_results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    model_results[name] = {
                        "R2 Score": r2_score(y_test, model.predict(X_test)),
                        "RMSE": np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
                    }
                except Exception as e:
                    st.warning(f"Error computing metrics for {name}: {str(e)}")
                    model_results[name] = {"R2 Score": np.nan, "RMSE": np.nan}
            
            results_df = pd.DataFrame(model_results).T
            results_df = results_df.dropna()
            if not results_df.empty:
                fig_models = px.bar(results_df, x=results_df.index, y='R2 Score',
                                   title='Model Comparison - R2 Scores')
                st.plotly_chart(fig_models)
                
                st.write("Model Performance Metrics:")
                st.write(results_df)
                
                best_model = results_df['R2 Score'].idxmax()
                st.write(f"Best performing model: {best_model} (R2 Score: {results_df.loc[best_model, 'R2 Score']:.4f})")
