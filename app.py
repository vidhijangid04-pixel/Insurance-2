import streamlit as st
import pandas as pd
import numpy as np
import io, base64
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import label_binarize

st.set_page_config(layout="wide", page_title="Insurance / Policy Dashboard")

st.title("Insurance — Policy Status Dashboard")
st.markdown("""
Upload an insurance dataset (CSV). The app will detect `POLICY_STATUS` as the target if present.
Features:
- 5 interactive charts with filters (role multi-select + satisfaction slider)
- Train Decision Tree, Random Forest, Gradient Boosting with stratified 5-fold CV and view evaluation metrics
- Upload a new dataset and get predictions with downloadable CSV
""")

# ---------- Utilities ----------
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def detect_columns(df):
    cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    return df.columns.tolist(), cat_cols, num_cols

def safe_preprocess(df, target_col):
    df = df.copy()
    # drop extremely high-cardinality columns (likely IDs/names)
    high_card = [c for c in df.columns if df[c].nunique(dropna=True)/len(df) > 0.9 and c != target_col]
    df = df.drop(columns=high_card)
    # drop constant cols
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1 and c != target_col]
    df = df.drop(columns=const_cols)
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    numeric_transformer = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_transformer = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_cols), ('cat', cat_transformer, cat_cols)], remainder='drop')
    return preprocessor, X, y, numeric_cols, cat_cols

def train_models(X, y, preprocessor, random_state=42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=random_state, stratify=y_enc)
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=random_state),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}
    for name, model in models.items():
        pipe = Pipeline([('pre', preprocessor), ('clf', model)])
        y_train_pred_cv = cross_val_predict(pipe, X_train, y_train, cv=cv, method='predict', n_jobs=1)
        pipe.fit(X_train, y_train)
        y_test_pred = pipe.predict(X_test)
        y_test_proba = pipe.predict_proba(X_test)
        if len(class_names) == 2:
            prec = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
            auc = roc_auc_score(y_test, y_test_proba[:,1])
        else:
            prec = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
            y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
            auc = roc_auc_score(y_test_bin, y_test_proba, average='macro', multi_class='ovr')
        results[name] = {
            'pipeline': pipe,
            'y_train': y_train,
            'y_train_pred_cv': y_train_pred_cv,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba,
            'metrics': {
                'train_acc_cv': accuracy_score(y_train, y_train_pred_cv),
                'test_acc': accuracy_score(y_test, y_test_pred),
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc
            },
            'class_names': class_names
        }
    return results, le

def plot_confusion(cm, class_names, title="Confusion Matrix", cmap='Blues'):
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    st.pyplot(fig)

def plot_roc(results):
    fig = go.Figure()
    for name, res in results.items():
        proba = res['y_test_proba']; y_test = res['y_test']; class_names = res['class_names']
        if len(class_names) == 2:
            fpr, tpr, _ = roc_curve(y_test, proba[:,1])
            auc_val = res['metrics']['auc']
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc_val:.3f})"))
        else:
            y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
            fpr = np.linspace(0,1,100); tprs = []
            for i in range(len(class_names)):
                fpr_i, tpr_i, _ = roc_curve(y_test_bin[:,i], proba[:,i])
                tprs.append(np.interp(fpr, fpr_i, tpr_i))
            mean_tpr = np.mean(tprs, axis=0); auc_val = res['metrics']['auc']
            fig.add_trace(go.Scatter(x=fpr, y=mean_tpr, mode='lines', name=f"{name} (AUC={auc_val:.3f})"))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig.update_layout(title="ROC Curves (Test Set)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

def get_feature_importances(results, feature_names):
    fi = {}
    for name, res in results.items():
        model = res['pipeline'].named_steps['clf']
        try:
            importances = model.feature_importances_
            fi[name] = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        except Exception:
            fi[name] = None
    return fi

def download_link(df, filename="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

# ---------- Sidebar & Data ----------
st.sidebar.header("Data & Filters")
uploaded = st.sidebar.file_uploader("Upload CSV (or skip to use sample)", type=['csv'])
if uploaded is not None:
    df = load_csv(uploaded)
else:
    try:
        df = pd.read_csv("Insurance.csv")
        st.sidebar.success("Using bundled sample Insurance.csv")
    except Exception:
        st.sidebar.info("No sample found. Please upload a CSV file.")
        st.stop()

cols, cat_cols, num_cols = detect_columns(df)
st.sidebar.write(f"Columns detected: {len(cols)} (cat: {len(cat_cols)}, num: {len(num_cols)})")

# Target selection
default_target = "POLICY_STATUS" if "POLICY_STATUS" in df.columns else cols[-1]
target_col = st.sidebar.selectbox("Target column", options=cols, index=cols.index(default_target))
st.sidebar.markdown(f"Target: **{target_col}**")

# Role-like column for filters
possible_roles = [c for c in cat_cols if c != target_col]
if not possible_roles:
    possible_roles = [c for c in cols if c != target_col]
role_col = st.sidebar.selectbox("Role / Segment column (for filters)", options=possible_roles, index=0)

# Satisfaction-like numeric column for slider
possible_sats = [c for c in num_cols if c != target_col]
if not possible_sats:
    possible_sats = [c for c in num_cols]
if possible_sats:
    sat_col = st.sidebar.selectbox("Satisfaction / score column (slider)", options=possible_sats, index=0)
else:
    sat_col = None

# Filters
selected_roles = st.sidebar.multiselect(f"Filter {role_col} (multi-select)", options=sorted(df[role_col].dropna().unique().tolist()), default=None)
sat_range = None
if sat_col and pd.api.types.is_numeric_dtype(df[sat_col]):
    minv, maxv = float(df[sat_col].min()), float(df[sat_col].max())
    sat_range = st.sidebar.slider(f"{sat_col} range", min_value=minv, max_value=maxv, value=(minv, maxv))
else:
    sat_col = None

df_chart = df.copy()
if selected_roles:
    df_chart = df_chart[df_chart[role_col].isin(selected_roles)]
if sat_col and sat_range:
    df_chart = df_chart[df_chart[sat_col].between(sat_range[0], sat_range[1])]

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Insights & Charts", "Train & Evaluate Models", "Predict New Data"])

with tab1:
    st.header("Interactive Charts & Insights")
    st.markdown("Five charts providing different and actionable insights for insurance availability / policy review. Use the left filters to refine the view.")

    # Chart 1: Stacked bar of target by role (proportions)
    if target_col in df_chart.columns:
        c1 = df_chart.groupby([role_col, target_col]).size().reset_index(name='count')
        fig1 = px.bar(c1, x=role_col, y='count', color=target_col, title=f"{target_col} by {role_col}", barmode='relative', text_auto=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("**Action:** Identify roles with high negative outcomes and investigate underwriting or process issues.")

    # Chart 2: Violin of satisfaction by target
    if sat_col and sat_col in df_chart.columns:
        fig2 = px.violin(df_chart, x=target_col, y=sat_col, box=True, points='outliers', title=f"{sat_col} distribution by {target_col}")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("**Action:** Low satisfaction groups correlated with negative outcomes might need targeted outreach.")

    # Chart 3: Correlation heatmap (numeric features including encoded target)
    st.subheader("Correlation heatmap (numeric features + encoded target)")
    numerics = df_chart.select_dtypes(include=['number']).columns.tolist()
    if target_col in df_chart.columns and numerics:
        tmp = df_chart[numerics].copy()
        try:
            tmp[target_col] = LabelEncoder().fit_transform(df_chart[target_col])
            corr = tmp.corr()
            fig3, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax')
            st.pyplot(fig3)
            st.markdown("**Action:** Use correlated numeric variables as candidate predictors for deeper modelling.")
        except Exception as e:
            st.write("Could not compute correlations:", e)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

    # Chart 4: Top categories contributing to negative target (ratio)
    st.subheader("Top segments by negative outcome proportion")
    if target_col in df_chart.columns:
        # pick negative label as last sorted by name for reproducibility
        labels_sorted = sorted(df_chart[target_col].dropna().unique().tolist())
        if labels_sorted:
            negative = labels_sorted[-1]
            grp = df_chart.groupby([role_col, target_col]).size().unstack(fill_value=0)
            if negative in grp.columns:
                grp['neg_ratio'] = grp[negative] / grp.sum(axis=1)
                top_neg = grp['neg_ratio'].sort_values(ascending=False).reset_index().head(12)
                fig4 = px.bar(top_neg, x=role_col, y='neg_ratio', title=f"Top {role_col} by proportion of '{negative}'", text=top_neg['neg_ratio'].round(2))
                st.plotly_chart(fig4, use_container_width=True)
                st.markdown("**Action:** Prioritize these top segments for process audits or customer outreach.")
            else:
                st.info("Negative label not identifiable for this chart.")
    # Chart 5: Sankey flow between role_col and another top categorical (helps see policy routing)
    st.subheader("Category flow (role -> other category)")
    other_cat = None
    for c in df_chart.select_dtypes(include=['object','category']).columns:
        if c not in [role_col, target_col]:
            other_cat = c; break
    if other_cat:
        temp = df_chart[[role_col, other_cat]].dropna()
        left_top = temp[role_col].value_counts().nlargest(8).index.tolist()
        right_top = temp[other_cat].value_counts().nlargest(8).index.tolist()
        temp = temp[temp[role_col].isin(left_top) & temp[other_cat].isin(right_top)]
        flow = temp.groupby([role_col, other_cat]).size().reset_index(name='count')
        labels = list(pd.Index(flow[role_col].tolist() + flow[other_cat].tolist()).unique())
        source = flow[role_col].apply(lambda x: labels.index(x)).tolist()
        target = flow[other_cat].apply(lambda x: labels.index(x)).tolist()
        fig5 = go.Figure(data=[go.Sankey(node=dict(label=labels, pad=15), link=dict(source=source, target=target, value=flow['count'].tolist()))])
        fig5.update_layout(title_text=f"Sankey: {role_col} -> {other_cat}", height=400)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("**Action:** Spot clusters or bottlenecks in policy routing.")

with tab2:
    st.header("Train & Evaluate Models (DT, RF, GBRT)")
    st.write("Click the button to run training (stratified 5-fold CV used for training confusion matrices).")
    if st.button("Run Training & Evaluation"):
        with st.spinner("Training models..."):
            preprocessor, X, y, numeric_cols, cat_cols = safe_preprocess(df, target_col)
            results, le = train_models(X, y, preprocessor)
        st.success("Training finished. See metrics below.")

        # Metrics table
        rows = []
        for name,res in results.items():
            m = res['metrics']
            rows.append({'Algorithm': name, 'Train Acc (cv-oof)': m['train_acc_cv'], 'Test Acc': m['test_acc'], 'Precision': m['precision'], 'Recall': m['recall'], 'F1': m['f1'], 'AUC': m['auc']})
        metrics_df = pd.DataFrame(rows).set_index('Algorithm')
        st.subheader("Metrics")
        st.dataframe(metrics_df.style.format("{:.4f}"))

        # Confusion matrices
        st.subheader("Confusion Matrices - Training (CV OOF)")
        cols_plot = st.columns(len(results))
        for i,(name,res) in enumerate(results.items()):
            with cols_plot[i]:
                cm = confusion_matrix(res['y_train'], res['y_train_pred_cv'])
                st.write(f"**{name} - Train (CV)**")
                plot_confusion(cm, res['class_names'], title=f"{name} - Train (CV)")

        st.subheader("Confusion Matrices - Test")
        cols_plot = st.columns(len(results))
        for i,(name,res) in enumerate(results.items()):
            with cols_plot[i]:
                cm = confusion_matrix(res['y_test'], res['y_test_pred'])
                st.write(f"**{name} - Test**")
                plot_confusion(cm, res['class_names'], title=f"{name} - Test", cmap='Greens')

        # ROC
        st.subheader("ROC Curves (Test)")
        plot_roc(results)

        # Feature importances
        st.subheader("Feature Importances (Top)")
        feature_names = (numeric_cols + cat_cols)
        fi = get_feature_importances(results, feature_names)
        for name, series in fi.items():
            st.markdown(f"**{name}**")
            if series is None:
                st.write("No feature importances available for this model.")
            else:
                st.bar_chart(series.head(15))

        # Save for prediction tab
        st.session_state['trained_results'] = results
        st.session_state['label_encoder'] = le

with tab3:
    st.header("Upload New Data & Predict POLICY_STATUS")
    uploaded_new = st.file_uploader("Upload new CSV for prediction (same schema)", type=['csv'], key="predict_new")
    if uploaded_new is not None:
        new_df = pd.read_csv(uploaded_new)
        st.write("Preview uploaded file:")
        st.dataframe(new_df.head())

        if 'trained_results' not in st.session_state:
            st.info("No trained models in session. Run training in 'Train & Evaluate Models' tab first.")
        else:
            model_choice = st.selectbox("Choose model", options=list(st.session_state['trained_results'].keys()))
            if st.button("Predict and Download"):
                res = st.session_state['trained_results'][model_choice]
                pipe = res['pipeline']
                try:
                    preds = pipe.predict(new_df)
                    labels = st.session_state['label_encoder'].inverse_transform(preds)
                    out = new_df.copy(); out['PRED_'+target_col] = labels
                    st.write("Predictions preview:")
                    st.dataframe(out.head())
                    href = download_link(out, filename="predictions_with_label.csv")
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Package prepared for Streamlit Cloud. If any error occurs for very different datasets, select appropriate target/filters from the sidebar.")
