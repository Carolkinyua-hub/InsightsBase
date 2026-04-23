import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px

st.set_page_config(page_title='InsightsBase', layout='wide')
st.title('InsightsBase')
st.caption('Upload → Clean → Analyze → Export')

# ---------- Helpers ----------
def load_data(file):
    name = file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(file)
    if name.endswith('.tsv') or name.endswith('.txt'):
        return pd.read_csv(file, sep='\t')
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(file)
    raise ValueError('Unsupported file format')


def clean_dataframe(df, drop_cols, trim_ws, remove_dupes, blanks_to_na):
    data = df.copy()
    if drop_cols:
        data = data.drop(columns=drop_cols, errors='ignore')
    if trim_ws:
        obj = data.select_dtypes(include='object').columns
        for c in obj:
            data[c] = data[c].astype(str).str.strip()
    if blanks_to_na:
        data = data.replace(r'^\s*$', np.nan, regex=True)
    if remove_dupes:
        data = data.drop_duplicates()
    return data


def numeric_stats(df, cols):
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors='coerce')
        rows.append({
            'column': c,
            'count': s.count(),
            'mean': s.mean(),
            'median': s.median(),
            'std': s.std(),
            'min': s.min(),
            'max': s.max(),
            'missing_%': round(s.isna().mean()*100,2)
        })
    return pd.DataFrame(rows)


def categorical_stats(df, cols):
    rows = []
    for c in cols:
        vc = df[c].astype(str).value_counts(dropna=False)
        rows.append({
            'column': c,
            'unique_values': df[c].nunique(dropna=True),
            'top_value': vc.index[0] if len(vc) else None,
            'top_count': int(vc.iloc[0]) if len(vc) else 0,
            'missing_%': round(df[c].isna().mean()*100,2)
        })
    return pd.DataFrame(rows)

# ---------- Upload ----------
file = st.file_uploader('Upload CSV / Excel / TSV', type=['csv','xlsx','xls','tsv','txt'])

if file:
    try:
        df = load_data(file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Sidebar
    st.sidebar.header('Cleaning Controls')
    missing_strategy = st.sidebar.selectbox('Missing Value Treatment', ['Leave As Is','Drop Rows With Missing','Fill Numeric Mean','Fill Numeric Median','Fill Text Mode'])
    drop_cols = st.sidebar.multiselect('Drop Columns', df.columns.tolist())
    trim_ws = st.sidebar.checkbox('Trim Whitespace', value=True)
    remove_dupes = st.sidebar.checkbox('Remove Duplicates', value=True)
    blanks_to_na = st.sidebar.checkbox('Convert Blanks to Null', value=True)

    clean_df = clean_dataframe(df, drop_cols, trim_ws, remove_dupes, blanks_to_na)
    if missing_strategy == 'Drop Rows With Missing':
        clean_df = clean_df.dropna()
    elif missing_strategy == 'Fill Numeric Mean':
        for c in clean_df.select_dtypes(include=np.number).columns:
            clean_df[c] = clean_df[c].fillna(clean_df[c].mean())
    elif missing_strategy == 'Fill Numeric Median':
        for c in clean_df.select_dtypes(include=np.number).columns:
            clean_df[c] = clean_df[c].fillna(clean_df[c].median())
    elif missing_strategy == 'Fill Text Mode':
        for c in clean_df.columns:
            if clean_df[c].dtype == 'object':
                m = clean_df[c].mode(dropna=True)
                if len(m):
                    clean_df[c] = clean_df[c].fillna(m.iloc[0])

    # Preview
    st.subheader('Dataset Overview')
    c1, c2, c3 = st.columns(3)
    c1.metric('Rows', len(clean_df))
    c2.metric('Columns', len(clean_df.columns))
    c3.metric('Missing Cells', int(clean_df.isna().sum().sum()))

    st.write('Preview')
    st.dataframe(clean_df.head(20), use_container_width=True)

    meta = pd.DataFrame({
        'column': clean_df.columns,
        'dtype': clean_df.dtypes.astype(str).values,
        'missing': clean_df.isna().sum().values
    })
    st.write('Columns Summary')
    st.dataframe(meta, use_container_width=True)

    # Stats
    st.subheader('Descriptive Statistics')
    numeric_cols = clean_df.select_dtypes(include=np.number).columns.tolist()
    object_cols = [c for c in clean_df.columns if c not in numeric_cols]

    sel_num = st.multiselect('Numeric Columns', numeric_cols, default=numeric_cols[:5])
    sel_cat = st.multiselect('Categorical Columns', object_cols, default=object_cols[:5])

    num_df = numeric_stats(clean_df, sel_num) if sel_num else pd.DataFrame()
    cat_df = categorical_stats(clean_df, sel_cat) if sel_cat else pd.DataFrame()

    if not num_df.empty:
        st.write('Numeric Statistics')
        st.dataframe(num_df, use_container_width=True)
    if not cat_df.empty:
        st.write('Categorical Statistics')
        st.dataframe(cat_df, use_container_width=True)

    # Pivot
    st.subheader('Pivot Table Builder')
    st.caption('Build and preview pivot summaries instantly.')
    rows = st.selectbox('Rows', ['None'] + clean_df.columns.tolist())
    cols = st.selectbox('Columns', ['None'] + clean_df.columns.tolist())
    vals = st.selectbox('Values', ['None'] + clean_df.columns.tolist())
    agg = st.selectbox('Aggregation', ['count','sum','mean','median'])

    try:
        pivot = pd.pivot_table(
            clean_df,
            index=None if rows=='None' else rows,
            columns=None if cols=='None' else cols,
            values=None if vals=='None' else vals,
            aggfunc=agg,
            fill_value=0
        )
        st.write('Pivot Preview')
        st.dataframe(pivot, use_container_width=True)
    except:
        pivot = pd.DataFrame()

    # Charts
    st.subheader('Visualisations')
    chart_col = st.selectbox('Chart Column', clean_df.columns.tolist())
    chart_type = st.selectbox('Chart Type', ['Bar','Histogram','Boxplot','Line','Pie','Frequency Table','CLT Simulation'])

    try:
        if chart_type == 'Bar':
            vc = clean_df[chart_col].astype(str).value_counts().head(20)
            fig = px.bar(x=vc.index, y=vc.values, labels={'x':chart_col,'y':'Count'})
        elif chart_type == 'Histogram':
            fig = px.histogram(clean_df, x=chart_col)
        elif chart_type == 'Boxplot':
            fig = px.box(clean_df, y=chart_col)
        elif chart_type == 'Pie':
            vc = clean_df[chart_col].astype(str).value_counts().head(10)
            fig = px.pie(values=vc.values, names=vc.index)
        elif chart_type == 'Frequency Table':
            vc = clean_df[chart_col].astype(str).value_counts().reset_index()
            vc.columns = [chart_col, 'count']
            st.dataframe(vc, use_container_width=True)
            fig = px.bar(vc, x=chart_col, y='count')
        elif chart_type == 'CLT Simulation':
            vals = pd.to_numeric(clean_df[chart_col], errors='coerce').dropna()
            if len(vals) > 5:
                means = [vals.sample(min(30, len(vals)), replace=True).mean() for _ in range(300)]
                fig = px.histogram(x=means, nbins=30, labels={'x':'Sample Means'})
            else:
                st.info('Need numeric data for CLT simulation.')
                fig = None
        else:
            num = clean_df.select_dtypes(include=np.number).columns.tolist()
            ycol = num[0] if num else None
            fig = px.line(clean_df.reset_index(), x=clean_df.index, y=ycol)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.info('Chart unavailable for selected column.')

    # Export
    st.subheader('Export Report')
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        clean_df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
        if not num_df.empty:
            num_df.to_excel(writer, sheet_name='Numeric_Stats', index=False)
        if not cat_df.empty:
            cat_df.to_excel(writer, sheet_name='Categorical_Stats', index=False)
        if not pivot.empty:
            pivot.to_excel(writer, sheet_name='Pivot_Table')
        meta.to_excel(writer, sheet_name='Data_Quality', index=False)

    st.download_button(
        'Download Excel Report',
        data=output.getvalue(),
        file_name='insightbase_report.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
else:
    st.info('Upload a file to begin.')
