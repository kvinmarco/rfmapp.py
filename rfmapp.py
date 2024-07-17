import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px 

# Title of the Streamlit app
st.set_page_config(page_title = 'RFM Analysis Application', page_icon=':bar_chart:')
st.title('Customer Segmentation using RFM Analysis')

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Preview of the dataset:")
    st.write(df.head())

    # Data Preprocessing
    df['customer_data'] = df['customer_data'].fillna('null name')
    df['account_id'] = df['account_id'].astype(str)

    # Display total amount spent
    total_amount_spent = df['amount/100'].sum()
    st.write(f'Total amount spent: Rp{total_amount_spent}')

    # Calculate RFM
    customer_spending = df.groupby('account_id')['amount/100'].sum().reset_index()
    customer_spending.columns = ['account_id', 'Monetary']

    customer_frequency = df['account_id'].value_counts().reset_index()
    customer_frequency.columns = ['account_id', 'Frequency']

    df['insert_dtime'] = pd.to_datetime(df['insert_dtime'], format='%d/%m/%Y %H:%M')
    current_date = datetime.now()

    df_most_recent = df.groupby('account_id')['insert_dtime'].max().reset_index()
    df_most_recent['Recency'] = (current_date - df_most_recent['insert_dtime']).dt.days

    df_rfm = pd.merge(df_most_recent[['account_id', 'Recency']], customer_frequency, on='account_id')
    df_rfm = pd.merge(df_rfm, customer_spending, on='account_id')
    df_rfm[['Recency', 'Frequency', 'Monetary']] = df_rfm[['Recency', 'Frequency', 'Monetary']].astype(int)

    # Keep a copy of the original RFM values
    original_rfm = df_rfm[['account_id', 'Recency', 'Frequency', 'Monetary']].copy()

    # Standardize RFM values
    sc = StandardScaler()
    df_rfm[['Recency', 'Frequency', 'Monetary']] = sc.fit_transform(df_rfm[['Recency', 'Frequency', 'Monetary']])

    # Function to plot 3D Scatter Plot
    def plot_3d_scatter(df_rfm, n_clusters):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df_rfm['Recency'], df_rfm['Frequency'], df_rfm['Monetary'], c=df_rfm['Cluster'], cmap='viridis', s=50)
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        ax.set_title(f'Clustering of Customers based on RFM (Clusters: {n_clusters})')
        legend = ax.legend(*scatter.legend_elements(), title='Clusters')
        ax.add_artist(legend)
        st.pyplot(fig)

    # Elbow method for K-means clustering
    st.subheader('Elbow Method for Optimal Number of Clusters')
    data = df_rfm[['Recency', 'Frequency', 'Monetary']]
    inertias = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Graph')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    st.pyplot(plt)

    # Slider to select number of clusters
    n_clusters = st.slider('Select number of clusters', 1, 10, 5)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_rfm['Cluster'] = kmeans.fit_predict(data)

    # Plot 3D scatter plot
    plot_3d_scatter(df_rfm, n_clusters)

    # Scale RFM values for visualization
    scaler = MinMaxScaler(feature_range=(0, 5))
    df_rfm[['Frequency', 'Monetary']] = scaler.fit_transform(df_rfm[['Frequency', 'Monetary']])

    # Display Scaled clusters based on frequency and monetary with background image
    background_image_path = "https://github.com/kvinmarco/rfmapp.py/raw/main/rfmtable.png"
    background_image = plt.imread(background_image_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(background_image, extent=[0, 5, 0, 5], aspect='auto', alpha=0.4)
    scatter = ax.scatter(df_rfm['Frequency'], df_rfm['Monetary'], c=df_rfm['Cluster'], cmap='viridis', s=50)
    ax.set_xlabel('Frequency (Scaled)')
    ax.set_ylabel('Monetary (Scaled)')
    ax.set_title('Scaled Clusters based on Frequency and Monetary')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)

    # Function to assign segments
    segments = {
        'Lost': ((-1, 2), (-1, 2)),
        'Hibernating': ((-1, 2), (2, 4)),
        'Can’t Lose Them': ((-1, 2), (4, 6)),
        'About to Sleep': ((2, 3), (-1, 2)),
        'Needs attention': ((2, 3), (2, 3)),
        'Loyal Customers': ((2, 4), (3, 6)),
        'Promising': ((3, 4), (-1, 1)),
        'Potential Loyalist': [((3, 4), (1, 3)), ((4, 6), (2, 3))],
        'Price Sensitive': ((4, 6), (-1, 1)),
        'Recent users': ((4, 6), (1, 2)),
        'Champions': ((4, 6), (3, 6))
    }

    def assign_segment(row):
        for segment, bounds in segments.items():
            if isinstance(bounds, list):
                for (x_range, y_range) in bounds:
                    if x_range[0] <= row['Frequency'] <= x_range[1] and y_range[0] <= row['Monetary'] <= y_range[1]:
                        return segment
            else:
                x_range, y_range = bounds
                if x_range[0] <= row['Frequency'] <= x_range[1] and y_range[0] <= row['Monetary'] <= y_range[1]:
                    return segment
        return 'Other'

    df_rfm['Segment'] = df_rfm.apply(assign_segment, axis=1)

    # Button to show customer segments
    if st.button('Show Customer Segments'):
        # Assign each coordinate to their respective account_id
        df_rfm['Original Frequency'] = original_rfm['Frequency']
        df_rfm['Original Monetary'] = original_rfm['Monetary']

        # Calculate average frequency and monetary for each segment based on original values
        segment_stats = df_rfm.groupby('Segment').agg({
            'account_id': 'count',
            'Recency': 'mean'
        }).reset_index()

        original_segment_stats = df_rfm.groupby('Segment').agg({
            'Original Frequency': 'mean',
            'Original Monetary': 'mean'
        }).reset_index()

        segment_stats = segment_stats.merge(original_segment_stats, on='Segment')
        segment_stats.columns = ['Segment', 'Count', 'Average Recency', 'Average Frequency', 'Average Monetary']

        total_customers = df_rfm.shape[0]
        segment_stats['Percentage'] = (segment_stats['Count'] / total_customers) * 100
        segment_stats['Percentage'] = segment_stats['Percentage'].apply(lambda x: f'{x:.2f}%')

        # Prepare the data for displaying in a table
        segment_stats['Average Monetary(Spending)'] = segment_stats['Average Monetary'].apply(lambda x: f'Rp{x:,.2f}')
        segment_stats = segment_stats[['Segment', 'Count', 'Percentage', 'Average Frequency', 'Average Monetary(Spending)']]

        # Reset index and start from 1
        segment_stats.reset_index(drop=True, inplace=True)
        segment_stats.index += 1

        # Display the segment statistics in a table without the index
        st.table(segment_stats)
        
        fig = px.pie(segment_stats, names='Segment', values='Count',
                 hover_data=['Percentage', 'Average Frequency', 'Average Monetary(Spending)'],
                 title='Customer Segments Distribution')
        fig.update_traces(
        textposition='inside', 
        textinfo='percent+label', 
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Count: %{value}<br>'
            'Percentage: %{customdata[0]}<br>'
        ),
        customdata=segment_stats[['Percentage']].values
    )
        st.plotly_chart(fig)

    # Multiselect button to show account IDs based on segments
    selected_segments = st.multiselect('Show account IDs based on segments', segments.keys())

    if selected_segments:
        for segment in selected_segments:
            segment_ids = df_rfm[df_rfm['Segment'] == segment]['account_id'].tolist()
            st.subheader(f"Account IDs based on '{segment}':")
            if segment_ids:
                for account_id in segment_ids:
                    st.markdown(f"- {account_id}")
            else:
                st.markdown(f"There are no accounts in the '{segment}' segment.")

        # If no segments are selected, display this message
        if not any(segment_ids for segment in selected_segments):
            st.markdown(f"There are no accounts in the selected segments.")
            
    # Text input to search for an account ID
    st.subheader('Search for Customer Details')
    search_account_id = st.text_input('Enter account ID')

    if search_account_id:
        # Check if the account ID exists in the dataset
        if search_account_id in df['account_id'].values:
            # Extract the customer data
            customer_data = df[df['account_id'] == search_account_id]
            total_spending = customer_data['amount/100'].sum()
            frequency = customer_data.shape[0]
            purchase_details = customer_data[['insert_dtime', 'amount/100']].to_dict('records')
            customer_name = customer_data['customer_data'].values[0]

            st.write(f"Account ID: {search_account_id}")
            st.write(f"Name: {customer_name}")
            st.write(f"Total Spending: Rp{total_spending}")
            st.write(f"Frequency: {frequency} times")

            st.write("Date of Purchase and Amount per Purchase:")
            for purchase in purchase_details:
                st.write(f"- Date: {purchase['insert_dtime']}, Amount: Rp{purchase['amount/100']}")
        else:
            st.write("Account ID not found in the dataset.")
