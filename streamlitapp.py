import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from collections import Counter

# Sample data creation
data = {
    'Damage Classification': [
        'Type1', 'Type2', 'Type1', 'Type3', 'Type2',
        'Type1', 'Type3', 'Type2', 'Type1', 'Type3',
        'Type2', 'Type1', 'Type3', 'Type2', 'Type1',
        'Type3', 'Type2', 'Type1', 'Type3', 'Type2'
    ],
    'Damage Description': [
        'Scratch on surface', 'Broken hinge', 'Dent on side', 'Cracked screen', 'Loose button',
        'Worn out rubber', 'Color fading', 'Battery issue', 'Overheating', 'Screen flickering',
        'Water damage', 'Speaker issue', 'Charging problem', 'Network issue', 'Camera issue',
        'Fingerprint sensor issue', 'Microphone problem', 'Power button issue', 'Volume control issue', 'Touchscreen issue'
    ],
    'Damage Location': [
        'top', 'bottom', 'middle', 'right side', 'left side',
        'top', 'bottom', 'middle', 'right side', 'left side',
        'top', 'bottom', 'middle', 'right side', 'left side',
        'top', 'bottom', 'middle', 'right side', 'left side'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Preprocess the data
tfidf = TfidfVectorizer(stop_words='english', max_features=10)
description_vectors = tfidf.fit_transform(df['Damage Description'].values.astype('U')).toarray()

label_encoder = LabelEncoder()
location_encoded = label_encoder.fit_transform(df['Damage Location'])

# Combine both features
features = np.hstack((description_vectors, location_encoded.reshape(-1, 1)))

# Streamlit title
st.title('Damage Data Clustering Visualization')

# Sidebar for filtering
st.sidebar.header('Filter Options')

# Slider for selecting number of clusters
num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=5)

# Cluster the data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# Add clusters to the DataFrame
df['Cluster'] = clusters

# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

df['PCA1'] = reduced_features[:, 0]
df['PCA2'] = reduced_features[:, 1]

# Assign descriptive names to clusters
cluster_names = {}
for cluster_num in range(num_clusters):
    descriptions = df[df['Cluster'] == cluster_num]['Damage Description']
    most_common_words = Counter(" ".join(descriptions).split()).most_common(3)
    cluster_name = " ".join([word for word, _ in most_common_words])
    cluster_names[cluster_num] = cluster_name

# Map cluster numbers to descriptive names
df['Cluster Name'] = df['Cluster'].map(cluster_names)

selected_clusters = st.sidebar.multiselect(
    'Clusters',
    options=cluster_names.keys(),
    default=cluster_names.keys(),
    format_func=lambda x: cluster_names[x]
)

# Filter the data based on selected clusters
filtered_df = df[df['Cluster'].isin(selected_clusters)]

# Show filtered data
st.dataframe(filtered_df)

# Plot the clusters
st.subheader('Clusters Visualization')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_df, x='PCA1', y='PCA2', hue='Cluster Name', palette='viridis')
plt.title('Problem Descriptions and Locations Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
st.pyplot(plt.gcf())

# Plot the first graph: Damage Classification by Cluster
st.subheader('Damage Classification by Cluster')

if not filtered_df.empty:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='Cluster Name', hue='Damage Classification')
    plt.title('Damage Classification by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    st.pyplot(plt.gcf())
else:
    st.write('No data to display. Please adjust your filters.')

# Plot the second graph: Damage Description by Cluster
st.subheader('Damage Description by Cluster')

if not filtered_df.empty:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='Cluster Name', hue='Damage Description')
    plt.title('Damage Description by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    st.pyplot(plt.gcf())
else:
    st.write('No data to display. Please adjust your filters.')

# Plot the third graph: Number of each cluster at each location
st.subheader('Number of Each Cluster at Each Location')

if not filtered_df.empty:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='Damage Location', hue='Cluster Name')
    plt.title('Number of Each Cluster at Each Location')
    plt.xlabel('Damage Location')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    st.pyplot(plt.gcf())
else:
    st.write('No data to display. Please adjust your filters.')

if __name__ == "__main__":
    st.write("")
