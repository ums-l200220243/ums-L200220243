from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

X= pd.read_csv("WA.csv")
X.head()

X.info()



import pandas as pd

# Baca file CSV
data = pd.read_csv("WA.csv")

# Tampilkan nama kolom yang ada di file
print("Kolom yang tersedia:", data.columns)

# Pilih kolom yang ingin disaring
columns_to_keep = ["Timestamp", "Message"]  # Pilih kolom sesuai kebutuhan
filtered_data = data.filter(items=columns_to_keep, axis=1)

# Tampilkan kolom setelah pemfilteran
print("Kolom setelah disaring:", filtered_data.columns)

# Opsional: Simpan hasil filter ke file baru
filtered_data.to_csv("filtered_WA.csv", index=False)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Baca file CSV
data = pd.read_csv("WA.csv")

# Ambil kolom 'Message' dan isi nilai kosong
messages = data['Message'].fillna('')

# Konversi teks menjadi fitur numerik menggunakan TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(messages)

# Metode elbow untuk menentukan jumlah cluster optimal
wcss = []

for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot hasil
plt.figure(figsize=(10, 6))
plt.plot(range(1, 15), wcss, marker='o', linestyle='--')
plt.title("Metode Elbow untuk Menentukan Jumlah Cluster Optimal")
plt.xlabel("Jumlah Cluster")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.grid()
plt.show()

from sklearn.cluster import KMeans
import numpy as np

# Example data (X should be your dataset)
X = np.array([[1, 2], [1, 3], [2, 2], [8, 8], [8, 9], [9, 8]])

# Apply KMeans with 3 clusters
model = KMeans(n_clusters=3)
model.fit(X)

# Getting the cluster centers and labels
print("Cluster centers:\n", model.cluster_centers_)
print("Labels:\n", model.labels_)
