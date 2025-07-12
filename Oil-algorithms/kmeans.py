import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# مسیر فایل دیتا را جایگزین کنید
file_path = "data\\a.csv"

# خواندن دیتا
data = pd.read_csv(file_path)

# درخواست سال از کاربر
target_year = int(input("Enter the target year: "))

# انتخاب دیتاهای مربوط به سال مورد نظر
selected_data = data[data['Year'] == target_year]

# استخراج فیچرهای مورد نظر
features = selected_data[['Oil consumption - TWh']]

# اعمال الگوریتم K-means با 7 کلاس
kmeans = KMeans(n_clusters=7, random_state=42)
selected_data['Cluster'] = kmeans.fit_predict(features)

#########################################################################################################
# Load population data
population_file_path = "data\\population-and-demography.csv"
population_data = pd.read_csv(population_file_path)

# Merge oil consumption data with population data
merged_data = pd.merge(data, population_data, left_on=['Entity', 'Year'], right_on=['Country name', 'Year'])

# Calculate oil consumption per capita (in TWh)
merged_data['Oil consumption per capita - TWh'] = merged_data['Oil consumption - TWh'] / merged_data['Population']

# Select data for the target year
selected_data2 = merged_data[merged_data['Year'] == target_year]

# Extract features
features = selected_data2[['Oil consumption per capita - TWh']]

# Apply K-means algorithm with 7 clusters
kmeans2 = KMeans(n_clusters=7, random_state=42)
selected_data2['Cluster'] = kmeans2.fit_predict(features)

# Plot countries with different colors based on clusters
fig, ax = plt.subplots(nrows=2, ncols=1)#, figsize=(15, 10))
ax[0].scatter(selected_data['Entity'], selected_data['Oil consumption - TWh'], c=selected_data['Cluster'], cmap='rainbow')
ax[0].set_title(f'Oil Consumption Clustering for {target_year}')
ax[0].set_xlabel('Country')
ax[0].set_ylabel('Oil Consumption - TWh')
#ax[0].set_xticks(90)
ax[1].scatter(selected_data2['Entity'], selected_data2['Oil consumption per capita - TWh'], c=selected_data2['Cluster'], cmap='rainbow')
ax[1].set_title(f'Oil Consumption Clustering per Capita for {target_year}')
ax[1].set_xlabel('Country')
ax[1].set_ylabel('Oil Consumption per Capita - TWh')
#ax[1].set_xticks(90)
plt.tight_layout()
plt.show()
