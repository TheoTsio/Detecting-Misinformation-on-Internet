import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.mosaicplot import mosaic

dataframe = pd.read_csv('C:/Users/teots/OneDrive/Υπολογιστής/Creating Queries For Batch Indexing/Final_Dataset_Finished_Processing_Train_Val_Test/train.csv')

# Bar Plot
plt.bar(dataframe["Credibility"], dataframe['Num_Bad_Words'])
plt.xlabel('Credibility Class (0, 1, 2)')
plt.ylabel('Total Number of bad words')
plt.show()

# Heatmap
correlation_matrix = dataframe[['Num_Bad_Words', 'Credibility']].corr()
print(correlation_matrix)
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap between Num_Bad_Words and Credibility')
plt.show()

# Heatmap
correlation_matrix = dataframe[['Num_Emoji', 'Credibility']].corr()
print(correlation_matrix)
plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap between Nume Emoji and Credibility')
plt.show()

# Mosaic Plot
mosaic(dataframe, ['Credibility', 'Num_Bad_Words'], title='Mosaic Plot: Credibility vs. Number of Bad Words')
plt.show()

# Clustered Bar Chart
frequency_table = dataframe.groupby(['Credibility', 'Num_Bad_Words']).size().unstack()

# Create a clustered bar chart
ax = frequency_table.plot(kind='bar', stacked=True, figsize=(10, 6))
ax.set_xlabel('Credibility')
ax.set_ylabel('Frequency')
ax.set_title('Clustered Bar Chart: Credibility vs. Number of Bad Words')
plt.legend(title='Number of Bad Words')
plt.show()

# original_length = len(dataframe)
# filtered_df = dataframe[dataframe['Num_Bad_Words'] < 30]
# filtered_length = len(filtered_df)
# excluded_rows = original_length - filtered_length
# print("excluded:",excluded_rows)
# Bar Plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Credibility', y="Num_Bad_Words", data=dataframe, estimator=np.mean)  # Use mean as the estimator for numerical values
plt.xlabel("Credibility")
plt.ylabel("Num_Emoji")
plt.title(f"Bar Plot: Average(Num_Emoji) vs Credibility")
plt.show()

