import pandas as pd
import matplotlib.pyplot as plt


dataframe = pd.read_csv('train.csv')

class_counts = dataframe['Credibility'].value_counts()
class_names = class_counts.index
print(class_names)

# BAR PLOT
plt.bar(class_names, class_counts)
plt.xlabel('Credibility Class (0, 1, 2)')
plt.ylabel('Number of rows')
plt.title('BAR PLOT')
# Set the x-axis tick labels to the class names
plt.xticks(class_names)
plt.show()

# PIE CHART
plt.pie(class_counts, labels=class_names, autopct='%1.1f%%')
plt.title('PIE CHART (Shows how many rows belong to each)')
plt.show()

# Histogram
plt.hist(dataframe['Credibility'], bins=len(class_names))
plt.xlabel('Credibility Class')
plt.ylabel('Count')
plt.title('HISTOGRAM')
plt.xticks(class_names)
plt.show()

# Seaborn Count Plot
import seaborn as sns

sns.countplot(data=dataframe, x='Credibility')
plt.xlabel('Credibility Class')
plt.ylabel('Count')
plt.title('SEABORN')
plt.show()


