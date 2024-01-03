

# # pip install scikit-surprise

from surprise import Dataset
# from surprise import Reader
# from surprise.model_selection import train_test_split
# from surprise import KNNBasic
# from surprise import accuracy

# # Load your data into Surprise
# # Replace this with your own dataset
# reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
# data = Dataset.load_from_file('diabetes.csv', reader)

# # Split the data into training and testing sets
# trainset, testset = train_test_split(data, test_size=0.2)

# # Use the KNNBasic collaborative filtering algorithm
# sim_options = {
#     'name': 'cosine',
#     'user_based': False
# }

# model = KNNBasic(sim_options=sim_options)
# model.fit(trainset)

# # Make predictions on the test set
# predictions = model.test(testset)

# # Evaluate the accuracy of the model
# accuracy.rmse(predictions)



import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('diabetes.csv')

# Display basic information about the dataset
print(df.info())

# Display descriptive statistics
print(df.describe())

# Plot a histogram for a specific column (replace 'column_name' with your actual column)
plt.hist(df['outcome'], bins=10, color='blue', edgecolor='black')
plt.title('Histogram of Column Name')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Create a scatter plot (replace 'x_column' and 'y_column' with your actual columns)
plt.scatter(df['x_column'], df['y_column'], color='green', marker='o')
plt.title('Scatter Plot')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.show()