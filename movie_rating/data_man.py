import pandas as pd

# Load the original dataset
original_dataset = pd.read_csv('new_movies.csv')

# Select specific columns
selected_columns = ['genre', 'budget', 'gross', 'runtime', 'rating', 'score']  # Replace with the names of the columns you want to select
new_dataset = original_dataset[selected_columns]

new_dataset.reset_index(inplace=True)

# Save the new dataset to a new CSV file
new_dataset.to_csv('new_movies_final.csv', index=False)
