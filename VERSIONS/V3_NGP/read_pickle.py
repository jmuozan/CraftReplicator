import pickle

# Open and load the pickle file
with open('testing_data_800x800.pkl', 'rb') as f:
    data = pickle.load(f)

# Print information about the loaded data
print(f"Type of data: {type(data)}")
print("\nPreview of data:")
print(data)