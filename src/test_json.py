import json

# Step 1: Create an empty JSON file
filename = "data.json"
with open(filename, "w") as file:
    json.dump({}, file)  # Write an empty dictionary to the file

# Step 2: Load the JSON file
with open(filename, "r") as file:
    data = json.load(file)

# Step 3: Print the loaded data
print(data)  # Output: {}
