import json

# Open the JSON file in read mode
with open('test4_cut.mp4.openpifpaf.json', 'r') as file:
    input_json = file.read()

# Split the input JSON by lines
lines = input_json.strip().split('\n')

# Create a list to store the processed objects
output_list = []

# Process each line and append to the output list
for line in lines:
    data = json.loads(line)
    output_list.append(data)

# Create the output dictionary
output_dict = {
    "project": "project_name",
    "output": output_list
}

# Convert the output dictionary to JSON
output_json = json.dumps(output_dict, indent=4)

# Open a new file to write the output JSON
with open('output.json', 'w') as file:
    file.write(output_json)

# Print a success message
print("Output JSON file created successfully.")
