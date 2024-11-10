import csv

# Read data from the CSV file
file_path = 'naive.csv'  # Replace with your actual file path
data = []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header row
    for row in reader:
        data.append(row)

# Define the target attribute (class) index and the features
target_column = -1  # The 'class_buys_computer' column
features = [1, 2, 3, 4]  # Columns for age, income, student, credit_rating

# Calculate the prior probabilities (P(class))
class_counts = {}
for row in data:
    target_value = row[target_column]
    if target_value not in class_counts:
        class_counts[target_value] = 0
    class_counts[target_value] += 1

total_rows = len(data)
class_probabilities = {}

# Calculate the prior probabilities for each class with Laplace smoothing
for class_value, count in class_counts.items():
    class_probabilities[class_value] = (count + 1) / (total_rows + len(class_counts))

# Calculate the conditional probabilities (P(feature|class)) with Laplace smoothing
feature_probabilities = {}
for feature_index in features:
    feature_values = {}
    for class_value in class_counts:
        # Initialize counts for each class value with Laplace smoothing
        feature_values[class_value] = {}
    
    for row in data:
        feature_value = row[feature_index]
        class_value = row[target_column]
        
        if feature_value not in feature_values[class_value]:
            feature_values[class_value][feature_value] = 1  # Laplace smoothing initial count of 1
        feature_values[class_value][feature_value] += 1
    
    feature_probabilities[feature_index] = feature_values

# Convert counts to probabilities (P(feature_value|class))
for feature_index, feature_values in feature_probabilities.items():
    for class_value, values in feature_values.items():
        total_class_count = class_counts[class_value] + len(values)  # Adjusted for Laplace smoothing
        for feature_value, count in values.items():
            feature_values[class_value][feature_value] = count / total_class_count

# Function to predict the class and probabilities based on input data
def predict(input_data):
    probabilities = {}
    
    # Calculate the probability for each class
    for class_value, class_prob in class_probabilities.items():
        probability = class_prob
        for i in range(len(input_data)):
            feature_value = input_data[i]
            # Check if feature_value is present in feature_values[class_value]
            if feature_value in feature_probabilities[features[i]][class_value]:
                probability *= feature_probabilities[features[i]][class_value][feature_value]
            else:
                probability *= 1 / (class_counts[class_value] + len(feature_probabilities[features[i]][class_value]))  # Laplace smoothing for unseen values
        
        probabilities[class_value] = probability

    # Return raw probabilities for each class without normalization
    return probabilities

# Show possible options for each feature
def show_options(attribute_index):
    if attribute_index == 1:
        return ["youth", "middle_aged", "senior"]
    elif attribute_index == 2:
        return ["high", "medium", "low"]
    elif attribute_index == 3:
        return ["yes", "no"]
    elif attribute_index == 4:
        return ["fair", "excellent"]
    return []

# Test prediction with user input
print("Enter the values for a new sample:")

# Gather user input for a new instance (e.g., for prediction)
input_data = []
for i in range(len(features)):
    # Display the possible options for the current attribute
    print(f"Possible values for {header[features[i]]}:")
    options = show_options(features[i])
    
    # Display options without using enumerate
    option_index = 1
    for option in options:
        print(f"{option_index}. {option}")
        option_index += 1
    
    # Get user input and validate the input
    user_choice = input(f"Enter the number corresponding to your choice for {header[features[i]]}: ")
    
    try:
        user_choice = int(user_choice)
        if 1 <= user_choice <= len(options):
            input_data.append(options[user_choice - 1])
        else:
            print("Invalid choice. Using the first option as default.")
            input_data.append(options[0])  # Default to the first option if input is invalid
    except ValueError:
        print("Invalid input. Using the first option as default.")
        input_data.append(options[0])  # Default to the first option if input is invalid

# Get the predicted probabilities for each class
probabilities = predict(input_data)

# Print out the predicted probabilities for both classes (yes/no)
print(f"\nPrediction probabilities for the sample:")

# Instead of using .get(), manually check if the class exists in the dictionary
if 'yes' in probabilities:
    print(f"Class 'yes': {probabilities['yes']:.4f}")
else:
    print(f"Class 'yes': 0.0000")

if 'no' in probabilities:
    print(f"Class 'no': {probabilities['no']:.4f}")
else:
    print(f"Class 'no': 0.0000")

# Make the final prediction based on the highest probability
if probabilities['yes'] > probabilities['no']:
    print(f"\nThe predicted class is: 'yes'")
else:
    print(f"\nThe predicted class is: 'no'")
