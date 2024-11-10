import csv
import math

def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def entropy(data):
    total = len(data)
    if total == 0:
        return 0
    class_labels = [row["class_buys_computer"] for row in data]
    count_yes = class_labels.count("yes")
    count_no = total - count_yes
    p_yes = count_yes / total
    p_no = count_no / total
    entropy_yes = -p_yes * math.log2(p_yes) if p_yes > 0 else 0
    entropy_no = -p_no * math.log2(p_no) if p_no > 0 else 0
    return entropy_yes + entropy_no

def information_gain(data, attribute):
    total_entropy = entropy(data)
    attribute_values = [row[attribute] for row in data]
    values = set(attribute_values)
    weighted_entropy = 0
    for value in values:
        subset = [row for row in data if row[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    gain = total_entropy - weighted_entropy
    print(f"Information Gain for {attribute}: {gain:.4f}")
    return gain

def split_data(data, attribute, value):
    return [row for row in data if row[attribute] == value]

def build_tree(data, attributes):
    if all(row['class_buys_computer'] == 'yes' for row in data):
        return "yes"
    if all(row['class_buys_computer'] == "no" for row in data):
        return "no"
    if not attributes:
        count_yes = 0
        for row in data:
            if row["class_buys_computer"] == "yes":
                count_yes += 1

            threshold = len(data) / 2

            if count_yes >= threshold:
                majority_class = "yes"
            else:
                majority_class = "no"

        return majority_class

    
    # Displaying information gain for each attribute
    print("Calculating Information Gain for each attribute:")
    information_gains = {}
    for attribute in attributes:
        gain = information_gain(data, attribute)
        information_gains[attribute] = gain

    best_attribute = max(information_gains, key=information_gains.get)
    tree = {best_attribute: {}}
    attribute_values = [row[best_attribute] for row in data]
    values = set(attribute_values)

    for value in values:
        subset = split_data(data, best_attribute, value)
        subtree = build_tree(subset, [attr for attr in attributes if attr != best_attribute])
        tree[best_attribute][value] = subtree
    return tree

def display_tree(tree, indent=''):
    if isinstance(tree, dict):
        for k, v in tree.items():
            print(f"{indent}{k}")
            for subkey, subvalue in v.items():
                if isinstance(subvalue, dict):
                    print(f"{indent}  {subkey} -> ")
                    display_tree(subvalue, indent + '    ')
                else:
                    print(f"{indent}  {subkey} -> {subvalue}")
    else:
        print(f"{indent}Predict: {tree}")

def predict(tree, instance, indent=''):
    if not isinstance(tree, dict):
        print(f"{indent}Predict: {tree}")
        return tree
    for attribute, subtree in tree.items():
        break
    value = instance.get(attribute, None)
    print(f"{indent}{attribute} = {value}")
    subtree = tree[attribute]
    
    if value in subtree:
        subtree = subtree[value]
    else:
        subtree = "no"
    return predict(subtree, instance, indent + '    ')

def get_user_input():
    print("Enter your details by selecting the corresponding number:\n")
    print("Age: 1 - Youth, 2 - Middle-aged, 3 - Senior")
    age = input("Enter age (1/2/3): ")
    age_map = {"1": "youth", "2": "middle_aged", "3": "senior"}
    
    print("Income: 1 - High, 2 - Medium, 3 - Low")
    income = input("Enter income (1/2/3): ")
    income_map = {"1": "high", "2": "medium", "3": "low"}
    
    print("Student: 1 - Yes, 2 - No")
    student = input("Are you a student? (1/2): ")
    student_map = {"1": "yes", "2": "no"}
    
    print("Credit Rating: 1 - Fair, 2 - Excellent")
    credit_rating = input("Enter credit rating (1/2): ")
    credit_rating_map = {"1": "fair", "2": "excellent"}
    
    return {
        'age': age_map.get(age, "youth"), 
        'income': income_map.get(income, "medium"), 
        'student': student_map.get(student, "no"), 
        'credit_rating': credit_rating_map.get(credit_rating, "fair")
    }

def main():
    filename = "decision.csv"
    data = load_data(filename)
    attributes = ['age', 'income', 'student', 'credit_rating']
    tree = build_tree(data, attributes)
    print("\nDecision Tree:")
    display_tree(tree)
    print("\nPrediction Process:")
    instance = get_user_input()
    prediction = predict(tree, instance)
    print(f"\nFinal Prediction for class_buys_computer: {prediction}")

if __name__ == "__main__":
    main()
