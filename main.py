# convert data into a list of lists
def load_dataset(file_path):
    rows = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if line == "":
                continue

            values = line.split()
            row = [float(value) for value in values]
            rows.append(row)

    classes = []
    features = []

    for row in rows:
        classes.append(int(row[0]))
        features.append(row[1:])

    return classes, features

# need smallest distance between two feature vectors
def squared_distance(row1, row2, selected_features):
    total = 0

    for i in selected_features:
        diff = row1[i] - row2[i]
        total += diff * diff
    
    return total

# predict the class of the test row using the nearest neighbor
def predict_one_nearest_neighbor(classes, features, index, selected_features):
    test_row = features[index]

    smallest_distance = float('inf')
    nearest_index = -1

    for i in range(len(features)):
        if i == index:
            continue

        current_distance = squared_distance(test_row, features[i], selected_features)

        if current_distance < smallest_distance:
            smallest_distance = current_distance
            nearest_index = i
    
    return classes[nearest_index]

# hide an instance, predict its class, and check if it was correct
def leave_one_out_cross_validation(classes, features, selected_features):
    correct = 0

    for i in range(len(features)):
        prediction = predict_one_nearest_neighbor(classes, features, i, selected_features)

        if prediction == classes[i]:
            correct += 1
    
    accuracy = correct / len(features)
    return accuracy

# adding 1 to the features for display since my features start at 0
def format_feature_set(selected_features):
    display_numbers = []

    for feature in selected_features:
        display_numbers.append(str(feature + 1))

    return "{" + ",".join(display_numbers) + "}"

# find good subset by building up from none
def forward_selection(classes, features):
    current_set = []
    best_accuracy = 0
    best_set = []
    num_features = len(features[0])

    print("\nBeginning search.\n")

    for level in range(num_features):
        feature_to_add_at_this_level = None
        best_accuracy_at_this_level = 0

        # dont add the same feature twice
        for feature in range(num_features):
            if feature in current_set:
                continue

            candidate = current_set + [feature]
            accuracy = leave_one_out_cross_validation(classes, features, candidate)

            print(f"Using feature(s) {format_feature_set(candidate)} accuracy is {accuracy:.4f}")

            if accuracy > best_accuracy_at_this_level:
                best_accuracy_at_this_level = accuracy
                feature_to_add_at_this_level = feature
        
        current_set.append(feature_to_add_at_this_level)

        print(f"Feature set {format_feature_set(current_set)} was best, accuracy is {best_accuracy_at_this_level:.4f}\n")

        if best_accuracy_at_this_level > best_accuracy:
            best_accuracy = best_accuracy_at_this_level
            best_set = current_set.copy()

    print(f"Finished search!! The best feature subset is {format_feature_set(best_set)}, "
        f"which has an accuracy of {best_accuracy:.4f}")

def main():
    print("Welcome to Natalie's Feature Selection Algorithm.")

    file_path = input("Type in the name of the file to test: ").strip()

    classes, features = load_dataset(file_path)

    num_instances = len(classes)
    num_features = len(features[0])

    print(f"\nThis dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")

    all_features = list(range(num_features))
    accuracy = leave_one_out_cross_validation(classes, features, all_features)

    print(
        f"Running nearest neighbor with all {num_features} features, using leaving-one-out evaluation, "
        f"I get an accuracy of {accuracy:.4f}"
    )

    forward_selection(classes, features)

if __name__ == "__main__":
    main()