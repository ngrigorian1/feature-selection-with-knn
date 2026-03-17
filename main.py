import numpy as np
import time

# load dataset into numpy arrays for efficiency, avoids python loops overhead
def load_dataset(file_path):
    data = np.loadtxt(file_path)

    classes = data[:, 0].astype(int)
    features = data[:, 1:].astype(float)

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
    test_row = features[index, selected_features]

    # compare all rows at once instead of one at a time with python loops
    candidate_rows = features[:, selected_features]

    distances = candidate_rows - test_row
    squared_distances = np.sum(distances ** 2, axis=1)

    squared_distances[index] = np.inf # to skip self

    nearest_index = np.argmin(squared_distances)
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

    print(f"Finished search!! The best feature subset is {format_feature_set(best_set)}, which has an accuracy of {best_accuracy:.4f}")

# find good subset by shrinking down from all
def backward_elimination(classes, features):
    num_features = len(features[0])
    current_set = list(range(num_features))
    best_accuracy = leave_one_out_cross_validation(classes, features, current_set)
    best_set = current_set.copy()

    print("\nBeginning search.\n")

    for level in range(num_features):
        feature_to_remove_at_this_level = None
        best_accuracy_at_this_level = 0

        for feature in current_set:
            candidate = current_set.copy()
            candidate.remove(feature)
            accuracy = leave_one_out_cross_validation(classes, features, candidate)

            print(f"Using feature(s) {format_feature_set(candidate)} accuracy is {accuracy:.4f}")

            if accuracy > best_accuracy_at_this_level:
                best_accuracy_at_this_level = accuracy
                feature_to_remove_at_this_level = feature

        current_set.remove(feature_to_remove_at_this_level)

        print(f"Feature set {format_feature_set(current_set)} was best, accuracy is {best_accuracy_at_this_level:.4f}\n")

        if best_accuracy_at_this_level > best_accuracy:
            best_accuracy = best_accuracy_at_this_level
            best_set = current_set.copy()

    print(f"Finished search!! The best feature subset is {format_feature_set(best_set)}, which has an accuracy of {best_accuracy:.4f}")
        
def main():
    print("Welcome to Natalie's Feature Selection Algorithm.")

    file_path = input("Type in the name of the file to test: ").strip()

    classes, features = load_dataset(file_path)

    num_instances = len(classes)
    num_features = len(features[0])

    print("Choose an algorithm:")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    choice = input().strip()

    print(f"\nThis dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")

    all_features = list(range(num_features))
    accuracy = leave_one_out_cross_validation(classes, features, all_features)

    print(f"Running nearest neighbor with all {num_features} features, using leaving-one-out evaluation, I get an accuracy of {accuracy:.4f}")

    start_time = time.perf_counter()

    if choice == "1":
        forward_selection(classes, features)
    elif choice == "2":
        backward_elimination(classes, features)
    else:
        print("Invalid choice. Please try again.")
        return

    elapsed = time.perf_counter() - start_time
    print(f"\nSearch completed in {elapsed:.2f}s")

if __name__ == "__main__":
    main()