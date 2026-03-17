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
def squared_distance(row1, row2):
    total = 0

    for i in range(len(row1)):
        diff = row1[i] - row2[i]
        total += diff * diff
    
    return total

# predict the class of the test row using the nearest neighbor
def predict_one_nearest_neighbor(classes, features, index):
    test_row = features[index]

    smallest_distance = float('inf')
    nearest_index = -1

    for i in range(len(features)):
        if i == index:
            continue

        current_distance = squared_distance(test_row, features[i])

        if current_distance < smallest_distance:
            smallest_distance = current_distance
            nearest_index = i
    
    prediction = classes[nearest_index]

    return prediction, nearest_index, smallest_distance


def main():
    print("Welcome to Natalie's Feature Selection Algorithm.")

    file_path = input("Type in the name of the file to test: ").strip()

    classes, features = load_dataset(file_path)

    num_instances = len(classes)
    num_features = len(features[0])

    print(f"\nThis dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")

    test_index = 0
    predicted, neighbor_index, distance = predict_one_nearest_neighbor(classes, features, test_index)

#printing to make sure the nearest neighbor is working
    print("\nNearest neighbor test:")
    print(f"Actual class: {classes[test_index]}")
    print(f"Nearest neighbor index: {neighbor_index}")
    print(f"Nearest neighbor class: {classes[neighbor_index]}")
    print(f"Predicted class: {predicted}")

if __name__ == "__main__":
    main()