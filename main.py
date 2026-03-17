# convert data into a list of lists
def load_dataset(file_path):
    data = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if line == "":
                continue

            values = line.split()
            row = [float(value) for value in values]
            data.append(row)

    return data


def main():
    print("Welcome to Natalie's Feature Selection Algorithm.")

    file_path = input("Type in the name of the file to test: ").strip()

    data = load_dataset(file_path)

    num_instances = len(data)
    num_features = len(data[0]) - 1

    print(f"\nThis dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")


if __name__ == "__main__":
    main()