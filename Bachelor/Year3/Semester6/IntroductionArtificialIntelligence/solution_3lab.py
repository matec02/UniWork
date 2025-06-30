import csv
import argparse
import math

# Korištene prezentacije:
# UUUI / 10. Strojno učenje


class Feature:
    # Klasa koja služi za pohranu svih značajki u zadatku (weather, temperature,...)
    def __init__(self):
        self.features = {}

    # Metoda za parsiranje ulaznih podataka
    def parse(self, headers, values_list):
        for header in headers[:-1]:
            self.features[header] = []
        for value_list in values_list:
            for header, value in zip(headers, value_list[:-1]):
                if value not in self.features[header]:
                    self.features[header].append(value)


class Label:
    # Klasa koja služi za pohranu oznaka u zadatku (yes, no,...)
    def __init__(self):
        self.classLabel = []

    # Metoda za parsiranje ulaznih podataka
    def parse(self, values_list):

        for value_list in values_list:
            if value_list[-1] not in self.classLabel:
                self.classLabel.append(value_list[-1])


class Dataset:
    # Klasa za pohranu cijelog skupa podataka koji se koristi u zadatku
    def __init__(self, path):
        self.path = path
        self.feature = Feature()
        self.label = Label()
        self.table = []

    # Metoda za parsiranje ulaznih podataka
    def parse(self):
        with open(self.path, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            values_list = list(csv_reader)
        self.table = values_list
        self.feature.parse(headers, values_list)
        self.label.parse(values_list)

    # Getteri
    def get_features(self):
        return self.feature.features

    def get_labels(self):
        return self.label.classLabel

    def get_table(self):
        return self.table


class Node:
    # Klasa koja predstavlja čvor u izgradnji stabla
    # Sadrži značajku, svoja podstabla i najčešću oznaku u tom trenutku za taj čvor
    def __init__(self, feature, subtrees, common_label):
        self.feature = feature
        self.subtrees = subtrees
        self.common_label = common_label


class Leaf:
    # Klasa koja predstavlja list
    # Sadržava samo oznaku
    def __init__(self, label):
        self.label = label


class ID3:
    # Klasa koja implementira model ID3
    def __init__(self):
        self.tree = None

    # Metoda kojoj treniramo model i ispisujemo dobivene grane
    def fit(self, dataset, max_depth=None):
        self.tree = self.id3(dataset.get_table(), dataset.get_table(), dataset.get_features(), dataset.get_labels(),
                             max_depth)
        print("[BRANCHES]:")
        self.print_tree(self.tree)

    # Metoda kojoj radimo predikciju modela
    # Ispisuje predikcije, točnost i konfucijsku matricu
    def predict(self, dataset):
        print("[PREDICTIONS]: ", end="")
        features = list(dataset.get_features().keys())
        table = dataset.get_table()
        correct = 0
        correct_labels = []
        predicted_labels = []
        for line in table:
            node = self.tree
            while isinstance(node, Node):
                index = features.index(node.feature)
                feature_value = line[index]
                if feature_value in node.subtrees:
                    node = node.subtrees[feature_value]
                else:
                    # Ako smo naišli na značajku koja se nije spominjala u treningu
                    # Odabiremo most_common_label od Node kao rješenje
                    if node.common_label == line[-1]:
                        correct += 1
                    correct_labels.append(line[-1])
                    predicted_labels.append(node.common_label)
                    print(node.common_label, end=" ")
                    break
            if isinstance(node, Leaf):
                if node.label == line[-1]:
                    correct += 1
                correct_labels.append(line[-1])
                predicted_labels.append(node.label)
                print(node.label, end=" ")

        # Ispis točnosti modela
        print()
        print(f"[ACCURACY]: {(correct / len(table)):.5f}")
        labels = dataset.get_labels()
        labels.sort()

        matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]

        # Izračun konfucijske matrice
        for i in range(len(correct_labels)):
            matrix[labels.index(correct_labels[i])][labels.index(predicted_labels[i])] += 1

        print("[CONFUSION_MATRIX]:")
        for row in matrix:
            print(' '.join(map(str, row)))

    def get_tree(self):
        return self.tree

    # Metoda za računanje najčešće oznake u tom trenutku za određeni čvor
    def argmax(self, d):
        labels_count = {}
        for value_list in d:
            if value_list[-1] not in labels_count:
                labels_count[value_list[-1]] = 1
            else:
                labels_count[value_list[-1]] += 1

        max_value = max(labels_count.values())
        keys_with_max_value = [key for key, value in labels_count.items() if value == max_value]
        return min(keys_with_max_value), max_value, labels_count

    # Metoda za izračun entropije koristeći math biblioteku
    def entropy(self, d):
        if not d:
            return 0
        _, _, count = self.argmax(d)
        total_number_of_labels = sum(count.values())
        entropy = 0
        for value in count.values():
            p_value = value / total_number_of_labels
            entropy += p_value * math.log2(p_value)
        return -entropy if entropy != 0 else entropy

    # Metoda za izračun informacijske dobiti
    def information_gain(self, d, features):
        gains = []
        for index, feature in enumerate(features):
            # Izračun ukupne entropije
            total_entropy = self.entropy(d)
            values = features[feature]
            for value in values:
                new_d = [case for case in d if case[index] == value]
                # Izračun pojedinačne entropije za vrijednost određene značajke
                specific_entropy = self.entropy(new_d)
                total_entropy -= (len(new_d) / len(d)) * specific_entropy
            gains.append(total_entropy)
            print(f"IG({feature})={total_entropy:.4f}", end=" ")
        return list(features.keys())[gains.index(max(gains))], gains.index(max(gains))

    # Metoda kojom se implementira id3 algoritam
    def id3(self, d, d_parent, features, label, max_depth, current_depth=0):
        # Ako nam je prazan skup podataka odabiremo najčešću oznaku za list (rijetko)
        if not d:
            v, _, _ = self.argmax(d_parent)
            return Leaf(v)
        v, max_value, _ = self.argmax(d)
        # Ako više nemamo značajki za obraditi ili je broj oznaka isti kao broj izjava
        # onda vraćamo tu oznaka za list (cesce)
        if not features or max_value == len(d) or (max_depth is not None and current_depth >= max_depth):
            return Leaf(v)
        # Izracun najveće informacije dobiti za značajke
        x, index = self.information_gain(d, features)
        print()
        subtrees = {}
        for feature in features[x]:
            new_d = []
            for case in d:
                # Izrada "novog" skupa podataka za sljedeću iteraciju i rekurzivni poziv
                if feature == case[index]:
                    new_case = case[:index] + case[index + 1:]
                    new_d.append(new_case)
            sub_features = {k: v for k, v in features.items() if k != x}
            t = self.id3(new_d, d, sub_features, label, max_depth, current_depth + 1)
            subtrees[feature] = t
        common_label, _, _ = self.argmax(d)
        # Ako još trebamo izvoditi algoritam vraćamo Node
        return Node(x, subtrees, common_label)

    # Metoda za formatirani ispis stabla
    def print_tree(self, node, path="", depth=1):
        if isinstance(node, Leaf):
            print(f"{path} {node.label}")
        elif isinstance(node, Node):
            for value, subtree in node.subtrees.items():
                new_path = f"{path} {depth}:{node.feature}={value}" if path else f"{depth}:{node.feature}={value}"
                self.print_tree(subtree, new_path, depth + 1)


def main():
    # Parsiranje ulaznih argumenata
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=str)
    parser.add_argument("test", type=str)
    parser.add_argument("depth", type=int, nargs='?')
    args = parser.parse_args()
    train_path = args.train
    test_path = args.test
    depth = args.depth

    # Izrada skupa podataka
    train_dataset = Dataset(train_path)
    train_dataset.parse()
    test_dataset = Dataset(test_path)
    test_dataset.parse()

    model = ID3()
    if depth:
        # Treniranje modela s maksimalnom dubinom
        model.fit(train_dataset, max_depth=depth)
    else:
        # Treniranje modela bez maksimalne dubine
        model.fit(train_dataset)

    # Predikcija modela
    model.predict(test_dataset)


if __name__ == "__main__":
    main()
