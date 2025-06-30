import argparse
import csv
import numpy as np

# Korištene prezentacije:
# UUUI / 11. Umjetne neuronske mreže
# UUUI / 12. Prirodom inspiracijski optimizacijski algoritmi


class Dataset:
    # Klasa koja sadrži ulazne podatke i njihove izlaze (x,y)
    def __init__(self, path):
        self.path = path
        self.input = None
        self.output = None

    # Metoda koja služi za parsiranje ulazne datoteke
    def parse(self):
        with open(self.path, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            values_list = list(csv_reader)

        self.input = np.empty((len(values_list), len(headers[:-1])))
        self.output = np.empty((len(values_list), 1))

        for i, row in enumerate(values_list):
            self.input[i] = np.array(row[:-1], dtype=float)
            self.output[i] = float(row[-1])

    # Getteri
    def get_input(self):
        return self.input

    def get_output(self):
        return self.output


# Funkcija za racunanje funkcije greske - srednja kvadratna pogreska
def mean_squared_error(true_outputs, generated_outputs):
    diff = true_outputs - generated_outputs
    square_diff = np.square(diff)
    mse = np.mean(square_diff)
    return mse

# Funkcija za racunanje prijenosne funkcije - sigmoida
def sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))


class OneLayerSigmoidNN:
    # Klasa za jednoslojnu neuronsku mrezu - 5s, 20s
    def __init__(self, dataset, hidden_layer_number):
        self.dataset = dataset
        self.hidden_layer_number = hidden_layer_number
        input_size = dataset.get_input().shape[1]
        output_size = 1
        # Inicijalizacija parametara mreze - tezine i bias
        self.weights_1 = np.random.normal(0, 0.01, (input_size, self.hidden_layer_number))
        self.bias_1 = np.random.normal(0, 0.01, (1, self.hidden_layer_number))
        self.weights_2 = np.random.normal(0, 0.001, (self.hidden_layer_number, output_size))
        self.bias_2 = np.random.normal(0, 0.01, (1, output_size))

    # Metoda za forward_pass NN koja takoder racuna gresku
    def compute_loss(self, inputs, true_outputs):
        #Skriveni sloj
        hidden_layer_1 = np.matmul(inputs, self.weights_1) + self.bias_1
        sigmoid_values = sigmoid(hidden_layer_1)
        # Izlaz
        generated_output = np.matmul(sigmoid_values, self.weights_2) + self.bias_2
        # Racunanje greske
        return mean_squared_error(true_outputs, generated_output)

    # Getteri
    def get_weights(self):
        return [self.weights_1, self.weights_2]

    def get_biases(self):
        return [self.bias_1, self.bias_2]

    # Setteri
    def set_weights(self, weights):
        self.weights_1 = weights[0]
        self.weights_2 = weights[1]

    def set_biases(self, biases):
        self.bias_1 = biases[0]
        self.bias_2 = biases[1]


class TwoLayerSigmoidNN:
    # Klasa za dvoslojnu neuronsku mrezu - 5s5s
    def __init__(self, dataset, hidden_layer_number):
        self.dataset = dataset
        self.hidden_layer_number = hidden_layer_number
        input_size = dataset.get_input().shape[1]
        output_size = 1

        self.weights_1 = np.random.normal(0, 0.01, (input_size, self.hidden_layer_number))
        self.bias_1 = np.random.normal(0, 0.01, (1, self.hidden_layer_number))
        self.weights_2 = np.random.normal(0, 0.001, (self.hidden_layer_number, self.hidden_layer_number))
        self.bias_2 = np.random.normal(0, 0.01, (1, self.hidden_layer_number))
        self.weights_3 = np.random.normal(0, 0.001, (self.hidden_layer_number, output_size))
        self.bias_3 = np.random.normal(0, 0.01, (1, output_size))

    # Metoda za forward_pass NN koja takoder racuna gresku
    def compute_loss(self, inputs, true_outputs):
        # Prvi skriveni sloj
        hidden_layer_1 = np.matmul(inputs, self.weights_1) + self.bias_1
        sigmoid_values = sigmoid(hidden_layer_1)

        # Drugi skriveni sloj
        hidden_layer_2 = np.matmul(sigmoid_values, self.weights_2) + self.bias_2
        sigmoid_values_2 = sigmoid(hidden_layer_2)

        # Izlaz
        generated_output = np.matmul(sigmoid_values_2, self.weights_3) + self.bias_3

        # Izracun greske
        return mean_squared_error(true_outputs, generated_output)

    # Getteri
    def get_weights(self):
        return [self.weights_1, self.weights_2, self.weights_3]

    def get_biases(self):
        return [self.bias_1, self.bias_2, self.bias_3]

    # Setteri
    def set_weights(self, weights):
        self.weights_1 = weights[0]
        self.weights_2 = weights[1]
        self.weights_3 = weights[2]

    def set_biases(self, biases):
        self.bias_1 = biases[0]
        self.bias_2 = biases[1]
        self.bias_3 = biases[2]


# Funkcija za krizanje roditelja
def cross_parents(p1, p2, nn_class, dataset, hidden_layer_number):
    weights_p1 = p1.get_weights()
    weights_p2 = p2.get_weights()
    bias_p1 = p1.get_biases()
    bias_p2 = p2.get_biases()
    # Roditelju su krizaju po bazi aritmeticke sredine
    new_weights = [(w1 + w2) / 2.0 for w1, w2 in zip(weights_p1, weights_p2)]
    new_biases = [(b1 + b2) / 2.0 for b1, b2 in zip(bias_p1, bias_p2)]
    child = nn_class(dataset, hidden_layer_number)
    child.set_weights(new_weights)
    child.set_biases(new_biases)
    return child


class GeneticAlgorithm:
    # Klasa koje generira genetski algoritam
    def __init__(self, pop_size, elitism_number, mutation_probability, std_gauss, iteration_count):
        self.pop_size = pop_size
        self.elitism_number = elitism_number
        self.mutation_probability = mutation_probability
        self.std_gauss = std_gauss
        self.iteration_count = iteration_count
        self.population = None
        self.best_pop = None

    # Metoda za inicijalizaciju pocetne populaacije
    def initialize_population(self, nn_class, dataset, hidden_layer_number):
        self.population = [nn_class(dataset, hidden_layer_number) for _ in range(self.pop_size)]
        return self.population

    # Metoda za izracun dobrote za populaciju
    def evaluate_population(self, dataset):
        fitness_scores = np.empty(self.pop_size)
        for i, pop in enumerate(self.population):
            # Racuna se kao 1 / MSE jer sto je populacija bolja MSE je manji
            fitness_scores[i] = 1 / pop.compute_loss(dataset.get_input(),
                                                     dataset.get_output())
        return fitness_scores

    # Metoda za odabir najbolje jedinke populacije
    def get_best_population(self, dataset):
        scores = self.evaluate_population(dataset)
        max_index = np.argmax(scores)
        return self.population[max_index]

    # Metoda koja sluzi za odrzavanje elitizma -> u novu populaciju dodaje n najboljih jedinki
    def elitism(self, dataset):
        scores = self.evaluate_population(dataset)
        elite_indices = np.argsort(scores)[-self.elitism_number:]
        new_population = [self.population[i] for i in elite_indices]
        return new_population

    # Metoda za odabir roditelja
    def choose_parents(self, dataset):
        scores = self.evaluate_population(dataset)
        sum_fitness = np.sum(scores)
        # Iznosi dobrote za svaku jedinku
        fitness_proportions = np.array(scores) / sum_fitness
        # Distribucija od 0 do 1 za iznose dobrote
        distribution = np.cumsum(fitness_proportions)
        parents = []
        while len(parents) < 2:
            # Odabir roditelje nasumicnim brojem od 0 do 1
            r = np.random.rand()
            for i, value in enumerate(distribution):
                if r <= value:
                    parent = self.population[i]
                    if parent not in parents:
                        parents.append(self.population[i])
                    break
        return parents[0], parents[1]

    # Metoda za mutaciju djeteta dodavanjem broja po odredenoj vjerojatnosti iz Gaussove distribucije za odredeni STD
    def mutate_child(self, child):
        weights_child = child.get_weights()
        biases_child = child.get_biases()
        for weight in weights_child:
            # Inicijalizacija maske koja je dimenzija kao matrica tezina
            # Sadrzi True, False ovisno je li zadovoljena vjerojatnost
            mutation_mask = np.random.rand(*weight.shape) < self.mutation_probability
            # Mutacija odnosno dodavanje broja iz Gaussove distribucije ako je u maski mutaciju True vrijednost
            # Ako je False ne mijenja se tezina
            weight += mutation_mask * np.random.normal(0, self.std_gauss, weight.shape)
        # Isti princip se primjenjuje za bias-e
        for bias in biases_child:
            mutation_mask = np.random.rand(*bias.shape) < self.mutation_probability
            bias += mutation_mask * np.random.normal(0, self.std_gauss, bias.shape)
        child.set_weights(weights_child)
        child.set_biases(biases_child)

    # Metoda koja implementira genetski algoritam/treniranje
    def train(self, nn_class, dataset, hidden_layer_number):
        # Inicijalizacija pocente populacije
        self.population = self.initialize_population(nn_class, dataset, hidden_layer_number)
        # Prolaz kroz odredeni broj iteracija
        for i in range(1, self.iteration_count+1):
            # Inicijalizacija pocetne populacije s elitizmom
            new_population = self.elitism(dataset)
            # Dodavanje jedinki u novu populaciju
            while len(new_population) < self.pop_size:
                # Odabir roditelja
                p1, p2 = self.choose_parents(dataset)
                # Krizanje roditelja
                child = cross_parents(p1, p2, nn_class, dataset, hidden_layer_number)
                # Mutacija djeteta
                self.mutate_child(child)
                new_population.append(child)
            # Stara populacija postaje nova
            self.population = new_population
            if i % 2000 == 0:
                # Pronalazak najbolje generacije i ispivanje njezine greske
                self.best_pop = self.get_best_population(dataset)
                train_loss = self.best_pop.compute_loss(dataset.get_input(), dataset.get_output())
                print(f"[Train error @{i}]: {train_loss}")
        return

    # Provjera na testnom skupu te izracun i ispis greske
    def test(self, dataset):
        test_loss = self.best_pop.compute_loss(dataset.get_input(), dataset.get_output())
        print(f"[Test error]: {test_loss}")


def main():
    # Parsiranje ulazne linije
    parser = argparse.ArgumentParser()
    # Svi obavezni argumenti
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--nn', type=str, required=True)
    parser.add_argument('--popsize', type=str, required=True)
    parser.add_argument('--elitism', type=str, required=True)
    parser.add_argument('--p', type=str, required=True)
    parser.add_argument('--K', type=str, required=True)
    parser.add_argument('--iter', type=str, required=True)
    args = parser.parse_args()

    train_dataset_path = args.train
    test_dataset_path = args.test
    nn_type = args.nn
    population_size = args.popsize
    elitism_number = args.elitism
    mutation_probability = args.p
    std_gauss = args.K
    iteration_count = args.iter

    # Inicijalizacija skupa za treniranje i testiranje
    train_dataset = Dataset(train_dataset_path)
    train_dataset.parse()
    test_dataset = Dataset(test_dataset_path)
    test_dataset.parse()

    # Inicijalizacija genetskog algoritma
    ga = GeneticAlgorithm(int(population_size), int(elitism_number),
                          float(mutation_probability), float(std_gauss), int(iteration_count))

    # Provodenje genetskog algoritma za odredenu mrezu
    if nn_type == "5s" or nn_type == "20s":
        ga.train(OneLayerSigmoidNN, train_dataset, int(nn_type[:-1]))
    elif nn_type == "5s5s":
        ga.train(TwoLayerSigmoidNN, train_dataset, int(nn_type[0]))

    # Testiranje
    ga.test(test_dataset)

    return


if __name__ == "__main__":
    main()
