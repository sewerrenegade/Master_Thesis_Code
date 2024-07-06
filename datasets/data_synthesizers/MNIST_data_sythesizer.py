import random
from datasets.MNIST.MNIST_base import MNIST_Dataset_Referencer
import copy

class SinglePresenceSythesizer:
    def __init__(self,postive_integer, bag_size,dataset) -> None:
        self.positive_integer = postive_integer
        self. bag_size = bag_size
        self.dataset = MNIST_Dataset_Referencer
        self.missing_classes = copy.deepcopy(self.dataset.INDEXER.classes)
        self.missing_classes.remove(self.positive_integer)
        

    def generate_bag_indicies(self,label,bag_size,training):
        if label:
            classes = self.dataset.INDEXER.classes
            indices = random.choices(classes[label], k=bag_size)
            forced_index = random.randint(0, bag_size - 1)
            indices[forced_index] = self.positive_integer
        else:
            indices = random.choices(self.missing_classes, k=bag_size)
        return self.get_random_instances(indices,training)


    def get_random_instances(self,instance_class,training):
        return self.dataset.INDEXER.get_random_instance_of_class(instance_class,training)
    
    def generate_train_bags(self,number_of_train_bags):
        training_bags = []
        for bag_index in range(number_of_train_bags):
            label = SinglePresenceSythesizer.generate_label()
            training_bags.append((self.generate_bag_indicies(label,self.bag_size,True),label))
        return training_bags
    
    def generate_test_bags(self,number_of_test_bags):
        test_bags = []
        for bag_index in range(number_of_test_bags):
            label = SinglePresenceSythesizer.generate_label()
            test_bags.append((self.generate_bag_indicies(label,self.bag_size,False),label))
        return test_bags

    def generate_label():
        return random.choice([1, 0])


class DoublePresenceSythesizer:
    def __init__(self,postive_integer: tuple[int, int], bag_size,dataset) -> None:
        self.positive_integer = postive_integer
        self. bag_size = bag_size
        self.label_classes= [0,1,2,3]
        self.dataset = MNIST_Dataset_Referencer
        self.classes = self.dataset.INDEXER.classes
        self.missing_classes =[None]*4
        self.generate_possible_permutations()

    def generate_possible_permutations(self):
        for perm in self.label_classes:
            temp = self.classes.copy()
            bit_label =self.number_to_tuple_representation(perm)
            for target_index in range(2):
                if not bit_label[target_index]:
                    temp.remove(self.positive_integer[target_index])
            self.missing_classes[perm] = temp


    def generate_two_unique_random_numbers(self,end,start =0):
        # Generate the first random number
        first_number = random.randint(start, end)

        # Generate the second random number until it is different from the first one
        second_number = random.randint(start, end)
        while second_number == first_number:
            second_number = random.randint(start, end)

        return first_number, second_number
    
    def number_to_tuple_representation(self,num):
        # Convert the number to its 2-bit binary representation
        binary_repr = bin(num)[2:].zfill(2)
        
        # Create a tuple with 0s and 1s representing each bit
        tuple_representation = tuple(int(bit) for bit in binary_repr)
        
        return tuple_representation

    def generate_bag_indicies(self,label,bag_size,training):
        bit_label =self.number_to_tuple_representation(label)
        test = bit_label[0]
        if bit_label[0] and bit_label[1]:
            classes = self.dataset.INDEXER.classes
            indices = random.choices(self.missing_classes[label], k=bag_size)
            forced_index = self.generate_two_unique_random_numbers(bag_size-1)
            indices[forced_index[0]] = self.positive_integer[0]
            indices[forced_index[1]] = self.positive_integer[1]
        elif bit_label[0] and not bit_label[1]:

            indices = random.choices(self.missing_classes[label], k=bag_size)
            forced_index = random.randint(0, bag_size - 1)
            indices[forced_index] = self.positive_integer[0]
        elif not bit_label[0] and bit_label[1]:

            indices = random.choices(self.missing_classes[label], k=bag_size)
            forced_index = random.randint(0, bag_size - 1)
            indices[forced_index] = self.positive_integer[1]
        else:
            assert label == 0
            
            indices = random.choices(self.missing_classes[label], k=bag_size)# np.random.randint(0, len(self.missing_classes[label]), bag_size)


        return self.get_random_instances(indices,training)


    def get_random_instances(self,instance_class,training):
        return self.dataset.INDEXER.get_random_instance_of_class(instance_class,training)
    
    def generate_train_bags(self,number_of_train_bags):
        training_bags = []
        for bag_index in range(number_of_train_bags):
            label = self.generate_label()
            training_bags.append((self.generate_bag_indicies(label,self.bag_size,True),label))
        return training_bags
    
    def generate_test_bags(self,number_of_test_bags):
        test_bags = []
        for bag_index in range(number_of_test_bags):
            label = self.generate_label()
            test_bags.append((self.generate_bag_indicies(label,self.bag_size,False),label))
        return test_bags

    def generate_label(self):
        return random.choice(self.label_classes)