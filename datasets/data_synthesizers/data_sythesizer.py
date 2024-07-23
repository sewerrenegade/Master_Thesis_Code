import random

from datasets.mil_dataset_abstraction import BagSizeTypes

class SinglePresenceMILSynthesizer:
    def __init__(self,postive_classes, bag_size) -> None:
        self.positive_classes = postive_classes
        self.bag_size = bag_size
        
        
    def generate_bag_level_indicies_per_class(self,training,indexer,number_of_bags_per_class):
        self.indexer = indexer
        assert all(item in indexer.classes for item in self.positive_classes)
        self.classes = indexer.classes
        self.missing_classes = [item for item in indexer.classes if item not in self.positive_classes]
        per_class_indicies = {1:[self.generate_bag_instance_from_label(1,training) for _ in range(int(number_of_bags_per_class) )],0:[self.generate_bag_instance_from_label(0,training) for _ in range(number_of_bags_per_class)]}    
        return per_class_indicies
        
    def generate_bag_instance_from_label(self,label,training):
        assert self.bag_size[0] == BagSizeTypes.CONSTANT
        if label:
            instance_labels = random.choices(self.classes, k=self.bag_size[1])
            forced_index = random.randint(0, self.bag_size[1] - 1)
            instance_labels[forced_index] = random.choice(self.positive_classes)
        else:
            instance_labels = random.choices(self.missing_classes, k=self.bag_size[1])
        return self.indexer.get_random_samples_of_class(instance_labels,training)
    
class MultiPresenceMILSynthesizer:
    def __init__(self,classes_of_significance, bag_size) -> None:
        self.classes_of_significance = classes_of_significance
        self.bag_size = bag_size

    
    def generate_bag_level_indicies_per_class(self,training,indexer,number_of_bags):
        self.indexer = indexer
        assert all(item in indexer.classes for item in self.classes_of_significance)
        self.classes = list(range(2**self.classes_of_significance))
        self.unimportant_classes = [item for item in indexer.classes if item not in self.positive_classes]
        self.per_class_count = [int(number_of_bags/len(self.classes))]*(len(self.classes)-1) + [number_of_bags -int(number_of_bags/len(self.classes))*(len(self.classes)-1)]
        per_class_indicies = {}
        for class_index, class_count in enumerate(self.per_class_count):
            per_class_indicies[self.classes[class_index]] = [self.generate_bag_instance_from_label(self.classes[class_index],training) for _ in range(class_count)] 
        return per_class_indicies
        
    def generate_bag_instance_from_label(self,label,training):
        assert isinstance(self.bag_size,int)
        presence_list = self.integer_to_binary_list(label)
        raise NotImplementedError()

    def integer_to_binary_list(self,number):
        # Convert number to binary and remove the '0b' prefix
        binary_representation = bin(number)[2:]
        
        # Pad the binary representation with leading zeros to match the desired length
        padded_binary_list = [int(digit) for digit in binary_representation.zfill(2**self.classes_of_significance)]
        
        # Ensure the list is exactly the desired length
        if len(padded_binary_list) > 2**self.classes_of_significance:
            raise ValueError("The length of the binary representation exceeds the specified length")
        
        return padded_binary_list[::-1]
    
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