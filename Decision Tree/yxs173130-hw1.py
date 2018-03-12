# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# Anthor: Yidan Sheng
import csv
import copy
import random
from math import log
from treeNode import node
from collections import deque
from collections import Counter
import sys
class ID3:

    """Initial the ID3 class.
    Args: Training dataSet, attribute type in dataset.
    """
    def __init__(self, data, fixed_classList):
        self.data = data
        self.fixed_classList = fixed_classList

    """Get the majority of value within the subset/whole set.
    Args: Subset of sample data.
    Returns: The most common value.
    """
    def majorityCnt(self, dataSet):
        #Get all the class value from the data set.
        classList = [row[-1] for row in dataSet]
        classCount = {}

        #Count the number of 0's and 1's respectively.
        for tag in classList:
            if tag not in classCount.keys():classCount[tag] = 0
            classCount[tag] += 1

        #Compare them and get the most common value.
        valueCount = -1
        commonValue = -1
        for key in classCount.keys():
            if classCount[key] > valueCount:
                valueCount = classCount[key]
                commonValue = key
        return commonValue

    """Caculate the variance impurity.
    Args: Subset of sample data.
    Returns: Variance impurity.
    """
    def varianceImpurity(self, dataSet):
        classList = [row[-1] for row in dataSet]
        results = Counter(classList)

        #Compute the variance impurity by formula vi = (k0/k) * (k1/k)
        if len(results) < 2:
            return 0
        else:
            vi = 1.0
            for i in results:
                prob = results[i]/len(classList)
                vi *= prob
            return vi

    """Select the best feature by variance impurity.
    Args: Subset of sample data, attribute types.
    Returns: Largest varianceImpurity value, best feature attribute, best subset of best feature.
    """
    def bestFeaturebyVarImpurity(self, dataSet, dynamic_classList):
        best_gain = 0.0
        best_feature = ''
        best_set = {}
        numEntries = len(dataSet)
        base_varImpurity = self.varianceImpurity(dataSet)
        new_varImpurity = 0.0
        #Compute the information gain by formula.
        for element in dynamic_classList:
            (negativeSet, positiveSet) = self.divideSet(dataSet, element)
            prob = float(len(positiveSet))/len(dataSet)
            new_varImpurity = prob * self.varianceImpurity(positiveSet) + (1 - prob)*self.varianceImpurity(negativeSet)
            gain = base_varImpurity - new_varImpurity

            #Choose the best feature as next attribute.
            if gain > best_gain:
                best_gain = gain
                best_feature = element

        return best_gain, best_feature

    """Caculate the entropy.
    Args: Subset of sample data.
    Returns: Entropy.
    """
    def entropy(self, dataSet):
        classList = [row[-1] for row in dataSet]
        result = Counter(classList)

        #Compute the entropy by formula entropy = - p1log2p1 - p0log2p2
        entropy = 0.0
        for i in result:
            prob = result[i]/len(classList)
            entropy -= prob * log(prob,2)
        return entropy

    """Select the best feature by information gain.
    Args: Subset of sample data, attribute types.
    Returns: Largest information gain value, best feature attribute, best subset of best feature.
    """
    def bestFeaturebyInfoGain(self, dataSet, dynamic_classList):
        best_gain = 0.0
        best_feature = ''
        best_set = {}
        base_entropy = self.entropy(dataSet)
        new_entropy = 0.0

        #Compute the information gain by formula.
        for element in dynamic_classList:
            (negativeSet, positiveSet) = self.divideSet(dataSet, element)
            prob = float(len(positiveSet))/len(dataSet)
            new_entropy = prob * self.entropy(positiveSet) + (1 - prob)*self.entropy(negativeSet)
            gain = base_entropy - new_entropy

            #Choose the best feature as next attribute.
            if gain > best_gain:
                best_gain = gain
                best_feature = element

        return best_gain, best_feature

    """Split instances into positive set(all 1's) and negtive set(all 0's).
    Args: Subset of sample data, attribute.
    Returns: Set included all 1's, set included all 0's.
    """
    def divideSet(self, dataSet, element):
        #Find the position of element in the all classes.
        i = self.fixed_classList.index(element)
        negativeSet=[]
        positiveSet=[]

        #Spilt into two subsets which the one contains all 0's and the other contains all 1's.
        for row in dataSet:
            if row[i] == 0:
                negativeSet.append(row)
            else:
                positiveSet.append(row)
        return (negativeSet, positiveSet)

    """Create tree by method which choosing largest information gain.
    Args: Subset of sample data, attribute.
    Returns: Current root.
    """
    def createTree(self, dataSet, dynamic_classList):
        if(len(dataSet) == 0):
            return None

        root = node()
        classList = [row[-1] for row in dataSet]

        #If all values of current attribute of root is 0 or 1, seems it as leaf.
        if classList.count(classList[0]) == len(classList):
            root.label = classList[0]
            root.isLeaf = 1
            root.left = None
            root.right = None
            return root

        #Store the most common value to label of root.
        root.label = self.majorityCnt(dataSet)
        entropy = self.entropy(dataSet)
        #All attributes have selected.
        if len(dynamic_classList) == 0 or entropy == 0:
            root.isLeaf = 1
            root.left = None
            root.right = None
            return root

        #Choose the Information gain, best feature and divided sets.
        best_gain, best_feature = self.bestFeaturebyInfoGain(dataSet, dynamic_classList)

        #Information gain is supposed to be more or equal than 0.
        if best_gain > 0:
            #Given the best attribute for node.
            root.attribute = best_feature
            #Remove the selected attibute, then continue to choose next attribute, build tree etc.
            dynamic_classList.remove(best_feature)
            sub_classfier1 = copy.deepcopy(dynamic_classList)
            sub_classfier2 = copy.deepcopy(dynamic_classList)
            dataSet_0, dataSet_1 = self.divideSet(dataSet, best_feature)
            root.left = self.createTree(dataSet_0, sub_classfier1)
            root.right = self.createTree(dataSet_1, sub_classfier2)

        return root

    """Create tree by method which choosing largest variance impurity.
    Args: Subset of sample data, attribute.
    Returns: Current root.
    """
    def createTree2(self, dataSet, dynamic_classList):
        if(len(dataSet) == 0):
            return None

        root = node()
        classList = [row[-1] for row in dataSet]

        #If all values of current attribute of root is 0 or 1, seems it as leaf.
        if classList.count(classList[0]) == len(classList):
            root.label = classList[0]
            root.leaf = 1
            root.left = None
            root.right = None
            return root

        #Store the most common value to label of root.
        root.label = self.majorityCnt(dataSet)

        #All attributes have selected.
        if len(dynamic_classList) == 0:
            root.leaf = 1
            root.left = None
            root.right = None
            return root

        #Choose the Information gain, best feature and divided sets.
        best_gain, best_feature = self.bestFeaturebyVarImpurity(dataSet, dynamic_classList)

        #Information gain is supposed to be more or equal than 0.
        if best_gain > 0:
            #Given the best attribute for node.
            root.attribute = best_feature
            #Remove the selected attibute, then continue to choose next attribute, build tree etc.
            dynamic_classList.remove(best_feature)
            sub_classfier1 = copy.deepcopy(dynamic_classList)
            sub_classfier2 = copy.deepcopy(dynamic_classList)
            dataSet_0, dataSet_1 = self.divideSet(dataSet, best_feature)
            root.left = self.createTree2(dataSet_0, sub_classfier1)
            root.right = self.createTree2(dataSet_1, sub_classfier2)

        return root

    """Print the decision tree.
    Args: Subset of sample data, level.
    Returns: The string representation of decision tree.
    """
    def printTree(self, root, level):
        string = ''
        if root == None:
            return ''

        #If meets the leaf node, print the label of the node.
        if root.left == None and root.right == None:
            string += str(root.label) + '\n'
            return string

        #Calculate the level of tree.
        levelBars = ''
        for i in range(0, level):
            levelBars += '|'
        string += levelBars

        #When the children of left root is none, put 0 as the condition. Otherwise, go to the next level.
        if root.left!= None and root.left.left == None and root.left.right == None:
            string +=  str(root.attribute) + " = 0 : "
        else:
            string +=  str(root.attribute) + " = 0 :\n"

        #Search for next level of the tree.
        string += self.printTree(root.left, level + 1)
        string += levelBars

        #When the children of right root is none, put 1 as the condition. Otherwise, go to the next level.
        if root.left != None and root.right.left == None and root.right.right == None:
            string += str(root.attribute) + " = 1 :"
        else:
            string += str(root.attribute) + " = 1 :\n"

        #Search for next level of the tree.
        string += self.printTree(root.right, level + 1)

        return string

    """Predict the class type by given instances.
    Args: Row, current root.
    Returns: Call the next level recursively.
    """
    def getPredictedValue(self, row, root):
        if root.right == None and root.left == None:
            return root.label

        #Find the position of element in the all classes.
        i = self.fixed_classList.index(root.attribute)
        #Find the target for a hypothesis by recursive calls to search into left/right subtrees.
        if row[i] == 0:
            return self.getPredictedValue(row, root.left)
        else:
            return self.getPredictedValue(row, root.right)

    """Calculate the accuracy for the test dataset.
    Args: Test dataset, root.
    Returns: Accuracy.
    """
    def calculateAccuracy(self, data, root):
        if root == None or len(data) == 0:
            return 0
        classCount = 0
        target = [row[-1] for row in data]
        i = 0

        #Count the number of correct result, and calculate the accuracy.
        for row in data:
            if int(self.getPredictedValue(row, root)) == int(target[i]):
                classCount +=1
            i += 1
        accuracy = float(classCount)/len(target)
        return accuracy

    """Pruning tree by replacing random node with majority class.
    Args: Test dataset, root, L, K.
    Returns: Best accuracy of root.
    """
    def post_pruning(self, data, root, l, k):
        # Let the best decision tree best_root be the current decision tree
        best_root = copy.deepcopy(root)
        for i in range(1, l):
            # Copy tree into a new tree current_root
            current_root = copy.deepcopy(root)
            # A random number between 1 and K.
            m = random.randint(1, k)
            for j in range(1, m):
                # Order the nodes in current_root from 1 to N.
                ordered_node = self.preorder(current_root)
                # Let N denote the number of non-leaf nodes in the decision tree current_root.
                n = len(ordered_node) - 1
                # Terminate pruning if meets the leaf node in the tree
                if n <= 0:
                    return best_root
                # Let P be a random number between 1 and N.
                p = random.randint(1, n)
                # Replace the subtree rooted at node P in current_root by a leaf node
                # Assign the majority class of the subset of the data at P
                # to the leaf node.
                replacednode = ordered_node[p]
                replacednode.isLeaf = 1
                replacednode.left = None
                replacednode.right = None
            # Evaluate the accuracy of current_root on validation set
            oldAccurancy = self.calculateAccuracy(data, best_root)
            newAccurancy = self.calculateAccuracy(data, current_root)
            # If current_root is more accurate than best_root, replace best_root by current_root
            if(newAccurancy > oldAccurancy):
                best_root = current_root
        return best_root

    """Pre-order the decision tree.
    Args: Root of tree/subtree.
    Returns: The list of ordered treenode in the decision tree.
    """
    def preorder(self,root):
        order = []
        if root == None or root.isLeaf == 1:
            return root

        #Store all node into deque by preorder traversal.
        queue = deque([root])
        while len(queue) > 0:
            curr_root = queue.popleft()
            #Preorder traversal.
            #Traverse the root first, then left of the root, finally the right of the tree.
            order.append(curr_root)
            if curr_root.left!= None and curr_root.left.isLeaf == -1:
                queue.append(curr_root.left)
            if curr_root.right!= None and curr_root.right.isLeaf == -1:
                queue.append(curr_root.right)
        return order

if __name__ == "__main__":
    L = int(sys.argv[1])
    K = int(sys.argv[2])
    training_file_path = sys.argv[3]
    validation_file_path = sys.argv[4]
    test_file_path = sys.argv[5]
    willPrint = sys.argv[6]

    with open(training_file_path) as f:
        reader = csv.reader(f)
        data = list(reader)
    classTypeList = data[0][:-1]
    fixed_classList = copy.deepcopy(classTypeList)
    data = data[1:]
    dataset = []
    dataset2 = []
    for row in data:
        dataset.append([int(i) for i in row])
        dataset2.append([int(i) for i in row])

    id3 = ID3(dataset, fixed_classList)
    dynamic_classList = copy.deepcopy(classTypeList)
    dynamic_classList2 = copy.deepcopy(classTypeList)
    root = id3.createTree(dataset, dynamic_classList)
    root2 = id3.createTree2(dataset2,dynamic_classList2)

    with open(validation_file_path) as f:
        reader = csv.reader(f)
        data = list(reader)
    classTypeList = data[0][:-1]

    data = data[1:]
    dataset = []
    for row in data:
        dataset.append([int(i) for i in row])

    with open(test_file_path) as f:
        test_reader = csv.reader(f)
        test_data = list(test_reader)
    test_data = test_data[1:]
    test_dataset = []
    for row in test_data:
        test_dataset.append([int(i) for i in row])

    print ('Before post_pruning, the accuracy of the decision tree by calculating entropy is: ')
    print(id3.calculateAccuracy(test_dataset, root))
    print ('Before post_pruning, the accuracy of the decision tree by calculating variance impurity is: ')
    print(id3.calculateAccuracy(test_dataset, root2))

    bestroot = id3.post_pruning(dataset, root, L, K)
    bestroot2 = id3.post_pruning(dataset,root2,L, K)
    print ('After post_pruning, the accuracy of the decision tree by calculating entropy is: ')
    print(id3.calculateAccuracy(test_dataset, bestroot))
    print ('After post_pruning, the accuracy of the decision tree by calculating variance impurity is:')
    print(id3.calculateAccuracy(test_dataset, bestroot2))
    if willPrint == 'yes':
        print('Before post_pruning, the tree by calculating entropy is:')
        print(id3.printTree(root, 0))
        print('Before post_pruning, the tree by calculating variance impurity is:')
        print(id3.printTree(root2, 0))
        print('After post_pruning, the tree by calculating entropy is:')
        print(id3.printTree(bestroot, 0))
        print('After post_pruning, the tree by calculating variance impurity is:')
        print(id3.printTree(bestroot2, 0))
