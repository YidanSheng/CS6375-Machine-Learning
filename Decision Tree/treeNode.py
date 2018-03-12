# -*- coding: utf-8 -*-

class node:
    def __init__(self, label=-1, isLeaf =-1, right=None,left=None,majority_class = None, attribute = None):
        self.label = label #The most common value in the targetAttribute {0, 1}.
        self.isLeaf = isLeaf
        self.right=right
        self.left=left
        self.attribute = attribute
