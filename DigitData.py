from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import mxnet as mx


class DigitData:
    #constructeur de la classe avec emplacement de fichier comme argument
    def __init__(self, path):
        self.path = path
        # ouvrir le fichier de données CSV et le mettre en mémoire dans une matrice de numpy
        self.train_data = np.loadtxt(open( self.path,"rb"),dtype='int', delimiter=",")
        # étant spécifier dans l’énonce que les donnes du Test sont dans un fichier séparé que ce de l’entraînement,
        # alors qu'il y avait qu'un seul fichier de données dans les ressources,
        # j'ai récupéré les 10 000 données du TEST du module keras  
        (_,_), (self.in_test, self.out_test) = mnist.load_data()


    # préparation de données 
    def perpareData(self):
        # les données du test étant récupéré de keras.dataset, il sont d'une autre forme que celle qu'on utilise comme entrée dans les perceptrons multi-couches
        # on change la forme de [nbre_images][width][height] =====> [nbre_images][width*height],
        #il s'agit de la même forme que dans le fichier CSV 
        self.in_test = self.in_test.reshape(self.in_test.shape[0], 28*28).astype('float32')
        # je sépare dans des tableaux de numpy les images et les chiffres resultants
        train_input = np.delete(self.train_data, np.s_[0], 1)
        train_output = np.delete(self.train_data,np.s_[1:],1)
        # je divise ma base d’entraînement en 2 pour créer une base de validation ~ 10 400
        self.in_train, self.in_validation, self.out_train, self.out_validation = train_test_split(train_input, train_output, test_size=0.166, random_state=42)
        # puisque il s'agit d'un probleme de classification, alors on classe les chiffres  par categories , ca s'appelle aussi "one hot encoding" 
        self.out_test = np_utils.to_categorical(self.out_test)
        self.out_train = np_utils.to_categorical(self.out_train)
        self.out_validation = np_utils.to_categorical(self.out_validation)
        #nbre de classe puisque on a seulement des chiffres de 0->9,  le resultat c'est 10
        self.num_categories = self.out_validation.shape[1]
        # nombre d'input 28*28
        self.num_input = np.size(self.in_train,1)


   # normalisation [0,1] d'un tableau numpy 
    @staticmethod
    def normalize(data):
        return data / 255
    # normalisation des entrées 
    def normalizeData(self):
        self.in_train = DigitData.normalize(self.in_train)
        self.in_validation = DigitData.normalize(self.in_validation)
        self.in_test = DigitData.normalize(self.in_test)
    # initialiser les iterations pour le modele mxnet
    def getIterations(self,batch_size):     
        train_iter = mx.io.NDArrayIter(self.in_train, self.out_train, batch_size, shuffle=True)
        val_iter = mx.io.NDArrayIter(self.in_validation, self.out_validation, batch_size)
        test_iter = mx.io.NDArrayIter(self.in_test, self.out_test, batch_size)
        return train_iter, val_iter, test_iter


