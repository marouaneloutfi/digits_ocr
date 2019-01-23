from keras.models import Sequential
from keras.layers import Dense
import mxnet as mx



class MultiLayerModel:

    # constructeur de la class
    def __init__( self, optimiser, learning_rate,input_number, hidden_layers = [], output_layer = [] ):
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.input_number = input_number

    # Construction dynamique du modele Keras de type perceptron multi-couche
    # a construction dynamique rend facile a expérimenter avec des différents architectures et paramètres afin de trouver le modèle avec la meilleur précision  
    def keras_mlp(self):
        model = Sequential()
        # ajout de la 1ere couche
        model.add(Dense(self.hidden_layers[0][0], input_dim=self.input_number, kernel_initializer='normal', activation=self.hidden_layers[0][1]))
        # on ajoute les couches selon le nombre des couche cachés dans les parametres
        #chaque couche est definit par le nombre de neurons layer[0] et sa fonction d'activation layer[1] 
        for layer in self.hidden_layers[1:]:
            model.add(Dense(layer[0], kernel_initializer='normal', activation=layer[1]))
        #ajout de la derniere couche: couche de sortie 
        model.add(Dense(self.output_layer[0], kernel_initializer='normal', activation = self.output_layer[1]))
        #compilation du  modele avec un des algorithmes de retro-propagation ( adam optimiser, gradient descent ...)
        model.compile(loss='categorical_crossentropy', optimizer = self.optimiser , metrics=['accuracy'])
        return model

      # Construction dynamique du modele Mxnet de type perceptron multi-couche
    def Mx_mlp(self):
        # declaration d'un conteneur pour les donnees d'entrees (dummy)  
        data = mx.sym.var('data')
        #pour Construire un modèle sous mxnet, on déclare d'une manière séquentielle pour chaque couche, le nombre de neurones et la fonction
        #d'activation, en donnant comme argument la couche qui précède
        all_layer = []
        all_layer.append(mx.sym.FullyConnected(data=data, num_hidden= self.hidden_layers[0][0]))
        all_layer.append( mx.sym.Activation(data=all_layer[0], act_type = self.hidden_layers[0][1]) )
        
        for layer in self.hidden_layers[1:]:
            all_layer.append( mx.sym.FullyConnected(data = all_layer[len(all_layer) - 1], num_hidden = layer[0]))
            all_layer.append( mx.sym.Activation(data =all_layer[len(all_layer) - 1], act_type = layer[1]) )

        all_layer.append( mx.sym.FullyConnected(data=all_layer[len(all_layer) - 1], num_hidden = self.output_layer[0]))
        mlp =  mx.sym.SoftmaxOutput(data = all_layer[len(all_layer) - 1], name = self.output_layer[1])
        return mlp



       
            
            
            
