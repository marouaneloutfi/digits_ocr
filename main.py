import sys
import os
import mxnet as mx
import logging
import argparse

# on redirige la sortie  stderr vers null pour ne pas afficher le message "Using tensorflow as backend" lorsqu' on charge les modules de Keras
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from DigitData import DigitData
from MultiLayerModel import MultiLayerModel
#redirection vers l'ecran 
sys.stderr = stderr

# On utilise le module argparse pour recuperer les arguments de la ligne de la commande
# et aussi pour afficher de l'aide avec  --help
parser = argparse.ArgumentParser()
parser.add_argument("--platform", help="pick a deeplearning framework: keras | mxnet")
parser.add_argument("--file", help="location of CSV file")
args = parser.parse_args()
# sélection du platforme de deeplearning choisis, l'utilisateur a le choix entre Keras sous tensorflow ou bien Mxnet
# si, l'utilisateur n'a rien spécifié, par défaut c'est keras
platform = args.platform
if platform == '':
    platform = 'keras'
# emplacement du fichier CSV contenant les données d'entrainement  MNIST
path = args.file



# on fixe la graine pour reproduire le meme resultat lors des executions multiples 
seed = 42

# le nombre d'image a traiter pour chaque iteration, c'est a dire,
# la modification des poids ne se fait qu'une seule fois apres chaque batch
batch_size = 200
# initialiser le context de calcul pour MXNET
# On favorise la puissance du  traitement sur GPU pour minimiser le temps d'execution,
# si ce n'est pas possible on utilise les CPUs
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# initialisation de données avec le fichier d'entrainement CSV
print("preparing Data...")
mydata = DigitData(path)
# preparation du données
mydata.perpareData()
# normalisation du données
mydata.normalizeData()

# Conception de la Topologie du Reseau NN et declaration des parametres d'apprentissage 
hidden_layers = [[784,'relu'],[100,'relu']]
output_layer = [mydata.num_categories,'softmax']
mlp = MultiLayerModel( 'adam', 0.1, mydata.num_input, hidden_layers, output_layer )

    
# choix de Keras comme platforme
if platform == 'keras':
    print("Using deep learning platform keras under tensorflow")
    #Construction du modele 
    model = mlp.keras_mlp()
    # entrainement du model sur la base Train en utilisant la base de validation pour eviter le sur ajustement (overfitting)
    model.fit(mydata.in_train, mydata.out_train, validation_data=(mydata.in_validation, mydata.out_validation), epochs=10, batch_size = 200, verbose=2)
    # Evaluation finale du model sur la  base de test
    scores = model.evaluate(mydata.in_test, mydata.out_test, verbose = 0)
    print("la precision sur la base du test: %.2f%%" % (scores[1]*100))

# choix de mxnet comme platforme
elif platform == 'mxnet':
    print("Using deep learning platform Mxnet")
    # initialiser la graine
    mx.random.seed(seed)
    # intialiser les compteurs (iteration) de mes donnees d'entrainment, de test, et de validations 
    train_iter, val_iter, test_iter = mydata.getIterations(200)
    #Construction du modele et l'attacher au contexte du calcul
    mlp = mlp.Mx_mlp()
    mlp_model = mx.mod.Module(symbol=mlp, context=ctx)
    # par defaut la fonction fit du model mxnet n'affiche pas les resultats d'entrainement directement sur l'ecran
    # une solution s'agit d'afficher le log en niveau "debug" directement sur l'ecran
    logging.getLogger().setLevel(logging.DEBUG)  
    # entraînement du model sur la base Train en utilisant la base de validation pour eviter le sur ajustement (overfitting)
    mlp_model.fit(train_iter,
                  eval_data = val_iter,
                  optimizer=  'adam', 
                  optimizer_params = {'learning_rate': 0.1}, 
                  eval_metric='acc',  # afficher la  precision 
                  batch_end_callback = mx.callback.Speedometer(batch_size, batch_size), # afficher le progres pour chaque batch  
                  num_epoch=10)  # s'entraîner pour un maximum de nombre de passes sur le jeu de données 

    # evaluation finale du modele 
    acc = mx.metric.Accuracy()
    mlp_model.score(test_iter, acc)
    print(acc)

else:
    print('platforme non disponible')

