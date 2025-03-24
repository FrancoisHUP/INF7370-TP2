# INF7370
## 1. Introduction (optionnelle)

> Mise en contexte/description général de code/ décrire l’objectif du travail avec les différentes étapes nécessaires pour le réaliser.

Ce travail a pour but d'entraîner un réseau de neurones pour la classification d'images avec une architecture personnalisée. L'objectif est de tester différentes architectures et de valider les performances de ces modèles. Le but est de trouver le modèle qui aura les meilleures performances (précision) et qui consommera le moins de ressources lors de l'entraînement.

**Données** 

Les données sont des images de 6 classes d'animaux marins. Nous avons des images de :

- Baleine
- Dauphin
- Morse
- Phoque
- Requin
- Requin-baleine

Au total, nous avons 30 000 images, réparties en 1 000 exemples de test et 4 000 exemples d'entraînement et de validation par classe.

Nous omettons d'ajouter une classe "autre" contenant des images d'autres types d'animaux marins, ou même des images prises au hasard sur le web. Cette classe est normalement utilisée pour que notre modèle soit capable d’identifier nos classes face à d'autres images. Les modèles entraînés avec cette classe permettent de soumettre toute sorte d'image au modèle et d'obtenir une bonne classification à chaque fois. Pour cet exercice, nous n'inclurons pas cette classe.

Les images sont en couleur (3 canaux) et sont de tailles diverses. Certaines sont en mode paysage, d'autres en mode "téléphone". En général, la longueur et la hauteur ne dépassent pas 256 pixels.

Les images sont de types et de positions variés. Il y a des images d'animaux vus de face, de côté, dans l'eau, sur la plage, ou même des dessins. Parfois, l’animal prend tout l’espace de l’image, d’autres fois seulement une partie ; parfois, il y a plusieurs spécimens sur une image, d'autres fois un seul animal. Certaines images contiennent du texte. Certaines sont en noir et blanc.

Concernant les erreurs de classification dans les données : nous ne sommes pas des experts en animaux marins, mais on peut dire qu’il n’y a pas d’erreurs aberrantes (comme s’il y avait un cheval dans la classe "baleine"). On peut assumer que les animaux sont relativement bien classés.

En ce qui concerne les duplications, nous n’avons pas remarqué de données dupliquées. On suppose donc que le dataset ne contient pas de duplications.

<img src="donnees\entrainement\baleine\0478.jpg"></img>

## 2. Montage de l’architecture et entrainement du modèle

### 2.1 Ensemble de données

>Description de l’ensemble des données avec les proportions (Nombre d’images par classe pour chacune des catégories : entrainement, validation et test)

### 2.2 Traitement de données

>Décrire le traitement effectué (data augmentation par exemple)

### 2.3 Paramètres et Hyperparamètres

>L’optimiseur utilisé avec les paramètres associés à cet optimiseur.

>La taille du lot (batch size) d’entrainement.

>Le nombre d’époques (number of Epochs) et l’arrêt précoce s’il y a lieu.

### 2.4 Architecture

>Décrire l’architecture de votre CNN :
>- Le nombre de couches utilisées avec le type les paramètres de chaque couche
>-Dropout : Oui/Non ?
>- Le type des fonctions d’activations.
>- Inclure une figure qui relate votre architecture. Ne pas inclure une figure dessinée manuellement.

### 2.5 Affichage des résultats d’entrainement

>Le temps total d’entraînement en minutes.

>L’erreur minimale commise lors de l’entrainement (Minimum Loss).

>L’exactitude maximale de l’entraînement (Maximum Accuracy).

>Inclure une figure qui relate la courbe d’exactitude par époque.

>Inclure une figure qui relate la courbe de perte.

### 2.6 Justification du choix de l’architecture

>Justifier vos choix en indiquant les facteurs ayant contribué à l’amélioration de l’entrainement.

## 3. Évaluation du Modèle

>Afficher les résultats de l’évaluation :

>L’exactitude (accuracy) du modèle développé sur les données de test. Minimum à atteindre est de 82% sur les données de test.

>Inclure une figure qui relate la matrice de confusion.

>Inclure une figure qui relate les images classées en suivant le même exemple d’affichage qui est présenté dans l’énoncé.

## 4. Conclusion

>Conclure votre rapport en discutant, d’une façon générale, les problèmes rencontrés ainsi que les démarches possibles qui peuvent considérés pour améliorer votre modèle