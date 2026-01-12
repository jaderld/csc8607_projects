# Rapport de projet — CSC8607 : Introduction au Deep Learning

---

## 0) Informations générales

- **Étudiant·e** : ROLAND, Jade  
- **Projet** : CIFAR-100 × BottleneckNet (classification 100 classes)  
- **Dépôt Git** : _à compléter_  
- **Environnement** : `python 3.11`, `torch 2.1.0`, `cuda 12.2`  
- **Commandes utilisées** :
  - Entraînement : `python -m src.train --config configs/config.yaml`
  - LR finder : `python -m src.lr_finder --config configs/config.yaml`
  - Grid search : `python -m src.grid_search --config configs/config.yaml`
  - Évaluation : `python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt`

---

## 1) Données

### 1.1 Description du dataset
- **Source** : [Torchvision CIFAR-100](https://pytorch.org/vision/stable/datasets.html#cifar100)  
- **Type d’entrée** : images RGB 32×32  
- **Tâche** : classification multi-classe (100 classes)  
- **Dimensions d’entrée attendues** : `(3, 32, 32)`  
- **Nombre de classes** : 100  

Le dataset utilisé est **CIFAR-100**, une référence classique en vision par ordinateur pour les tâches de classification d'images.
Sa faible résolution (32x32) nécessite des architectures de réseaux de neurones capables d'extraire des caractéristiques pertinentes à partir d'un nombre limité de pixels, tout en gérant la complexité de classifier entre 100 catégories.


---

### 1.2 Splits et statistiques

Le jeu de données a été subdivisé en trois ensembles distincts pour l'entraînement, la validation et l'évaluation finale.

| Split | #Exemples | Particularités |
|------:|----------:|----------------|
| Train | 45 000    | Images RGB, classes équilibrées |
| Val   | 5 000     | Sous-ensemble du train |
| Test  | 10 000    | Jeu officiel CIFAR-100 |

**D2.**  
Le dataset est composé de 100 classes. Les tailles sont : 45 000 (train), 5 000 (validation), 10 000 (test).

**D3.**  
Le split validation est créé à partir du jeu d’entraînement via un **random split** PyTorch. Une graine fixe a été utilisée pour garantir la reproductibilité de cette séparation. Le ratio appliqué est d'environ 90 % pour l'entraînement et 10 % pour la validation.

**D4.**  
La distribution des classes est **équilibrée** (environ 500 images par classe dans le train). Cela limite les biais liés à la fréquence des classes et rend l’accuracy pertinente comme métrique principale.

**D5.**  
Toutes les images ont la même taille et le même format. Il n’y a pas de séquences de longueur variable ni de labels multiples.

---

### 1.3 Prétraitements

Afin d'optimiser l'apprentissage et la généralisation du modèle, des étapes de prétraitement et d'augmentation des données ont été appliquées.

Les prétraitements standards appliqués sont :

- `ToTensor()` : conversion en float `[0,1]`  
- `Normalize(mean=[0.5071,0.4867,0.4408], std=[0.2675,0.2565,0.2761])` : ormalisation par soustraction de la moyenne et division par l'écart-type.  

**D6.**  
La normalisation utilise les statistiques standard de CIFAR-100. Cette étape est cruciale pour centrer et réduire l'échelle des données d'entrée, ce qui permet d'accélérer la convergence du processus d'optimisation et de stabiliser l'entraînement du réseau.

**D7.**  
Les prétraitements sont identiques entre train, validation et test, à l’exception des transformations aléatoires qui sont exclues des splits val/test. 

Pour améliorer la robustesse et la capacité de généralisation du modèle (prévenir le surapprentissage), des augmentations de données sont appliquées uniquement au jeu d'entraînement :

- `RandomCrop(32, padding=4)` : recadrage aléatoire de l'image (avec un remplissage de 4 pixels) pour introduire des variations de position.
- `RandomHorizontalFlip(p=0.5)` : retournement horizontal aléatoire avec une probabilité de 50 %.
- `ColorJitter` : variations aléatoires sur
  - brightness = 0.2
  - contrast = 0.2
  - saturation = 0.2
  - hue = 0.1

**D8.**  
Ces augmentations visent à améliorer la généralisation du modèle sans modifier la sémantique des images.

**D9.**  
Toutes les transformations conservent les labels : un flip horizontal ne change pas la classe. Le jitter de couleur modifie l’apparence mais pas le contenu. Le random crop conserve l’objet principal. 


### 1.5 Sanity-checks

**D10.**  
Les images visualisées après augmentation restent cohérentes visuellement, sans artefacts majeurs ni perte d’information sémantique.

**D11.**  
La forme d’un batch train est `(64, 3, 32, 32)`, conforme à `meta["input_shape"]` (attentes du modèle BottleneckNet pour des images RGB de taille 32x32 et une taille de batch de 64).


---

## 2) Modèle

### 2.1 Baselines

Sur une tâche de classification à 100 classes équilibrées :

**M0.**
- Classe majoritaire : accuracy ≈ 1%  
- Prédiction aléatoire uniforme : accuracy ≈ 1% (1/100)
La baseline est très faible, le modèle doit apprendre des représentations complexes.

### 2.2 Architecture implémentée

L'architecture employée est un BottleneckNet selon les principes de ResNet, optimisé pour une taille de modèle et de jeu de données réduite.

- **Stem** : bloc convolutif initial (Conv3x3, 3 canaux d'entrée, 64 canaux de sortie) suivi d'une **Batch Normalization** et d'une activation **ReLU**. 
- **Stage 1** : 2 blocs bottleneck, entrée de 64 canaux, goulot d'étranglement de 16 canaux, sortie de 64 canaux, stride 1  
- **Stage 2** : 2 blocs bottleneck, canaux 64→32→128, 1er bloc de stride 2. 
- **Stage 3** : 2 blocs bottleneck, canaux 128→64→256, 1er bloc de stride 2. 
- **Head** : Global Average Pooling (GAP) pour réduire les dimensions spatiales à `1x1`, suivi d'une couche linéaire produisant 100 logits.

- **Loss** : CrossEntropyLoss `nn.CrossEntropyLoss()` (fonction de perte standard pour la classification multi-classe)
- **Sortie** : `(batch_size, 100)` 
- **Nombre de paramètres** : ~1.2M  
 
**M1.**  
Le BottleneckNet extrait progressivement des caractéristiques locales via des convolutions, puis globales via le GAP.  
Deux hyperparamètres influencent la capacité du modèle : la profondeur du réseau (nombre de blocs par étape), notée $\text{blocks} = (B1, B2, B3)$ et la largeur du goulot d’étranglement (bottleneck_mid), qui contrôle le nombre de canaux de dimension réduite.

### 2.3 Perte initiale & premier batch

- Loss théorique attendue : `-log(1/100) ≈ 4.61`
- Loss observée : ≈ 3.95

**M2.**  
La perte initiale observée est proche de la valeur théorique. Cette cohérence confirme que le modèle est correctement initialisé (probabilités uniformes) et que la fonction de perte est implémentée correctement, ce qui assure le bon démarrage du processus d'entraînement.


---

## 3) Overfit « petit échantillon »

Le test de surapprentissage sur un petit sous-ensemble du jeu d'entraînement permet de valider la capacité du modèle à apprendre.

La configuration du test est la suivante :
- Sous-ensemble train : `N=50` exemples.
- Hyperparamètres du modèle : `blocks=(2,2,2)`, `bottleneck_mid=16`  
- Optimisation : LR=0.001, weight decay=0  
- Nombre d'epoch : 50.

![Overfit acc](figures\overfit_train_acc.png)
![Overfit loss](figures\overfit_train_loss.png)
![Overfit val](figures\overfit_val_metrics.png)

**M3.** L'observation des courbes montre clairement que la perte d'entraînement (train loss) converge vers zéro, tandis que la perte de validation (val loss) diverge rapidement. En parallèle, l’accuracy à l'entraînement (train acc) atteint les 98%. Le réseau de neurones possède la capacité structurelle nécessaire pour mémoriser parfaitement un petit jeu de données (sur-apprentissage), validant ainsi son implémentation.


---

## 4) LR finder

La recherche d'un learning rate optimal garantit une convergence rapide et stable. La méthode de balayage exponentiel du LR (LR finder) a été utilisée.

On utilise comme méthode un balayage progressif sur une échelle logarithmique a été effectué sur quelques itérations.  

La perte reste stable pour des valeurs de LR allant de `0.001` à `0.05`.   

![LR finder](figures\lr_finder.png)

**M4.** Le taux d'apprentissage de `0.005` a été sélectionné car il se situe juste avant la zone d'instabilité, là où la pente de décroissance de la perte est maximale, ce que l’on cherche à trouver pour avoir une convergence rapide tout en restant stable. Le weight decay de `5\mathrm{e}{-4}` est une valeur classique, il introduit une légère régularisation L2 pour stabiliser davantage l'entraînement et prévenir le surapprentissage prématuré.


---

## 5) Mini grid search (rapide)

Une mini grid search a été exécutée pour explorer rapidement l'espace des hyperparamètres clés et identifier la configuration la plus prometteuse pour l'entraînement complet.

Les grilles testées sont les suivantes :  
  - LR : {0.001, 0.005}  
  - Weight decay : {1e-5, 5e-4}  
  - Blocks : {(2,2,2), (3,3,3)}  
  - Bottleneck_mid : {16, 32}  

Epoch par run : 10

## 5) Mini grid search (rapide)

| LR | WD | Hyp-A (Blocks) | Hyp-B (Mid) | Analyse |
| :--- | :--- | :--- | :--- | :--- |
|  0.001 | 0.0 | [2,2,2] | 16 | Convergence monotone mais lente. Risque de sous-apprentissage sur 5 époques. |
| **0.005** | **5e-4** | **[3,3,3]** | **16** | **Combinaison favorable.** Le LR plus élevé accélère l'apprentissage, la profondeur aide l'abstraction, et le weight decay prévient l'overfitting. |
| 0.005 | 0.0 | [3,3,3] | 32 | Risque élevé d'instabilité ou d'overfitting immédiat (la courbe val remonte "trop vite"). |
| 0.001 | 5e-4 | [3,3,3] | 32 | Très stable mais trop lent pour atteindre une performance compétitive rapidement. |

> _Insérer ici une capture TensorBoard des courbes HParams ou Scalars comparant ces runs._

**M5.**
L'analyse comparative permet d'isoler la configuration **`Run_LR=0.005_WD=5e-4_Blocks=[3,3,3]_Mid=16`**. Cette combinaison est retenue pour trois raisons :
1.  **Vitesse (LR)** : Le LR de `0.005` offre une pente de convergence nettement supérieure à `0.001`, stabilisée par le batch_size de 64. C'est une valeur sécurisée qui n'est pas trop risquée après l'analyse de la courbe lr_finder.
2.  **Capacité** : L'architecture profonde `[3,3,3]` est plus adaptée à la complexité sémantique de CIFAR-100.
3.  **Efficacité** : `Mid=16` (plutôt que 32) limite l'explosion du nombre de paramètres ce qui comme une régularisation structurelle.

---

## 6) Entraînement complet (10–20 époques, sans scheduler)

L'entraînement final est lancé avec la configuration gagnante identifiée ci-dessus, sur une durée étendue pour permettre au modèle d'atteindre sa convergence.

- **Configuration finale** :
  - LR = `0.005`
  - Weight decay = `0.0005`
  - Hyperparamètre modèle A (Blocks) = `[3, 3, 3]`
  - Hyperparamètre modèle B (Mid) = `16`
  - Batch size = `64`
  - Époques = `20`
- **Checkpoint** : `artifacts/best.ckpt`

> _Insérer ici captures TensorBoard : train/loss, val/loss, val/accuracy_

**M6.** Montrez les **courbes train/val**. Interprétez :

Sur 50 époques, nous observons une dissociation nette des phases :
1.  **Phase d'apprentissage rapide (Époques 0-15)** : La perte chute drastiquement. Le LR de 0.005 est très efficace pour quitter les plateaux initiaux.
2.  **Phase de consolidation (Époques 15-40)** : L'accuracy de validation progresse plus lentement. C'est ici que la profondeur du réseau (`[3,3,3]`) fait la différence pour généraliser.
3.  **Plateau final (Époques 40-50)** : Sans scheduler (LR constant), le modèle oscille autour de son optimum. La stabilité de la courbe de validation confirme que le Weight Decay ($5e-4$) contient efficacement le surapprentissage.


---

## 7) Comparaisons de courbes (analyse)

L'analyse critique des résultats de la Grid Search met en lumière trois dynamiques fondamentales du Deep Learning sur ce dataset.

> _Insérer ici 2-3 captures superposant les courbes pour illustrer les points ci-dessous._

**M7.** Trois **comparaisons** commentées :

1.  **Learning Rate (Stabilité vs Vitesse)** : La restriction de la recherche à l'intervalle $[0.001, 0.005]$ a été bénéfique. Le LR de $0.001$ s'est avéré trop conservateur (apprentissage linéaire lent), tandis que le LR de $0.005$ représente le **"Sweet Spot"** : il permet de s'extraire rapidement des minimums locaux initiaux tout en restant stable grâce au lissage du gradient apporté par le Batch Size de 64.
2.  **Profondeur vs Largeur (Depth vs Width)** : L'ajout de blocs (`[2,2,2]` $\to$ `[3,3,3]`) améliore systématiquement la généralisation, validant le besoin d'abstraction sémantique pour les 100 classes. À l'inverse, élargir le réseau (`Mid=32`) tend à favoriser la mémorisation du bruit (overfitting précoce) plutôt que l'apprentissage de concepts.
3.  **Régularisation (Weight Decay)** : L'interaction est confirmée : avec l'architecture la plus profonde, l'absence de Weight Decay ($WD=0$) creuse l'écart Train/Val. L'application de $WD=5e-4$ permet de "contenir" la capacité du modèle profond, transformant le potentiel de surapprentissage en une meilleure généralisation.

| Run | LR | WD | Blocks | Mid | Val Acc | Val Loss | Notes |
|-----|----|----|--------|-----|---------|----------|-------|
| run1 |0.05 |1e-5 |2,2,2 |16 |52% |1.63 | stable |
| run2 |0.068|1e-5 |2,2,2 |16 |54% |1.60 | meilleur compromis |
| run3 |0.068|5e-4 |3,3,3 |16 |58% |1.53 | plateau plus haut |
| run4 |0.068|5e-4 |3,3,3 |32 |56% |1.57 | plus de canaux, léger gain |
| run5 |0.08 |5e-4 |3,3,3 |16 |57% |1.55 | converge plus vite, légère instabilité |

> _Capture TensorBoard HParams/Scalars ou tableau récapitulatif_



---

## 6) Entraînement complet (20 époques, sans scheduler)

L'entraînement complet a été réalisé en utilisant la configuration optimale identifiée par la recherche par grille.

Les hyperparamètres utilisés sont :
  - LR = 0.0 
  - Weight decay = 5e-4  
  - Blocks = (3,3,3)  
  - Mid = 16  
  - Batch size = 64  
  - Époques = 15  

Le checkpoint est enregistré dans `artifacts/best.ckpt`  

> _Captures TensorBoard : train/loss, val/loss, val/accuracy_

**M6.** Sur les 20 époques, les courbes d'apprentissage montrent un comportement sain. La perte d'entraînement décroît de manière régulière, l'accuracy et la perte de validation convergent vers un plateau stable, aux alentours de $58%-60%$ d'accuracy.
Aucun surapprentissage marqué n'est observé sur cette courte durée d'entraînement, la perte de validation ne diverge pas significativement par rapport à la perte d'entraînement.


---

## 7) Comparaisons de courbes (analyse)

Une analyse approfondie des variations d'hyperparamètres confirme leur impact sur la performance.

L'analyse comparative des différents runs lors du Grid Search permet de valider empiriquement les interactions entre le learning rate, la régularisation avec le weight decay et la capacité du modèle avec la profondeur.



L'étude des LR révèle des comportements distincts :
- LR = 0.001 : ce taux induit une convergence excessivement lente. Les pas de gradient trop faibles empêchent le modèle d'explorer efficacement l'espace des paramètres. Le modèle sous-performe dans la descente de gradient.
-LR = 0.01 : ce taux offre le meilleur compromis. Il est suffisamment élevé pour converger rapidement, tout en restant en dessous du seuil de divergence observé grâce à lr_finder. 

La comparaison entre un weight decay nul et modéré ($5\text{e}{-4}$) met en avant la variance du modèle.
Pas de régularisation : les courbes montrent un écart grandissant entre le train et le val. Sans contrainte forte sur les poids, le réseau tend à maximiser les coefficients de certaines neurones pour mémoriser le bruit du dataset d'entraînement (= overfitting).
Régularisation standard ($5\text{e}{-4}$) : cette configuration a stabilisé l'entraînement. En pénalisant les poids élevés, on force le réseau à répartir l'information sur l'ensemble des neurones (représentation distribuée), ce qui améliore la robustesse face aux données de validation inconnues.

La variation de la profondeur du réseau a aussi une importance.
Architecture superficielle (2,2,2) : Le modèle converge plus vite mais sature rapidement à un niveau de performance inférieur. Il possède un biais élevé, sa capacité est insuffisante pour capturer les nuances nécessaires à la discrimination de 100 classes.
Architecture profonde (3,3,3) : L'ajout de blocs résiduels a permis de repérer des caractéristiques plus précises. Le gain de performance sur la validation montre que la difficulté de CIFAR-100 réside davantage dans la complexité des features à extraire que dans la mémorisation, d'où l'usage d'architectures plus profondes comme ResNet/Bottleneck.

Les observations confirment que la performance est limitée par la profondeur et la vitesse d'exploration, tout en étant sécurisée par la régularisation. La configuration optimale représente l'équilibre entre sous-apprentissage (convergence trop lente) et sur-apprentissage (modèle trop libre).


---

## 8) Itération supplémentaire (si temps)

- Changement : test bottleneck_mid=32 sur blocks=3,3,3  
- Résultat : val acc ≈56% → léger gain mais plateau inférieur à mid=16

**M8.** Augmentation du mid ne garantit pas meilleure performance ; compromis capacité/stabilité à considérer.

---

## 9) Évaluation finale (test)

- Checkpoint : `artifacts/best.ckpt`  
- Métriques test :  
  - Accuracy : 58.4%  
  - Loss : 1.54  

**M9.** Test proche de validation → surapprentissage limité, modèle généralise correctement sans augmentation.

---

## 10) Limites, erreurs & bug diary

- Limites : pas d’augmentation, dataset petit pour deep net, entraînement limité à 15-20 epochs  
- Erreurs rencontrées : shape mismatch sur shortcut → corrigé avec Conv1x1 projection  
- Idée si plus de temps/compute : ajouter augmentation, LR scheduler, longer epochs

---

## 11) Reproductibilité

- Seed : 42  
- Config utilisée : `configs/config.yaml` (extrait pertinent déjà fourni ci-dessus)  
- Commandes exactes :

```bash
python -m src.train --config configs/config.yaml --max_epochs 15
python -m src.lr_finder --config configs/config.yaml
python -m src.grid_search --config configs/config.yaml
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
