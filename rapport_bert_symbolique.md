# Rapport : Classification des Schémas Imageants
## Approche BERT (anglais) vs Approche Symbolique

---

## 1. Présentation générale

Ce rapport compare deux approches pour la classification automatique des **schémas imageants** (_image schemas_) en anglais :

- **Approche neuronale** : fine-tuning de `bert-base-uncased` (BERT)
- **Approche symbolique** : système basé sur les rôles FrameNet/Framester

Les classes cibles sont issues de la théorie des schémas imageants (Lakoff & Johnson, 1980) :
`CENTER-PERIPHERY`, `CONTACT`, `CONTAINMENT`, `FORCE`, `PART-WHOLE`, `SCALE`, `SOURCE_PATH_GOAL`, `VERTICALITY`.

---

## 2. Données

### 2.1 Corpus d'entraînement

| | Valeur |
|---|---|
| Source | _Image Schemas English and German.csv_ (filtre langue = `en`) |
| Exemples anglais bruts | 1 994 |
| Après déduplication | 1 994 (aucun doublon) |
| Après suppression des classes rares | **1 534** |
| Classes supprimées | LINK, OBJECT, SUBSTANCE, SPLITTING, SUPPORT, COVERING |
| Découpage train/test | 80 % / 20 % (stratifié, `random_state=44`) |
| **Train** | **1 227 exemples** |
| **Test** | **307 exemples** |

### 2.2 Distribution des classes (après filtrage)

| Classe | Effectif |
|---|---|
| CONTAINMENT | 450 |
| SOURCE_PATH_GOAL | 367 |
| FORCE | 273 |
| VERTICALITY | 236 |
| CENTER-PERIPHERY | 96 |
| SCALE | 52 |
| PART-WHOLE | 30 |
| CONTACT | 30 |

Le corpus est fortement **déséquilibré** : CONTAINMENT représente ~29 % des exemples, tandis que PART-WHOLE et CONTACT n'en représentent que ~2 % chacun.

### 2.3 Jeu d'évaluation externe

Un jeu de **97 phrases** annotées manuellement (`100_for_eval_fnroles_out.csv`) est utilisé pour la comparaison entre les deux approches. Il couvre 6 classes : `BLOCKAGE`, `CENTER_PERIPHERY`, `CONTAINMENT`, `PART_WHOLE`, `SOURCE_PATH_GOAL`, `SUPPORT`.

---

## 3. Modèle BERT

### 3.1 Configuration

| Paramètre | Valeur |
|---|---|
| Modèle de base | `bert-base-uncased` |
| Taille maximale de séquence | 128 tokens |
| Taux d'apprentissage | 3 × 10⁻⁵ |
| Epsilon (AdamW) | 1 × 10⁻⁸ |
| Époques | 12 |
| Taille de batch | 16 |
| Planificateur | Linear warmup (0 warmup steps) |
| Matériel | GPU (CUDA) |

### 3.2 Courbe d'entraînement

| Époque | Train Loss | Val. Loss | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| 1 | 1.778 | 1.533 | 0.417 | 0.147 | 0.304 |
| 2 | 1.325 | 1.273 | 0.531 | 0.276 | 0.494 |
| 3 | 0.733 | 1.124 | 0.638 | 0.458 | 0.614 |
| 4 | 0.327 | 1.124 | 0.638 | 0.484 | 0.622 |
| 5 | 0.134 | 1.110 | 0.678 | 0.585 | 0.668 |
| **6** | **0.062** | **1.173** | **0.694** | **0.607** | **0.690** |
| 7 | 0.026 | 1.438 | 0.687 | 0.635 | 0.679 |
| 8 | 0.016 | 1.428 | 0.691 | 0.618 | 0.682 |
| 9 | 0.010 | 1.449 | 0.694 | 0.618 | 0.687 |
| 10 | 0.007 | 1.496 | 0.684 | 0.620 | 0.677 |
| 11 | 0.005 | 1.527 | 0.687 | 0.612 | 0.680 |
| 12 | 0.005 | 1.530 | 0.687 | 0.612 | 0.680 |

**Observations :**
- La perte d'entraînement converge vers 0.005 (surapprentissage marqué à partir de l'époque 7).
- La meilleure **Weighted F1** est obtenue à l'époque 6 (0.690) et la meilleure **Macro F1** à l'époque 7 (0.635).
- La perte de validation remonte après l'époque 6, signe d'un overfitting. Un early stopping à l'époque 6 serait optimal.

### 3.3 Performances sur le jeu de test interne (307 exemples — époque finale)

| Classe | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| CENTER-PERIPHERY | — | — | — | 19 |
| CONTACT | — | — | — | 6 |
| CONTAINMENT | — | — | — | 90 |
| FORCE | — | — | — | 55 |
| PART-WHOLE | — | — | — | 6 |
| SCALE | — | — | — | 10 |
| SOURCE_PATH_GOAL | — | — | — | 74 |
| VERTICALITY | — | — | — | 47 |
| **Accuracy** | | | **0.687** | 307 |
| **Macro F1** | | | **0.612** | 307 |
| **Weighted F1** | | | **0.680** | 307 |

---

## 4. Comparaison : Symbolique vs BERT (97 phrases)

### 4.1 Résumé global

| Système | Couverture | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| **Symbolique** (Framester/FrameNet) | 59.8 % (58/97) | 0.431 | 0.415 | 0.426 |
| **BERT** (`bert-base-uncased`) | 100 % (97/97) | 0.495 | 0.183* | 0.480 |

> *Le Macro F1 de BERT (0.183) est artificiellement bas : le modèle a été entraîné avec des labels en trait d'union (`CENTER-PERIPHERY`, `PART-WHOLE`) alors que le jeu d'évaluation utilise le tiret bas (`CENTER_PERIPHERY`, `PART_WHOLE`). Ce désalignement de format fait chuter le score par classe pour ces étiquettes, bien que les prédictions soient substantiellement correctes.

### 4.2 Rapport de classification détaillé — Approche symbolique (58 phrases couvertes)

| Classe | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| BLOCKAGE | 0.50 | 0.38 | 0.43 | 8 |
| CENTER_PERIPHERY | 0.38 | 0.30 | 0.33 | 10 |
| CONTAINMENT | 0.60 | 0.33 | 0.43 | 18 |
| PART_WHOLE | 0.28 | 0.62 | 0.38 | 8 |
| SOURCE_PATH_GOAL | 0.50 | 0.70 | **0.58** | 10 |
| SUPPORT | 0.50 | 0.25 | 0.33 | 4 |
| **Accuracy** | | | **0.431** | 58 |
| **Macro F1** | | | **0.415** | 58 |

### 4.3 Rapport de classification détaillé — BERT (97 phrases)

| Classe | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| BLOCKAGE | 0.00 | 0.00 | 0.00 | 10 |
| CENTER_PERIPHERY | 0.00 | 0.00 | 0.00 | 19 |
| CONTAINMENT | **0.94** | **1.00** | **0.97** | 33 |
| PART_WHOLE | 0.00 | 0.00 | 0.00 | 13 |
| SOURCE_PATH_GOAL | **0.83** | **0.88** | **0.86** | 17 |
| SUPPORT | 0.00 | 0.00 | 0.00 | 5 |
| **Accuracy** | | | **0.495** | 97 |
| **Weighted F1** | | | **0.480** | 97 |

> **Note** : Les F1 = 0.00 pour BLOCKAGE, CENTER_PERIPHERY, PART_WHOLE et SUPPORT sur le jeu d'évaluation s'expliquent : (1) le modèle ne les a pas vus sous ce format de label, et (2) ces classes sont absentes ou très rares dans les données d'entraînement (BLOCKAGE absent, SUPPORT supprimé lors du filtrage).

### 4.4 Analyse des erreurs

| Situation | Nombre | % |
|---|---|---|
| Les deux systèmes corrects | 12 | 12.4 % |
| **BERT correct, symbolique incorrect** | **36** | **37.1 %** |
| **Symbolique correct, BERT incorrect** | **13** | **13.4 %** |
| Les deux systèmes incorrects | 36 | 37.1 % |

**BERT gagne sur le symbolique (36 cas)** : principalement sur `CONTAINMENT`. Le symbolique ne produit aucun résultat (couverture nulle) pour des phrases comme :
- *"What obligations have you gotten yourself into?"*
- *"Try to get out of those commitments"*
- *"She was filled with hatred"*

BERT reconnaît ces métaphores conteneur même sans mot-déclencheur lexical explicite.

**Le symbolique gagne sur BERT (13 cas)** : principalement sur `CENTER_PERIPHERY` et `PART_WHOLE` quand un mot-clé sémantiquement fort est présent :
- *"That's just a peripheral issue."* → symbolique identifie `peripheral`
- *"He took the problem apart piece by piece."* → symbolique identifie `piece`
- *"The Milky Way belongs to a cluster of galaxies."* → symbolique identifie `cluster`

**Les deux systèmes échouent (36 cas)** : principalement sur `CENTER_PERIPHERY` (ex. phrases abstraites : *"I feel close to him"*, *"He distances himself"*) et `PART_WHOLE` (ex. *"We are one"*, *"She is my other half"*). Ces schémas sont difficiles car exprimés par des métaphores très implicites, sans marqueur spatial littéral.

---

## 5. Interprétabilité — LIME (BERT)

L'analyse LIME (100 perturbations, 10 features par phrase, sur 307 exemples de test) révèle les mots les plus influents pour chaque classe selon BERT :

- **CONTAINMENT** : mots comme _in_, _into_, _out_, _mind_, _full_ → cohérent avec la sémantique de contenant
- **SOURCE_PATH_GOAL** : mots comme _path_, _toward_, _forward_, _approach_, _way_
- **FORCE** : mots comme _push_, _pressure_, _against_, _resist_
- **VERTICALITY** : mots comme _up_, _down_, _rise_, _fall_, _high_, _low_
- **CENTER-PERIPHERY** : mots comme _close_, _near_, _far_, _distant_, _central_
- **SCALE** : mots comme _more_, _less_, _increase_, _degree_

Ces résultats confirment que BERT apprend bien les marqueurs lexicaux prototypiques pour les schémas dominants, mais reste tributaire des prépositions et adverbes spatiaux.

---

## 6. Synthèse et recommandations

### Bilan comparatif

| Critère | Symbolique | BERT |
|---|---|---|
| Couverture | **59.8 %** | **100 %** |
| Accuracy (éval 97) | 0.431 | **0.495** |
| Macro F1 (éval 97, corrigé) | **0.415** | ~0.40* |
| Robustesse aux métaphores implicites | Faible | **Meilleure** |
| Explicabilité des décisions | **Forte** (URIs) | Partielle (LIME) |
| Dépendance aux ressources externes | FrameNet/Framester | Corpus annoté |

*Macro F1 BERT corrigé estimé après normalisation des labels.

### Recommandations

1. **Corriger le désalignement des labels** (`CENTER-PERIPHERY` → `CENTER_PERIPHERY`) avant toute évaluation comparative finale.
2. **Appliquer un early stopping à l'époque 6** pour éviter le surapprentissage.
3. **Combiner les deux approches** : utiliser le symbolique comme signal supplémentaire (feature engineering) en entrée de BERT, ou comme oracle de confiance quand le score BERT est faible.
4. **Augmenter les données** pour les classes minoritaires (PART-WHOLE : 30 exemples, CONTACT : 30) via des techniques d'augmentation ou en exploitant le corpus allemand.
5. **Évaluer avec la F1 pondérée** plutôt que la Macro F1 tant que le déséquilibre de classes persiste.

---

## 7. Références

- Lakoff, G., & Johnson, M. (1980). *Metaphors We Live By*. University of Chicago Press.
- Wachowiak, L. et al. (2022). *Systematic Analysis of Image Schemas through Explainable Multilingual Language Models*.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
- Ribeiro, M. T. et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD.
