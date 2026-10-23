# Projet de Personnalisation

## Description

Ce dépôt contient l'implémentation d'un moteur de personnalisation basé sur le Deep Q‑Learning (DQN) et des composants de streaming Scala. Il fournit des API pour entraîner et servir des recommandations personnalisées.

## Installation

```bash
# Clone le dépôt
git clone https://github.com/maxit-prototype-personalization/ia-data.git
cd ia-data

# Installer les dépendances Python
pip install -r requirements.txt
```

## Utilisation

- **API Python** : voir `api.py` pour les points d'accès Flask.
- **Entraînement** : exécuter le moteur DQN avec `python dqn_engine.py`.
- **Streaming Scala** : compiler et lancer le service de streaming Scala situé dans `src/main/scala/`.

## Démarrage

Voici les commandes de démarrage typiques :

```bash
# Lancer l'API Flask
python -m flask run
uvicorn api:app --host [IP_ADDRESS] --port 8000

# Démarrer le moteur DQN
python dqn_engine.py

# Lancer le service de streaming Scala
sbt run
spark-submit --class MainStreamingPersonalization --master "local[*]" --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 target\out\jvm\scala-2.13.17\api-spark-scala\api-spark-scala_2.13-1.0.jar 
```

## Contribution

N'hésitez pas à ouvrir des issues ou à proposer des pull requests. Suivez les directives de contribution dans `CONTRIBUTING.md` (si présent).

## Licence

Ce projet est sous licence MIT. Consultez le fichier `LICENSE` pour plus de détails.
