# Custom RAG Pipeline

Une implémentation personnalisée de RAG (Retrieval Augmented Generation) utilisant Llama-Index et MistralAI.

## Caractéristiques principales

- **Pipeline RAG complet** avec vector store et chat
- **Optimisation des performances** avec uvloop et threading
- **Gestion robuste des erreurs** avec retry et backoff
- **Support multi-format** pour les documents (PDF, DOCX, TXT, etc.)
- **Vectorisation intelligente** avec embedding MistralAI
- **Chat en streaming** avec historique

## Configuration requise

- Python 3.x
- MistralAI API
- uvloop (optionnel mais recommandé)
- faiss-cpu
- Llama-Index core

## Installation

```bash
pip install   llama-index   llama-index-embeddings-mistralai   llama-index-vector-stores-faiss   llama-index-readers-file>=0.4.0,<0.5.0   llama-index-llms-mistralai   pymupdf   pandas   beautifulsoup4   openpyxl   xlrd   python-pptx   docx2txt>=0.9.0   unstructured   mistralai>=1.0.0,<2.0.0   torch>=2.6.0   transformers>=4.51.3   safetensors>=0.5.3   huggingface_hub[hf_xet]>=0.30.2   faiss-cpu==1.11.0   Pillow>=11.0.0   requests>=2.28.0
```

Ou, si vous avez un `requirements.txt` dans votre dépôt :

```bash
pip install -r requirements.txt
```

## Variables d'environnement

| Variable           | Description                                                        | Valeur par défaut                          |
|--------------------|--------------------------------------------------------------------|--------------------------------------------|
| PIPELINE_TITLE     | Titre du pipeline (transformé en slug)                             | Custom RAG Pipeline                        |
| VECTOR_INDEX_DIR   | Répertoire pour l’index FAISS persistant                           | /app/pipelines                             |
| DOCS_DIR           | Répertoire des documents source                                    | /app/pipelines/docs                        |
| DOCS_ROOT_DIR      | Racine pour génération des URL de sources                          | /app/pipelines/docs                        |
| FILES_HOST         | Hôte des fichiers pour liens                                        | https://sourcefiles.test.local             |
| MAX_LOADERS        | Nombre maximal de threads de chargement                            | CPU_COUNT × 4                              |
| EMBED_DIM          | Dimension des embeddings                                            | 1024                                       |
| BATCH_SIZE         | Taille du batch pour l’indexation                                  | 5000                                       |
| HNSW_M             | Paramètre M pour HNSW (nombre de voisins)                          | 32                                         |
| HNSW_EF_CONS       | Paramètre efConstruction pour HNSW                                 | 100                                        |
| HNSW_EF_SEARCH     | Paramètre efSearch pour HNSW                                       | 64                                         |
| MIN_CHUNK_LENGTH   | Longueur minimale d’un chunk en caractères                         | 50                                         |
| MAX_TOKENS         | Nombre maximum de tokens pour l’appel LLM                          | 2048                                       |
| CHAT_MAX_RETRIES   | Nombre maximal de tentatives de retry pour le chat                 | 5                                          |
| CHAT_BACKOFF       | Backoff exponentiel initial (en secondes)                          | 1.0                                        |
| CHAT_MAX_PARALLEL  | Nombre maximal de requêtes chat parallèles                         | 2                                          |
| SIM_THRESHOLD      | Seuil de similarité pour arrêt de la récupération (0.0–1.0)        | 0.75                                       |
| BASE_TOP_K         | Pas de top_k initial pour la récupération adaptative               | 5                                          |
| MAX_TOP_K          | Top_k maximal pour la récupération adaptative                      | 15                                         |

## Optimisations avancées

- Ajustement des paramètres HNSW (`HNSW_M`, `HNSW_EF_CONS`, `HNSW_EF_SEARCH`) pour performance vs précision.  
- Réglage de `BASE_TOP_K` et `MAX_TOP_K` pour la stratégie de récupération adaptative.  
- Taille de batch (`BATCH_SIZE`) ajustable selon volume de données.

## Fonctionnalités principales

### Pipeline

- Indexation vectorielle des documents
- Recherche sémantique avec seuil de similarité
- Chat en streaming avec historique
- Support multi-format des documents
- Optimisation des performances avec threading

### Gestion des erreurs

- Retry automatique sur les erreurs 429
- Backoff exponentiel
- Gestion des limites de tokens
- Logging détaillé

### Optimisations

- Utilisation de uvloop pour les performances
- Threading pour le traitement parallèle
- Optimisation FAISS pour la recherche
- Batch processing pour les embeddings

## Utilisation

### Exemple de déploiement Kubernetes

Voici un exemple complet de déploiement avec Helm et Kubernetes :

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: openwebui
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://helm.openwebui.com
    chart: open-webui
    targetRevision: "6.*"
    helm:
      values: |
        extraEnvVars:
          - name: LOG_LEVEL
            value: "debug"
          - name: PIPELINE_TITLE
            value: "Cocktail GFC"
          - name: MISTRAL_API_KEY
            valueFrom:
              secretKeyRef:
                name: mistral-secret
                key: api-key
          - name: PIPELINES_URLS
            value: "https://raw.githubusercontent.com/pmietlicki/openwebui-pipelines/refs/heads/main/custom_rag_pipeline.py"
          - name: VECTOR_INDEX_DIR
            value: "/app/pipelines"
          - name: DOCS_DIR
            value: "/app/pipelines/docs/doc-fonc/gfc"
          - name: PIPELINES_REQUIREMENTS_PATH
            value: "/config/requirements.txt"

        pipelines:
          enabled: true
          persistence:
            enabled: true
            existingClaim: openwebui-index-pvc
          volumes:
            - name: config
              configMap:
                name: pipelines-requirements
          volumeMounts:
            - name: config
              mountPath: /config

  destination:
    server: https://kubernetes.default.svc
    namespace: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

### Exemple de déploiement multi-pipelines

Pour déployer plusieurs instances de pipelines (par exemple pour différents départements), utilisez cette configuration :

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: openwebui-pipelines
  namespace: argocd
spec:
  project: default
  sources:
    # Instance GRH
    - repoURL: https://helm.openwebui.com
      chart: pipelines
      targetRevision: "0.*"
      helm:
        releaseName: openwebui-pipelines-grh
        values: |
          extraEnvVars:
            - name: PIPELINE_TITLE
              value: "Cocktail GRH"
            - name: DOCS_DIR
              value: "/app/pipelines/docs/doc-fonc/grh"
            - name: VECTOR_INDEX_DIR
              value: "/app/pipelines/index/grh"

    # Instance GFC
    - repoURL: https://helm.openwebui.com
      chart: pipelines
      targetRevision: "0.*"
      helm:
        releaseName: openwebui-pipelines-gfc
        values: |
          extraEnvVars:
            - name: PIPELINE_TITLE
              value: "Cocktail GFC"
            - name: DOCS_DIR
              value: "/app/pipelines/docs/doc-fonc/gfc"
            - name: VECTOR_INDEX_DIR
              value: "/app/pipelines/index/gfc"

  destination:
    server: https://kubernetes.default.svc
    namespace: default
  syncPolicy:
    automated: { prune: true, selfHeal: true }
    syncOptions: [ CreateNamespace=true ]
```

### Configuration des volumes

Le pipeline nécessite plusieurs volumes persistants :

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: openwebui-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: openwebui-ollama-models-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: openwebui-index-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
```

## Structure du projet

```
.
├── custom_rag_pipeline.py    # Implémentation principale
└── README.md                # Documentation
```

## Contributing

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## Licence

Ce projet est sous licence MIT.
