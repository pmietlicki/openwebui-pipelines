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

## Variables d'environnement

```bash
# Configuration de base
PIPELINE_TITLE="Custom RAG Pipeline"
VECTOR_INDEX_DIR="/app/pipelines"
DOCS_DIR="/app/pipelines/docs"
DOCS_ROOT_DIR="/app/pipelines/docs"
FILES_HOST="https://sourcefiles.test.local"

# Paramètres de performance
MAX_LOADERS=${CPU_COUNT}*4
EMBED_DIM=1024
BATCH_SIZE=5000

# Paramètres FAISS
HNSW_M=32
HNSW_EF_CONS=100
HNSW_EF_SEARCH=64

# Paramètres de chunking
MIN_CHUNK_LENGTH=50
MAX_TOKENS=2048

# Paramètres de retry
CHAT_MAX_RETRIES=5
CHAT_BACKOFF=1.0
CHAT_MAX_PARALLEL=2
```

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
