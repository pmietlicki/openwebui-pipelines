import os
import time
import json
import logging
import asyncio
import re
import unicodedata
from typing import List, Generator
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

title = os.getenv("PIPELINE_TITLE", "Custom RAG Pipeline")
version = "1.0.0"
author  = "PMietlicki"

def _slugify(s: str) -> str:
    """minuscule, accents supprimés, espaces/ponctuation → _ (slug)."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    return s.strip("_")

pipeline_id = _slugify(title)

# accélérateur asyncio
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

import faiss
faiss.omp_set_num_threads(os.cpu_count() or 1)

# ─── Llama-Index core ─────────────────────────────────────────────────────────
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    get_response_synthesizer,
)

from typing import Any
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore


from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.llms import LLM as BaseLLM

from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage

# ─── Nouveau parser pour chunking ─────────────────────────────────────────────
from llama_index.core.node_parser import TokenTextSplitter

# ─── Readers spécialisés ─────────────────────────────────────────────────────
from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PDFReader,
    EpubReader,
    FlatReader,
    HTMLTagReader,
    ImageCaptionReader,
    ImageReader,
    ImageVisionLLMReader,
    IPYNBReader,
    MarkdownReader,
    MboxReader,
    PptxReader,
    PandasCSVReader,
    VideoAudioReader,
    UnstructuredReader,
    PyMuPDFReader,
    ImageTabularChartReader,
    XMLReader,
    PagedCSVReader,
    CSVReader,
    RTFReader,
)

from llama_index.core.schema import TextNode
from urllib.parse import quote
from pathlib import Path

# patch pour que Llama-Index trouve get_doc_id()
TextNode.get_doc_id = lambda self: self.id_
TextNode.doc_id = property(lambda self: self.id_)

# ─── Mistral client & retries ────────────────────────────────────────────────
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from pydantic import Field, PrivateAttr

# ─── Logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger("custom_rag_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── Concurrence & extensions autorisées ──────────────────────────────────────
CPU_COUNT   = os.cpu_count() or 1
EXTENSIONS  = {".pdf", ".docx", ".txt", ".md", ".html", ".csv", ".xlsx", ".xls", ".xlsm", ".pptx"}
FILES_HOST = os.getenv("FILES_HOST", "https://sourcefiles.test.local")
MAX_LOADERS = int(os.getenv("MAX_LOADERS", CPU_COUNT * 4))
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "5000"))
HNSW_M          = int(os.getenv("HNSW_M", "32"))
HNSW_EF_CONS    = int(os.getenv("HNSW_EF_CONS", "100"))
HNSW_EF_SEARCH  = int(os.getenv("HNSW_EF_SEARCH", "64"))
MIN_CHUNK_L     = int(os.getenv("MIN_CHUNK_LENGTH", "50"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

CHAT_MAX_RETRIES   = int(os.getenv("CHAT_MAX_RETRIES", 5))
CHAT_BACKOFF       = float(os.getenv("CHAT_BACKOFF", 1.0))
CHAT_MAX_PARALLEL  = int(os.getenv("CHAT_MAX_PARALLEL", 2))

SIM_THRESHOLD = 0.75          # seuil “pertinent”
MAX_TOP_K     = 15            # plafond
BASE_TOP_K    = 5 

# ─── Embedding avec back-off ──────────────────────────────────────────────────
class RetryingMistralEmbedding(MistralAIEmbedding):
    batch_size:    int   = Field(32,  description="Taille des sous-lots")
    max_retries:   int   = Field(5,   description="Nb max de tentatives")
    retry_backoff: float = Field(1.0, description="Back-off exponentiel")

    def _call_with_retry(self, func, *args):
        for attempt in range(self.max_retries):
            try:
                return func(*args)
            except SDKError as e:
                msg = str(e)
                if "429" in msg:
                    wait = self.retry_backoff * (2 ** attempt)
                    logger.warning(f"429 reçu (requête), back-off de {wait}s…")
                    time.sleep(wait)
                    continue
                elif "Too many tokens" in msg and isinstance(args[0], list) and len(args[0]) > 1:
                    mid = len(args[0]) // 2 or 1
                    left  = self._call_with_retry(func, args[0][:mid])
                    right = self._call_with_retry(func, args[0][mid:])
                    return left + right
                else:
                    raise
        raise RuntimeError(f"Embeddings échoués après {self.max_retries} tentatives")

    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False, **kwargs) -> List[List[float]]:
        parent = super(RetryingMistralEmbedding, self)._get_text_embeddings
        all_embeds = []
        for i in range(0, len(texts), self.batch_size):
            sub = texts[i : i + self.batch_size]
            all_embeds.extend(self._call_with_retry(parent, sub))
        return all_embeds

    def _get_query_embedding(self, query: str) -> List[float]:
        # on wrappe exactement la même logique autour de la requête simple
        parent = super(RetryingMistralEmbedding, self)._get_query_embedding
        return self._call_with_retry(parent, query)

# ─── Embedding avec back-off ──────────────────────────────────────────────────
class RetryingLLM(CustomLLM):
    """Un LLM avec back-off exponentiel sur 429 + retry token-limite."""

    # 1) Champs Pydantic
    max_retries: int = CHAT_MAX_RETRIES
    backoff: float    = CHAT_BACKOFF

    # 2) PrivateAttr pour stocker l'instance interne
    _inner_llm: BaseLLM = PrivateAttr()

    def __init__(
        self,
        llm: BaseLLM,
        max_retries: int = CHAT_MAX_RETRIES,
        backoff: float = CHAT_BACKOFF,
        **kwargs: Any,
    ):
        # on ne passe rien à CustomLLM, on le configure dynamiquement
        super().__init__(**kwargs)
        self._inner_llm = llm
        self.max_retries = max_retries
        self.backoff = backoff

    @property
    def metadata(self) -> LLMMetadata:
        # on délègue les métadonnées au LLM original
        return self._inner_llm.metadata

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        kwargs.setdefault("max_tokens", MAX_TOKENS)
        for attempt in range(self.max_retries):
            try:
                # appel synchrone à l’API complete
                return self._inner_llm.complete(prompt, **kwargs)
            except SDKError as e:
                msg = str(e)
                if "429" in msg and attempt < self.max_retries - 1:
                    wait = self.backoff * (2 ** attempt)
                    time.sleep(wait)
                    continue
                raise

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        kwargs.setdefault("max_tokens", MAX_TOKENS)
        for attempt in range(self.max_retries):
            try:
                # appel synchrone (bloquant) au stream_complete
                return self._inner_llm.stream_complete(prompt, **kwargs)
            except SDKError as e:
                msg = str(e)
                if "429" in msg and attempt < self.max_retries - 1:
                    wait = self.backoff * (2 ** attempt)
                    time.sleep(wait)
                    continue
                raise
    # 2) **méthodes publiques requises par CustomLLM**
    def complete(self, prompt: str, **kw) -> CompletionResponse:
        """Signature obligatoire – appelle la version protégée avec retry."""
        return self._complete(prompt, **kw)

    def stream_complete(self, prompt: str, **kw) -> CompletionResponseGen:
        """Signature obligatoire – appelle la version protégée avec retry."""
        return self._stream_complete(prompt, **kw)

# ─── Chunking parallèle (fonction au module) ─────────────────────────────────
def _chunk_batch(docs_batch):
    for doc in docs_batch:
        doc.metadata = {"file_path": doc.metadata.get("file_path", "")}
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=64, include_metadata=True)
    return splitter.get_nodes_from_documents(docs_batch, show_progress=False)

# ── helper pour échapper les caractères spéciaux Markdown ──────────────
_MD_SPECIALS = re.compile(r'([\\`*_{}\[\]()#+\-.!>])')
def md_escape(txt: str) -> str:
    return _MD_SPECIALS.sub(r'\\\1', txt)

def _render_sources(raw: str, docs_root: str, host: str) -> str:
    """
    Convertit la sortie de `get_formatted_sources()` en liste à puces :
    - **facture_2024.pdf** – « Débit 408 pour 120 € … »
    """
    # chaque source est sur sa propre ligne
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            line = line[1:].lstrip()          # on enlève le chevron
        # petite tentative pour récupérer le chemin si présent entre parenthèses
        # ex : Source (Node id: …): chemin|texte
        parts = line.split("):", 1)
        texte = parts[1].strip() if len(parts) == 2 else line
        # nom cliquable ? ici on ne dispose pas du chemin, donc juste texte
        out.append(f"- {md_escape(texte)}")
    return "\n".join(out)

def _group_sources(source_nodes):
    """
    Regroupe les chunks par fichier et renvoie
    {file_path: [node_id1, node_id2, …]}
    """
    files: dict[str, list[str]] = {}
    for node_with_score in source_nodes:
        node = node_with_score.node
        fp   = node.metadata.get("file_path", "inconnu")
        files.setdefault(fp, []).append(node.node_id)
    return files

def _sources_markdown(files_by_path: dict[str, list[str]], docs_root: str) -> str:
    """
    Rend la section Markdown “Sources” sans afficher les node_ids.
    """
    if not files_by_path:
        return ""                   # pas de sources → rien à ajouter

    lines = ["\n\n---\n### Fichiers sources\n"]
    for i, fp in enumerate(files_by_path.keys(), 1):
        try:
            rel = Path(fp).relative_to(docs_root)
        except ValueError:
            # fp hors de docs_root : on garde le chemin absolu
            rel = Path(fp)
        name = rel.name or fp
        url  = f"{FILES_HOST}/{quote(str(rel))}"
        lines.append(f"{i}. [{name}]({url})")
    return "\n".join(lines)

class NoCondenseChatEngine(CondenseQuestionChatEngine):
    """Historique conservé, question transmise telle quelle (sync + async)."""
    def _condense_question(self, chat_history, question):       # sync
        return question
    async def _acondense_question(self, chat_history, question):  # async
        return question

# ─── Pipeline RAG ─────────────────────────────────────────────────────────────
class Pipeline:
    id   = pipeline_id

    def __init__(self) -> None:
        self.name = title
        self.version = version
        self.author = author
        self.persist_dir = os.getenv("VECTOR_INDEX_DIR", "/app/pipelines")
        self.docs_dir    = os.getenv("DOCS_DIR",        "/app/pipelines/docs")
        self.doc_root_dir = os.getenv("DOCS_ROOT_DIR",   "/app/pipelines/docs")
        self.meta_path   = os.path.join(self.persist_dir, "metadata.json")
        self.index_file  = os.path.join(self.persist_dir, "index.faiss")
        self.index: VectorStoreIndex | None = None
        self.mistral: Mistral | None        = None
        self._chat_sem = asyncio.Semaphore(CHAT_MAX_PARALLEL)

    @staticmethod
    def _distance_to_similarity(d: float) -> float:
        """Convertit une distance FAISS en similarité [0-1]."""
        return 1.0 / (1.0 + d)            # simple mais efficace


    def _retrieve_with_threshold(self, query: str):
        """Ajout d’un retry sur l’embedding de requête (429)."""
        for attempt in range(CHAT_MAX_RETRIES):
            try:
                k = BASE_TOP_K
                while k <= MAX_TOP_K:
                    retriever = self.index.as_retriever(similarity_top_k=k)
                    results   = retriever.retrieve(query)
                    if results:
                        best_sim = max(1.0 / (1.0 + r.score) for r in results)
                        if best_sim >= SIM_THRESHOLD:
                            return results
                    k += BASE_TOP_K
                return results
            except SDKError as e:
                if "429" in str(e):
                    wait = CHAT_BACKOFF * (2 ** attempt)
                    logger.warning(f"429 rate-limit retrieve – back-off {wait}s…")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Retrieval échoué après {CHAT_MAX_RETRIES} tentatives (rate-limit)")

    def _load_meta(self) -> dict:
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                return json.load(f)
        return {}

    def _scan_docs(self) -> dict:
        out = {}
        for root, _, files in os.walk(self.docs_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in EXTENSIONS:
                    path = os.path.join(root, fn)
                    out[path] = os.path.getmtime(path)
        return out

    async def _chat_stream_with_retry(
        self,
        messages: list[dict],
        model: str = "mistral-large-latest",
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ):
        async with self._chat_sem:
            for attempt in range(CHAT_MAX_RETRIES):
                try:
                    return await asyncio.to_thread(
                        self.mistral.chat.stream,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        **kwargs,
                    )
                except SDKError as e:
                    if "429" in str(e):
                        wait = CHAT_BACKOFF * (2 ** attempt)
                        logger.warning(f"429 rate-limit – back-off {wait:.1f}s…")
                        await asyncio.sleep(wait)
                        continue
                    raise
            raise RuntimeError(
                f"Chat échoué après {CHAT_MAX_RETRIES} tentatives (rate-limit)"
            )

    @staticmethod
    def reconstruct_metadata(storage_context: StorageContext, meta_path: str):
        """Reconstruit le fichier metadata.json à partir du DocumentStore."""
        logger.info("🔄 Reconstruction complète du metadata.json depuis DocumentStore")
        new_meta = {}
        doc_count = 0
        
        try:
            # Construction du dictionnaire de métadonnées
            for doc_id, node in storage_context.docstore.docs.items():
                fp = node.metadata.get("file_path", "")
                if not fp:
                    logger.warning(f"Node {doc_id} sans file_path, ignoré.")
                    continue
                
                doc_count += 1
                if fp not in new_meta:
                    try:
                        mtime = os.path.getmtime(fp)
                    except FileNotFoundError:
                        logger.warning(f"Fichier source absent {fp}, utilisation timestamp actuel.")
                        mtime = time.time()  # fallback fiable
                    new_meta[fp] = {"mtime": mtime, "doc_ids": []}
                
                # Éviter les doublons potentiels
                if doc_id not in new_meta[fp]["doc_ids"]:
                    new_meta[fp]["doc_ids"].append(doc_id)
            
            # Écriture atomique avec fichier temporaire
            temp_path = f"{meta_path}.tmp"
            with open(temp_path, "w") as f:
                json.dump(new_meta, f, indent=2)
            
            # Remplacement atomique
            os.replace(temp_path, meta_path)
            
            logger.info(f"✅ Reconstruction de metadata.json terminée. {doc_count} documents indexés pour {len(new_meta)} fichiers.")
            return new_meta
        except Exception as e:
            logger.error(f"❌ Erreur lors de la reconstruction des métadonnées: {str(e)}")
            # Nettoyage du fichier temporaire si nécessaire
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise


    async def _load_file(self, path: str) -> List:
        """Choisit dynamiquement le reader le plus rapide selon l’extension."""
        ext = os.path.splitext(path)[1].lower()
        logger.info(f"      ⏳ Charger {path}")

        # on mappe chaque extension sur son reader
        if ext == ".pdf":
            reader = PyMuPDFReader()
        elif ext == ".docx":
            reader = DocxReader()
        elif ext == ".pptx":
            reader = PptxReader()
        elif ext == ".csv":
            reader = CSVReader()
        elif ext in {".xls", ".xlsx", ".xlsm"}:
            # les trois formats Excel passent par Unstructured (tableaux)
            reader = UnstructuredReader()
        elif ext == ".html":
            reader = HTMLTagReader()
        elif ext == ".md":
            reader = MarkdownReader()
        else:
            # txt, rtf, hwp, xml, etc.
            reader = FlatReader()

        try:
            # on passe TOUJOURS une liste de chemins, en thread, avec timeout
            docs = await asyncio.wait_for(
                asyncio.to_thread(reader.load_data, path),
                timeout=120  # 2 minutes max par fichier
            )
            for doc in docs:
                doc.metadata["file_path"] = path
            logger.info(f"      ✅ OK {path}")
            return docs

        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout sur {path}")
            return []

        except Exception as e:
            logger.error(f"      ❌ Échec {path} : {e}")
            return []

    async def on_startup(self) -> None:
        logger.info("🚀 Démarrage pipeline RAG")

        # ─── FORCE_INITIAL ────────────────────────────────────────────────────────
        force_initial = os.getenv("FORCE_INITIAL", "false").lower() in ("1", "true", "yes")
        if force_initial:
            logger.info("🔄 Force initial indexation demandée via FORCE_INITIAL")
        # ───────────────────────────────────────────────────────────────────────────

        # 1) ThreadPool pour le parsing
        loop     = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_LOADERS)
        loop.set_default_executor(executor)
        self._loop = loop
        logger.info(f"🔧 ThreadPoolExecutor(max_workers={MAX_LOADERS})")

        # 2) Mistral & embedding
        self.mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        Settings.embed_model = RetryingMistralEmbedding(
            model_name="mistral-embed",
            api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.7,
            top_p=0.9,
        )
        # 3) LLM synchrone : on construit MistralAI puis on le wrappe
        sync_mistral = MistralAI(
            model="mistral-large-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        Settings.llm = RetryingLLM(llm=sync_mistral)

        os.makedirs(self.persist_dir, exist_ok=True)

        old_meta = {} if force_initial else self._load_meta()
        file_exists = False if force_initial else os.path.exists(self.meta_path)

        # Si metadata.json vide ou inexistant, mais l'index existe :
        if (not old_meta) and os.path.exists(os.path.join(self.persist_dir, "docstore.json")):
            logger.warning("⚠️ metadata.json vide ou absent détecté, reconstruction en cours.")
            ds = SimpleDocumentStore.from_persist_path(os.path.join(self.persist_dir, "docstore.json"))
            is_ = SimpleIndexStore.from_persist_path(os.path.join(self.persist_dir, "index_store.json"))
            vs = FaissVectorStore.from_persist_dir(self.persist_dir)
            ctx = StorageContext.from_defaults(docstore=ds, index_store=is_, vector_store=vs, persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context=ctx)

            Pipeline.reconstruct_metadata(storage_context=ctx, meta_path=self.meta_path)
            old_meta = self._load_meta()

        # 3) Initial vs incrémental
        if file_exists and old_meta:
            logger.info("🔄 Mode incrémental détecté")

            # 3.1) reload index
            ds = SimpleDocumentStore.from_persist_path(os.path.join(self.persist_dir, "docstore.json"))
            is_ = SimpleIndexStore.from_persist_path(os.path.join(self.persist_dir, "index_store.json"))
            vs = FaissVectorStore.from_persist_dir(self.persist_dir)
            ctx = StorageContext.from_defaults(
                docstore=ds, index_store=is_, vector_store=vs, persist_dir=self.persist_dir
            )
            self.index = load_index_from_storage(storage_context=ctx)

            # 3.2) diff
            old_meta = self._load_meta()
            new_meta = self._scan_docs()
            old_set, new_set = set(old_meta), set(new_meta)
            removed = old_set - new_set

            added   = new_set - old_set
            updated = {p for p in old_set & new_set if new_meta[p] > old_meta[p]["mtime"]}

            logger.info(f"   • suppressions : {len(removed)}")
            logger.info(f"   • ajouts       : {len(added)}")
            logger.info(f"   • modifs       : {len(updated)}")

            # 3.3) delete & reindex
            to_delete = []
            # On stocke temporairement les entrées à supprimer sans pop immédiat
            for p in removed | updated:
                to_delete.extend(old_meta[p]["doc_ids"])

            if to_delete:
                logger.warning(
                    "delete_nodes() ignoré : HNSW ne supporte pas remove_ids. "
                    "Les vecteurs restent dans l’index mais seront filtrés."
                )
            else:
                logger.info("   → aucun chunk à supprimer")

            to_index = list(added | updated)
            if to_index:
                logger.info(f"   → réindexation de {len(to_index)} fichiers")
                tasks = [self._load_file(path) for path in to_index]
                results = await asyncio.gather(*tasks)
                docs = [doc for batch in results for doc in batch]

                # Découpage parallèle
                batches = [docs[i::CPU_COUNT] for i in range(CPU_COUNT)]
                with ThreadPoolExecutor(max_workers=MAX_LOADERS) as chunk_exec:
                    futures = [
                        asyncio.get_running_loop().run_in_executor(chunk_exec, _chunk_batch, batch)
                        for batch in batches
                    ]
                    chunked_lists = await asyncio.gather(*futures)

                nodes = [node for sub in chunked_lists for node in sub]
                self.index.insert_nodes(nodes)
                logger.info(f"   → total chunks produits : {len(nodes)}")

                # Reconstruction explicite des entrées modifiées ou ajoutées
                for path in updated | added:
                    old_meta[path] = {"mtime": new_meta[path], "doc_ids": []}

                for d in nodes:
                    fp = d.metadata.get("file_path", "")
                    if fp not in old_meta:
                        old_meta[fp] = {"mtime": new_meta.get(fp, os.path.getmtime(fp)), "doc_ids": []}
                    old_meta[fp]["doc_ids"].append(d.doc_id)

                # ici seulement, supprimer définitivement les entrées des fichiers supprimés
                for p in removed:
                    old_meta.pop(p, None)

                # persist (incrémental)
                self.index.storage_context.persist(persist_dir=self.persist_dir)
                with open(self.meta_path, "w") as f:
                    json.dump(old_meta, f)
                logger.info("✅ Synchronisation terminée")
            else:
                logger.info("   → rien à indexer")
                # Même si rien à indexer, on supprime définitivement les fichiers supprimés
                for p in removed:
                    old_meta.pop(p, None)
                # persist
                with open(self.meta_path, "w") as f:
                    json.dump(old_meta, f)
                logger.info("✅ Synchronisation terminée sans réindexation")
        else:
            logger.info("🆕 Création initiale complète")

            # 3.1) lister fichiers
            files = []
            for r, _, fs in os.walk(self.docs_dir):
                for fn in fs:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in EXTENSIONS:
                        files.append(os.path.join(r, fn))
            logger.info(f"   → {len(files)} fichiers détectés")

            # 3.2) parser en parallèle
            tasks   = [self._load_file(p) for p in files]
            results = await asyncio.gather(*tasks)
            docs0   = [d for sub in results for d in sub]

            # 3.3) chunking parallèle avec ThreadPoolExecutor
            batches = [docs0[i::CPU_COUNT] for i in range(CPU_COUNT)]
            with ThreadPoolExecutor(max_workers=CPU_COUNT) as chunk_exec:
                futures = [
                    loop.run_in_executor(chunk_exec, _chunk_batch, batch)
                    for batch in batches
                ]
                results = await asyncio.gather(*futures)

            docs = [node for sub in results for node in sub]
            logger.info(f"   → split en {len(docs)} chunks (parallélisé)")

            # ——— Filtrage des chunks trop petits ———
            before = len(docs)
            docs = [d for d in docs if len(d.text.strip()) >= MIN_CHUNK_L]
            logger.info(f"   → filtré {before - len(docs)} chunks < {MIN_CHUNK_L} caractères, reste {len(docs)}")

            # 3.4) indexation (HNSWFlat + batching)
            hnsw_index      = faiss.IndexHNSWFlat(EMBED_DIM, HNSW_M)
            hnsw_index.hnsw.efConstruction = HNSW_EF_CONS
            hnsw_index.hnsw.efSearch = HNSW_EF_SEARCH

            vs = FaissVectorStore(faiss_index=hnsw_index)
            ctx = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore(),
                vector_store=vs,
                persist_dir=self.persist_dir,
            )

            # 1) première batch pour créer l'index
            batch_size  = BATCH_SIZE
            first_batch = docs[:batch_size]
            self.index  = VectorStoreIndex(
                nodes=first_batch,
                storage_context=ctx,
            )
            logger.info(f"   → batch 1 : indexé {len(first_batch)} chunks")

            meta = {}
            for d in first_batch:
                fp = d.metadata.get("file_path", "")
                if not fp:
                    logger.warning(f"Chunk {d.doc_id!r} sans chemin de fichier, ignoré")
                    continue
                if fp not in meta:
                    try:
                        mtime = os.path.getmtime(fp)
                    except FileNotFoundError:
                        logger.warning(f"Fichier source absent {fp!r}, utilisation timestamp actuel")
                        mtime = time.time()  # fallback si le fichier est absent
                    meta[fp] = {
                        "mtime": mtime,
                        "doc_ids": []
                    }
                meta[fp]["doc_ids"].append(d.doc_id)

            # 2) batchs suivants via insert()
            for i in range(batch_size, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                self.index.insert_nodes(batch)
                logger.info(f"   → batch {i//batch_size + 1} : indexé {len(batch)} chunks")

                for d in batch:
                    fp = d.metadata.get("file_path", "")
                    if not fp or not os.path.isfile(fp):
                        logger.warning(
                            f"Chemin invalide ou introuvable pour le chunk {d.doc_id!r} : {fp!r}, on l’ignore"
                        )
                        continue
                    if fp not in meta:
                        meta[fp] = {
                            "mtime": os.path.getmtime(fp),
                            "doc_ids": []  # <-- Liste pour tous les chunks du fichier
                        }
                    meta[fp]["doc_ids"].append(d.doc_id) 

                # checkpoint tous les 5 batchs
                if (i // batch_size) % 5 == 0:
                    self.index.storage_context.persist(persist_dir=self.persist_dir)
                    with open(self.meta_path, "w") as f:
                        json.dump(meta,f)
                    logger.info("   → checkpoint persisted")

            # 3) persistance finale
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            logger.info(f"   → indexé au total {len(docs)} chunks")

            if self.index.docstore.docs:
                sample_id = next(iter(self.index.docstore.docs))
                sample_node = self.index.docstore.docs[sample_id]
                try:
                    emb = Settings.embed_model.get_query_embedding(sample_node.text)
                    norm = sum(x*x for x in emb)**0.5
                    logger.info(f"TEST EMBEDDING - Dimensions: {len(emb)}, Norme: {norm:.2f}")
                except Exception as e:
                    logger.error(f"Échec du test d'embedding : {str(e)}")

            with open(self.meta_path, "w") as f:
                json.dump(meta,f)
            logger.info("✅ Création initiale terminée")

    async def on_shutdown(self) -> None:
        pass

    # ─── Conversation RAG + historique ────────────────────────────────────
    def pipe(self, user_message: str, model_id: str, messages, body):
        """
        RAG streaming avec historique utilisateur (CondenseQuestionChatEngine)
        + prompt système strict conservé dans le QueryEngine.
        """
        if user_message.lstrip().startswith("### Task:"):
            yield ""
            return

        # 1) Prompt système
        system_strict = (
            "Si tu ne connais pas la réponse, indique-le clairement."
            "En cas d'incertitude, demande des précisions à l'utilisateur."
            "Réponds en t'appuyant *principalement* sur le contexte."
            "Pour chaque point de ta réponse, fournis d’abord un extrait de 2–3 phrases."
            "IMPORTANT : Réponds toujours dans la même langue que la requête."
            "Si la requête est en français, TA RÉPONSE DOIT ÊTRE EXCLUSIVEMENT EN FRANÇAIS."
            "Si le contexte est illisible ou de mauvaise qualité, informe l'utilisateur."
            "Si l’information est incomplète, précise-le."
            "Si elle est absente, réponds « Je ne sais pas »."
            "Ne rajoute aucune autre information hors contexte."
            "Assure-toi que les citations sont concises et directement liées à l'information fournie."
        )

        qa_prompt = PromptTemplate(
            f"""{system_strict}
---------------------
{{context_str}}
---------------------
Question : {{query_str}}

Réponds à la question le plus précisément possible.
"""
        )

        try:
            # 2) QueryEngine (RAG)
            qe = self.index.as_query_engine(
                llm=Settings.llm,
                similarity_top_k=BASE_TOP_K,
                text_qa_template=qa_prompt,
                include_source_nodes=True,
                response_synthesizer=get_response_synthesizer(
                    response_mode="tree_summarize", streaming=True
                ),
                streaming=True,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=1.0 - SIM_THRESHOLD)
                ],
            )

            # 3) Mémoire : conversion des messages dict → ChatMessage puis set()
            chat_history = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in messages
                if m["role"] in {"user", "assistant"}         # on exclut les system existants
            ]
            memory = ChatMemoryBuffer(token_limit=4096)
            memory.set(chat_history)

            dummy_prompt = PromptTemplate("{question}")

            # 4) Chat engine = condensation + historique
            chat_engine = NoCondenseChatEngine(
                query_engine=qe,
                memory=memory,
                llm=Settings.llm,
                condense_question_prompt=dummy_prompt,
                verbose=False,
            )

            # 5) Streaming réponse
            resp = chat_engine.stream_chat(user_message)
            for tok in resp.response_gen:
                yield getattr(tok, "delta", tok)

            # 6) Section “Fichiers sources”
            files = _group_sources(resp.source_nodes)
            yield _sources_markdown(files, self.doc_root_dir)

        except Exception as e:
            yield f"Erreur lors du streaming de la réponse: {str(e)}"
