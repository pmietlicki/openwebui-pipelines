# pipelines/mistral_passthrough.py
import os, time, random, logging
from typing import List, Dict, Any, Optional, Tuple

from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

# ── Métadonnées pipeline
title   = os.getenv("PIPELINE_TITLE", "Mistral")
version = "1.3.0"
author  = "PMietlicki"
pipeline_id = title.lower().replace(" ", "_")

# ── ENV (tune au besoin)
CHAT_MAX_RETRIES  = int(os.getenv("CHAT_MAX_RETRIES", "6"))
CHAT_BACKOFF      = float(os.getenv("CHAT_BACKOFF", "1.0"))

MODEL             = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
API_KEY           = os.getenv("MISTRAL_API_KEY")

TEMP              = float(os.getenv("MISTRAL_TEMPERATURE", "0.3"))
TOP_P             = float(os.getenv("MISTRAL_TOP_P", "0.9"))
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", "2048"))
OCR_MAX_TOKENS    = int(os.getenv("OCR_MAX_TOKENS", str(MAX_TOKENS)))
OCR_OUTPUT_MODE   = os.getenv("OCR_OUTPUT_MODE", "auto").strip().lower()  # 'auto' | 'tika' | 'mistral'

SYSTEM_PROMPT     = os.getenv("RAW_SYSTEM_PROMPT", "Tu es un assistant utile. Réponds en français si possible.")

# Fallbacks (séparés par virgules)
DEFAULT_FALLBACKS = "mistral-small-latest,mistral-medium-latest,open-mistral-nemo"
FALLBACKS = [m.strip() for m in os.getenv("MISTRAL_FALLBACK_MODELS", DEFAULT_FALLBACKS).split(",") if m.strip()]

# Cooldown (en secondes) quand le tier est saturé (code 3505)
CAPACITY_COOLDOWN_S = int(os.getenv("MISTRAL_CAPACITY_COOLDOWN_S", "45"))
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}

# Debug (log des events du stream)
DEBUG_EVENTS = os.getenv("DEBUG_MISTRAL_STREAM", "0").lower() in ("1","true","yes")
LOG_FIRST_TOKEN = os.getenv("LOG_FIRST_TOKEN", "0").lower() in ("1","true","yes")

# ── Mode OCR/plein texte via découpage et multi‑appels
OCR_MODE_ENABLED = os.getenv("OCR_MODE_ENABLED", "1").lower() in ("1","true","yes")
OCR_KEYWORDS = [k.strip().lower() for k in os.getenv("OCR_KEYWORDS", "ocr,transcription,texte brut,plein texte,extraire texte,transcrire,texte complet").split(",") if k.strip()]
OCR_MIN_DOC_CHARS = int(os.getenv("OCR_MIN_DOC_CHARS", "4000"))
OCR_TRIGGER_MODE = os.getenv("OCR_TRIGGER_MODE", "keyword").strip().lower()  # 'auto' ou 'keyword'
_approx_chars_per_token = 4
_safe_ratio = 0.7
_default_chunk = int(MAX_TOKENS * _approx_chars_per_token * _safe_ratio) if MAX_TOKENS else 6000
OCR_CHUNK_SIZE_CHARS = int(os.getenv("OCR_CHUNK_SIZE_CHARS", str(max(1000, _default_chunk))))
OCR_CHUNK_OVERLAP_CHARS = int(os.getenv("OCR_CHUNK_OVERLAP_CHARS", "0"))
OCR_VERBATIM_SYSTEM_PROMPT = os.getenv(
    "OCR_VERBATIM_SYSTEM_PROMPT",
    "Tu es un transcripteur OCR. Répète exactement le texte reçu, sans commentaire, format ni correction. Ne rien ajouter ni supprimer."
)

log = logging.getLogger("mistral_passthrough")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ───────────────────────── Helpers ───────────────────────── #

def _sleep_with_backoff(attempt: int, base: float, retry_after: Optional[float] = None, cap: float = 30.0):
    """Respecte Retry-After si présent, sinon backoff exponentiel + jitter."""
    if retry_after and retry_after > 0:
        log.warning(f"429 → attente Retry-After {retry_after:.2f}s (attempt={attempt+1})")
        time.sleep(retry_after)
        return
    wait = min(base * (2 ** attempt), cap) + random.uniform(0, 0.5)
    log.warning(f"429 → attente backoff {wait:.2f}s (attempt={attempt+1})")
    time.sleep(wait)

def _safe_char_limit_for_tokens(tokens: int, ratio: float = _safe_ratio) -> int:
    """Calcule un plafond prudent de caractères à restituer pour un nombre de tokens."""
    try:
        return max(1000, int(tokens * _approx_chars_per_token * ratio))
    except Exception:
        return 2000

def _extract_text_blocks(content: Any) -> str:
    """Récupère le texte d'un bloc OpenWebUI (list ou str), y compris Tika/attachments."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            btype = (b.get("type") or "").lower()
            allowed = {
                "text", "plain_text", "ocr_text", "document", "file", "markdown", "tika_text", "pdf_text", "doc_text"
            }
            if btype in allowed:
                t = b.get("text") or b.get("content") or (isinstance(b.get("data"), dict) and b.get("data", {}).get("text")) or ""
                if t:
                    parts.append(str(t))
        return "\n".join(parts)
    return ""

def _get_document_text(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Extrait le texte document depuis les messages 'user' (inclut Tika/attachments)."""
    if not messages:
        return None
    buf: List[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        t = _extract_text_blocks(m.get("content"))
        if t:
            buf.append(t)
    if buf:
        joined = "\n".join(buf)
        log.info("Detecté texte document côté messages user: ~%d caractères", len(joined))
        return joined
    return None

def _wants_direct_passthrough(user_message: str) -> bool:
    """Heuristique simple pour demander le texte brut (Tika) sans Mistral."""
    um = (user_message or "").lower()
    direct_keywords = [
        "ocr", "transcription", "extraire", "extraction", "texte brut",
        "texte complet", "plaintext", "verbatim"
    ]
    return any(k in um for k in direct_keywords)

def _is_ocr_request(user_message: str, messages: List[Dict[str, Any]]) -> bool:
    """Détecte une demande d'OCR/texte brut selon la stratégie et un document volumineux."""
    if not OCR_MODE_ENABLED:
        return False
    doc_text = _get_document_text(messages or [])
    if not doc_text:
        return False
    if len(doc_text) < OCR_MIN_DOC_CHARS:
        return False

    mode = OCR_TRIGGER_MODE
    if mode == "auto":
        return True
    # mode 'keyword' (par défaut)
    um = (user_message or "").lower()
    keyword_hit = any(k in um for k in OCR_KEYWORDS)
    return keyword_hit

def _split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Découpe le texte en chunks ~chunk_size (caractères), avec overlap optionnel."""
    if chunk_size <= 0:
        chunk_size = 4000
    if overlap < 0:
        overlap = 0

    paragraphs = text.split("\n")
    chunks: List[str] = []
    buf: List[str] = []
    curr_len = 0

    for p in paragraphs:
        add_len = len(p) + 1  # inclure le saut de ligne
        if curr_len == 0 or curr_len + add_len <= chunk_size:
            buf.append(p)
            curr_len += add_len
        else:
            chunk = "\n".join(buf)
            chunks.append(chunk)

            # préfixe overlap dans le prochain buffer (en caractères)
            overlap_str = chunk[-overlap:] if overlap > 0 and len(chunk) > overlap else ""
            buf = []
            curr_len = 0
            if overlap_str:
                buf.append(overlap_str)
                curr_len += len(overlap_str)

            buf.append(p)
            curr_len += add_len

    if buf:
        chunks.append("\n".join(buf))

    # Ajuster si un chunk dépasse chunk_size (paragraphe géant)
    final: List[str] = []
    for c in chunks:
        if len(c) <= chunk_size:
            final.append(c)
            continue
        i = 0
        L = len(c)
        while i < L:
            end = min(i + chunk_size, L)
            segment = c[i:end]
            final.append(segment)
            i = end - overlap if overlap > 0 else end
            if i < 0:
                i = 0
    return final

def _build_verbatim_msgs(chunk_text: str) -> List[Dict[str, str]]:
    """Construit des messages pour demander une reproduction verbatim du chunk."""
    msgs: List[Dict[str, str]] = []
    if OCR_VERBATIM_SYSTEM_PROMPT:
        msgs.append({"role": "system", "content": OCR_VERBATIM_SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": chunk_text})
    return msgs

def _to_chat_messages(user_message: str, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Transforme l'historique OpenWebUI en format Mistral Chat API."""
    out: List[Dict[str, str]] = []
    if SYSTEM_PROMPT:
        out.append({"role": "system", "content": SYSTEM_PROMPT})

    for m in messages or []:
        role = m.get("role")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join([b.get("text", "") for b in content if b.get("type") == "text"])
        if role in {"system", "user", "assistant"} and content:
            out.append({"role": role, "content": content})

    if user_message and not (messages and (messages[-1].get("role") == "user")):
        out.append({"role": "user", "content": user_message})

    return out

def _parse_mistral_error(e: SDKError) -> Tuple[bool, Optional[str], Optional[float], str]:
    """Retourne (is_429, code, retry_after, message lisible)."""
    msg = str(e) or ""
    is_429 = "429" in msg or "Too Many Requests" in msg
    retry_after: Optional[float] = None
    code: Optional[str] = None
    body_msg = msg

    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            hdrs = getattr(resp, "headers", None)
            if hdrs:
                ra = hdrs.get("Retry-After") or hdrs.get("retry-after")
                if ra:
                    try:
                        retry_after = float(ra)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            j = resp.json()
            code = j.get("code") or (j.get("error") or {}).get("code") or j.get("type")
            body_msg = j.get("message") or (j.get("error") or {}).get("message") or msg
        except Exception:
            pass

    return is_429, code, retry_after, body_msg


# ───────────────────────── Pipeline ───────────────────────── #

class Pipeline:
    id = pipeline_id

    def __init__(self):
        self.name = title
        self.version = version
        self.author = author
        self.client: Optional[Mistral] = None

    async def on_startup(self):
        if not API_KEY:
            raise RuntimeError("MISTRAL_API_KEY manquant")
        self.client = Mistral(api_key=API_KEY)
        log.info("✅ Mistral brut (stream) prêt (model=%s; fallbacks=%s)", MODEL, ",".join(FALLBACKS))

    async def on_shutdown(self):
        pass

    def _stream_once(self, msgs: List[Dict[str, str]], model: str, max_tokens_param: Optional[int] = None):
        """
        Ouvre un stream Mistral et yield les morceaux de texte au fil de l'eau.
        Compatible avec:
        1) chunk.data.choices[0].delta.content (forme actuelle)
        2) événements .type == 'message.delta' / 'content.delta' (forme legacy)
        Si aucun delta n'est reçu, renvoie la réponse finale (fallback).
        """
        assert self.client is not None

        yielded_any = False
        first_logged = False

        with self.client.chat.stream(
            model=model,
            messages=msgs,
            temperature=TEMP,
            top_p=TOP_P,
            max_tokens=(max_tokens_param or MAX_TOKENS),
        ) as stream:
            for chunk in stream:
                if DEBUG_EVENTS:
                    try:
                        et = getattr(chunk, "type", None)
                        log.info("stream event type=%s", et or "data")
                    except Exception:
                        pass

                data = getattr(chunk, "data", None)
                if data is not None:
                    try:
                        choice = data.choices[0]
                        delta = choice.delta
                        content = getattr(delta, "content", None)

                        if isinstance(content, str):
                            if content:
                                yielded_any = True
                                if LOG_FIRST_TOKEN and not first_logged:
                                    log.info("first token (modern): %r", content[:120])
                                    first_logged = True
                                yield content
                        elif isinstance(content, list):
                            for part in content:
                                ptype = getattr(part, "type", None) or (isinstance(part, dict) and part.get("type"))
                                if ptype == "text":
                                    text = getattr(part, "text", None) or (isinstance(part, dict) and part.get("text"))
                                    if text:
                                        yielded_any = True
                                        if LOG_FIRST_TOKEN and not first_logged:
                                            log.info("first token (modern-list): %r", text[:120])
                                            first_logged = True
                                        yield text
                    except Exception:
                        pass
                    continue

                etype = getattr(chunk, "type", None)
                if etype in ("message.delta", "content.delta"):
                    delta = getattr(chunk, "delta", None)
                    content = getattr(delta, "content", None) if delta else None

                    if isinstance(content, str):
                        if content:
                            yielded_any = True
                            if LOG_FIRST_TOKEN and not first_logged:
                                log.info("first token (legacy): %r", content[:120])
                                first_logged = True
                            yield content
                    elif isinstance(content, list):
                        for part in content:
                            ptype = getattr(part, "type", None) or (isinstance(part, dict) and part.get("type"))
                            if ptype == "text":
                                text = getattr(part, "text", None) or (isinstance(part, dict) and part.get("text"))
                                if text:
                                    yielded_any = True
                                    if LOG_FIRST_TOKEN and not first_logged:
                                        log.info("first token (legacy-list): %r", text[:120])
                                        first_logged = True
                                    yield text

            try:
                final = stream.get_final_response()
                if not yielded_any and final and getattr(final, "choices", None):
                    msg_content = final.choices[0].message.content
                    if isinstance(msg_content, str):
                        if msg_content:
                            yield msg_content
                    elif isinstance(msg_content, list):
                        buf = []
                        for part in msg_content:
                            ptype = getattr(part, "type", None) or (isinstance(part, dict) and part.get("type"))
                            if ptype == "text":
                                text = getattr(part, "text", None) or (isinstance(part, dict) and part.get("text"))
                                if text:
                                    buf.append(text)
                        if buf:
                            yield "".join(buf)
            except Exception:
                pass

    def _stream_with_retry_and_fallback(self, msgs: List[Dict[str, str]], max_tokens_override: Optional[int] = None):
        chain: List[str] = []
        seen = set()
        for m in [MODEL] + FALLBACKS:
            if m and m not in seen:
                seen.add(m)
                chain.append(m)

        last_err: Optional[Exception] = None

        for model in chain:
            now = time.time()
            until = _MODEL_COOLDOWN_UNTIL.get(model, 0)
            if now < until:
                log.info("↷ Skip %s (capacity cooldown encore %.0fs)", model, until - now)
                continue

            log.info("→ Stream Mistral model=%s", model)
            attempt = 0

            while attempt < CHAT_MAX_RETRIES:
                try:
                    for chunk in self._stream_once(msgs, model, max_tokens_param=max_tokens_override):
                        yield chunk
                    return
                except SDKError as e:
                    is_429, code, retry_after, body_msg = _parse_mistral_error(e)

                    if is_429 and (code == "3505" or code == "service_tier_capacity_exceeded"):
                        log.warning("Modèle %s saturé pour ton tier (code=%s) → fallback.", model, code)
                        _MODEL_COOLDOWN_UNTIL[model] = time.time() + CAPACITY_COOLDOWN_S
                        last_err = e
                        break

                    if is_429:
                        _sleep_with_backoff(attempt, CHAT_BACKOFF, retry_after)
                        attempt += 1
                        last_err = e
                        continue

                    last_err = e
                    log.error("Erreur Mistral non récupérable (model=%s): %s", model, body_msg)
                    raise

        if last_err:
            raise last_err
        raise RuntimeError("Échec inconnu sans erreur retournée")

    def pipe(self, user_message: str, model_id: str, messages, body):
        """
        Passthrough Mistral : pas de RAG.
        Ajoute: 
        - mode Tika direct (passthrough, sans appel Mistral) si demandé
        - mode OCR/texte brut via découpage et multi‑appels, streaming continu
        """
        # Récupère le texte document (Tika/attachments)
        doc_text = None
        try:
            doc_text = _get_document_text(messages or [])
        except Exception:
            doc_text = None

        # 1) Mode Tika direct (passthrough), sans appel Mistral
        #    Déclenché si OCR_OUTPUT_MODE='tika' ou 'auto' + mots-clés de verbatim.
        try:
            if doc_text:
                direct_mode = (
                    OCR_OUTPUT_MODE == "tika" or (OCR_OUTPUT_MODE == "auto" and _wants_direct_passthrough(user_message))
                )
                if direct_mode:
                    parts = _split_text_into_chunks(doc_text, OCR_CHUNK_SIZE_CHARS, 0)
                    log.info(
                        "OCR Tika passthrough: %d caractères, %d chunks (chunk=%d chars)",
                        len(doc_text), len(parts), OCR_CHUNK_SIZE_CHARS,
                    )
                    for idx, chunk_text in enumerate(parts, 1):
                        log.info("Tika chunk %d/%d (len=%d)", idx, len(parts), len(chunk_text))
                        # Passthrough: on envoie directement le texte sans appel API
                        yield chunk_text
                    return
        except Exception as e:
            log.error("Tika passthrough erreur: %s", e)

        # 2) Mode OCR/verbatim via Mistral : texte volumineux issu de Tika + découpage
        try:
            if OCR_MODE_ENABLED and doc_text and _is_ocr_request(user_message, messages or []):
                # Calibrer la taille des chunks selon les tokens de sortie disponibles
                safe_char_limit = _safe_char_limit_for_tokens(OCR_MAX_TOKENS)
                effective_chunk_size = min(OCR_CHUNK_SIZE_CHARS, safe_char_limit)
                parts = _split_text_into_chunks(doc_text, effective_chunk_size, OCR_CHUNK_OVERLAP_CHARS)
                log.info(
                    "OCR mode actif: %d caractères, %d chunks (chunk=%d chars, max_tokens=%d)",
                    len(doc_text), len(parts), effective_chunk_size, OCR_MAX_TOKENS,
                )
                for idx, chunk_text in enumerate(parts, 1):
                    log.info("OCR chunk %d/%d (len=%d)", idx, len(parts), len(chunk_text))
                    verbatim_msgs = _build_verbatim_msgs(chunk_text)
                    for token in self._stream_with_retry_and_fallback(verbatim_msgs, max_tokens_override=OCR_MAX_TOKENS):
                        yield token
                return
        except Exception as e:
            log.error("OCR mode erreur: %s", e)

        # Mode normal (pas de RAG) : streaming avec retry + fallback modèle
        msgs = _to_chat_messages(user_message, messages)
        try:
            for token in self._stream_with_retry_and_fallback(msgs, max_tokens_override=None):
                yield token
        except Exception as e:
            yield f"Erreur Mistral: {e}"
import os, time, random, logging
from typing import List, Dict, Any, Optional, Tuple

from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

# ── Métadonnées pipeline
title   = os.getenv("PIPELINE_TITLE", "Mistral")
version = "1.3.0"
author  = "PMietlicki"
pipeline_id = title.lower().replace(" ", "_")

# ── ENV (tune au besoin)
CHAT_MAX_RETRIES  = int(os.getenv("CHAT_MAX_RETRIES", "6"))
CHAT_BACKOFF      = float(os.getenv("CHAT_BACKOFF", "1.0"))

MODEL             = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
API_KEY           = os.getenv("MISTRAL_API_KEY")

TEMP              = float(os.getenv("MISTRAL_TEMPERATURE", "0.3"))
TOP_P             = float(os.getenv("MISTRAL_TOP_P", "0.9"))
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", "2048"))

SYSTEM_PROMPT     = os.getenv("RAW_SYSTEM_PROMPT", "Tu es un assistant utile. Réponds en français si possible.")

# Fallbacks (séparés par virgules)
DEFAULT_FALLBACKS = "mistral-small-latest,mistral-medium-latest,open-mistral-nemo"
FALLBACKS = [m.strip() for m in os.getenv("MISTRAL_FALLBACK_MODELS", DEFAULT_FALLBACKS).split(",") if m.strip()]

# Cooldown (en secondes) quand le tier est saturé (code 3505)
CAPACITY_COOLDOWN_S = int(os.getenv("MISTRAL_CAPACITY_COOLDOWN_S", "45"))
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}

# Debug (log des events du stream)
DEBUG_EVENTS = os.getenv("DEBUG_MISTRAL_STREAM", "0").lower() in ("1","true","yes")

log = logging.getLogger("mistral_passthrough")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ───────────────────────── Helpers ───────────────────────── #

def _sleep_with_backoff(attempt: int, base: float, retry_after: Optional[float] = None, cap: float = 30.0):
    """Respecte Retry-After si présent, sinon backoff exponentiel + jitter."""
    if retry_after and retry_after > 0:
        log.warning(f"429 → attente Retry-After {retry_after:.2f}s (attempt={attempt+1})")
        time.sleep(retry_after)
        return
    wait = min(base * (2 ** attempt), cap) + random.uniform(0, 0.5)
    log.warning(f"429 → attente backoff {wait:.2f}s (attempt={attempt+1})")
    time.sleep(wait)

def _to_chat_messages(user_message: str, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Transforme l'historique OpenWebUI en format Mistral Chat API."""
    out: List[Dict[str, str]] = []
    if SYSTEM_PROMPT:
        out.append({"role": "system", "content": SYSTEM_PROMPT})

    for m in messages or []:
        role = m.get("role")
        content = m.get("content", "")
        if isinstance(content, list):
            # concatène les blocs textuels
            content = "\n".join([b.get("text", "") for b in content if b.get("type") == "text"])
        if role in {"system", "user", "assistant"} and content:
            out.append({"role": role, "content": content})

    # Ajoute la question de l'utilisateur si elle n'est pas déjà le dernier message
    if user_message and not (messages and (messages[-1].get("role") == "user")):
        out.append({"role": "user", "content": user_message})

    return out

def _parse_mistral_error(e: SDKError) -> Tuple[bool, Optional[str], Optional[float], str]:
    """Retourne (is_429, code, retry_after, message lisible)."""
    msg = str(e) or ""
    is_429 = "429" in msg or "Too Many Requests" in msg
    retry_after: Optional[float] = None
    code: Optional[str] = None
    body_msg = msg

    resp = getattr(e, "response", None)
    if resp is not None:
        # Retry-After
        try:
            hdrs = getattr(resp, "headers", None)
            if hdrs:
                ra = hdrs.get("Retry-After") or hdrs.get("retry-after")
                if ra:
                    try:
                        retry_after = float(ra)
                    except Exception:
                        pass
        except Exception:
            pass

        # Corps JSON
        try:
            j = resp.json()
            code = j.get("code") or (j.get("error") or {}).get("code") or j.get("type")
            body_msg = j.get("message") or (j.get("error") or {}).get("message") or msg
        except Exception:
            pass

    return is_429, code, retry_after, body_msg


# ───────────────────────── Pipeline ───────────────────────── #

class Pipeline:
    id = pipeline_id

    def __init__(self):
        self.name = title
        self.version = version
        self.author = author
        self.client: Optional[Mistral] = None

    async def on_startup(self):
        if not API_KEY:
            raise RuntimeError("MISTRAL_API_KEY manquant")
        self.client = Mistral(api_key=API_KEY)
        log.info("✅ Mistral brut (stream) prêt (model=%s; fallbacks=%s)", MODEL, ",".join(FALLBACKS))

    async def on_shutdown(self):
        pass

    # ── Streaming d’un seul appel
    def _stream_once(self, msgs: List[Dict[str, str]], model: str):
        """
        Ouvre un stream Mistral et yield les morceaux de texte au fil de l'eau.
        Compatible avec:
        1) chunk.data.choices[0].delta.content (forme actuelle)
        2) événements .type == 'message.delta' / 'content.delta' (forme legacy)
        Si aucun delta n'est reçu, renvoie la réponse finale (fallback).
        """
        assert self.client is not None

        yielded_any = False

        with self.client.chat.stream(
            model=model,
            messages=msgs,
            temperature=TEMP,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        ) as stream:
            for chunk in stream:
                # Debug éventuel
                if DEBUG_EVENTS:
                    try:
                        et = getattr(chunk, "type", None)
                        log.info("stream event type=%s", et or "data")
                    except Exception:
                        pass

                # ----- Forme "moderne" (docs Mistral) -----
                data = getattr(chunk, "data", None)
                if data is not None:
                    try:
                        choice = data.choices[0]
                        delta = choice.delta
                        content = getattr(delta, "content", None)

                        if isinstance(content, str):
                            if content:
                                yielded_any = True
                                yield content
                        elif isinstance(content, list):
                            for part in content:
                                ptype = getattr(part, "type", None) or (isinstance(part, dict) and part.get("type"))
                                if ptype == "text":
                                    text = getattr(part, "text", None) or (isinstance(part, dict) and part.get("text"))
                                    if text:
                                        yielded_any = True
                                        yield text
                    except Exception:
                        pass
                    continue  # chunk suivant

                # ----- Forme "legacy" (événements typés) -----
                etype = getattr(chunk, "type", None)
                if etype in ("message.delta", "content.delta"):
                    delta = getattr(chunk, "delta", None)
                    content = getattr(delta, "content", None) if delta else None

                    if isinstance(content, str):
                        if content:
                            yielded_any = True
                            yield content
                    elif isinstance(content, list):
                        for part in content:
                            ptype = getattr(part, "type", None) or (isinstance(part, dict) and part.get("type"))
                            if ptype == "text":
                                text = getattr(part, "text", None) or (isinstance(part, dict) and part.get("text"))
                                if text:
                                    yielded_any = True
                                    yield text

            # ---- Fallback : pas de deltas reçus → renvoyer la réponse finale ----
            try:
                final = stream.get_final_response()
                if not yielded_any and final and getattr(final, "choices", None):
                    msg_content = final.choices[0].message.content
                    if isinstance(msg_content, str):
                        if msg_content:
                            yield msg_content
                    elif isinstance(msg_content, list):
                        buf = []
                        for part in msg_content:
                            ptype = getattr(part, "type", None) or (isinstance(part, dict) and part.get("type"))
                            if ptype == "text":
                                text = getattr(part, "text", None) or (isinstance(part, dict) and part.get("text"))
                                if text:
                                    buf.append(text)
                        if buf:
                            yield "".join(buf)
            except Exception:
                # pas de final response dispo → on ignore
                pass

    # ── Streaming avec retry + fallback modèle
    def _stream_with_retry_and_fallback(self, msgs: List[Dict[str, str]]):
        # Construit la chaîne de modèles (dedup, ordre conservé)
        chain: List[str] = []
        seen = set()
        for m in [MODEL] + FALLBACKS:
            if m and m not in seen:
                seen.add(m)
                chain.append(m)

        last_err: Optional[Exception] = None

        for model in chain:
            # Cooldown de capacité (service_tier_capacity_exceeded)
            now = time.time()
            until = _MODEL_COOLDOWN_UNTIL.get(model, 0)
            if now < until:
                log.info("↷ Skip %s (capacity cooldown encore %.0fs)", model, until - now)
                continue

            log.info("→ Stream Mistral model=%s", model)
            attempt = 0

            while attempt < CHAT_MAX_RETRIES:
                try:
                    # Stream et ré-émission des chunks vers OpenWebUI
                    for chunk in self._stream_once(msgs, model):
                        yield chunk
                    return  # succès → on sort dès que le stream est fini
                except SDKError as e:
                    is_429, code, retry_after, body_msg = _parse_mistral_error(e)

                    # 429 “tier capacity exceeded” → tenter directement le modèle suivant
                    if is_429 and (code == "3505" or code == "service_tier_capacity_exceeded"):
                        log.warning("Modèle %s saturé pour ton tier (code=%s) → fallback.", model, code)
                        _MODEL_COOLDOWN_UNTIL[model] = time.time() + CAPACITY_COOLDOWN_S
                        last_err = e
                        break  # on passe au prochain modèle dans la chaîne

                    # Autre 429 (rate limit) → backoff et retry sur le même modèle
                    if is_429:
                        _sleep_with_backoff(attempt, CHAT_BACKOFF, retry_after)
                        attempt += 1
                        last_err = e
                        continue

                    # Erreur non récupérable → on remonte l'erreur
                    last_err = e
                    log.error("Erreur Mistral non récupérable (model=%s): %s", model, body_msg)
                    raise

        # Aucun modèle n'a abouti
        if last_err:
            raise last_err
        raise RuntimeError("Échec inconnu sans erreur retournée")

    # ── Entrée OpenWebUI
    def pipe(self, user_message: str, model_id: str, messages, body):
        """
        Passthrough Mistral : pas de RAG.
        Stream token par token avec retry + fallback modèle.
        """
        msgs = _to_chat_messages(user_message, messages)
        try:
            for token in self._stream_with_retry_and_fallback(msgs):
                yield token
        except Exception as e:
            yield f"Erreur Mistral: {e}"
