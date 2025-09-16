#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Alerts RSS -> résumés quotidiens
- Récupère les articles depuis un (ou plusieurs) flux RSS Google Alerts
- Extrait l'URL d'origine (pas le lien Google)
- Récupère le texte de l'article
- Produit un court résumé extractif (TextRank) en français
- Génère un rapport Markdown dans ./output/YYYY-MM-DD.md (+ latest.md)

Utilisation locale:
    FEEDS="https://...rss1...,https://...rss2..." python main.py
Variables d'environnement (toutes optionnelles):
    FEEDS: liste d'URLs RSS séparées par des virgules ou des sauts de ligne
    SENTENCES: nb de phrases par résumé (defaut: 4)
    MAX_PER_FEED: nb max d'articles par flux (defaut: 20)
    TIMEOUT: timeout réseau en secondes (defaut: 20)
    OUTPUT_DIR: dossier de sortie (defaut: ./output)
"""
import os, re, sys, time, json, logging, hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs, unquote

import feedparser
import trafilatura
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# --- bootstrap NLTK data (français) ---
try:
    import nltk
    for res in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
except Exception:
    pass
# --- fin bootstrap ---

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LANGUAGE = "french"

from bs4 import BeautifulSoup  # déjà dans requirements

def html_to_text(html: str) -> str:
    """Convertit un fragment HTML (RSS) en texte brut propre."""
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Enlève les scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Nettoyage d'espaces multiples
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def get_env_list(name: str):
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    parts = []
    for chunk in raw.replace("\\n", "\n").splitlines():
        parts += [p.strip() for p in chunk.split(",") if p.strip()]
    return parts

def extract_original_url(url: str) -> str:
    """Pour les liens Google Alerts: essaye d'extraire le vrai lien (paramètre url= ou q=)."""
    try:
        p = urlparse(url)
        qs = parse_qs(p.query)
        for key in ("url", "q"):
            if key in qs and qs[key]:
                return unquote(qs[key][0])
        # parfois c'est dans le fragment
        frag_qs = parse_qs(p.fragment)
        if "url" in frag_qs and frag_qs["url"]:
            return unquote(frag_qs["url"][0])
        return url
    except Exception:
        return url

def fetch_text(url: str, timeout: int = 20) -> str:
    """Télécharge et extrait le texte principal de l'article, compatible toutes versions."""
    downloaded = None

    # 1) Essai avec trafilatura (sans arg timeout pour compatibilité)
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception:
        downloaded = None

    # 2) Plan B: requests avec User-Agent "navigateur"
    if not downloaded:
        try:
            import requests
            headers = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/124.0 Safari/537.36"),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            }
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            downloaded = r.text
        except Exception:
            return ""

    # 3) Extraction du texte principal
    text = trafilatura.extract(
        downloaded,
        include_tables=False,
        include_formatting=False,
        include_comments=False,
        favor_recall=False,
        no_fallback=True,
        url=url,
        output_format="txt",
    )
    return text or ""



def summarize_text(text: str, sentences: int = 4) -> str:
    """Résumé extractif (TextRank) en français -> puces."""
    if not text:
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    try:
        sents = [str(s) for s in summarizer(parser.document, sentences)]
    except Exception:
        # repli très simple: premières phrases
        sents = [str(s) for s in parser.document.sentences[:sentences]]
    sents = [re.sub(r"\s+", " ", s).strip(" .") for s in sents if s.strip()]
    if not sents:
        return ""
    return "\n".join(f"- {s}." for s in sents)

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.","")
    except Exception:
        return ""

def hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def load_seen(path: str) -> set:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_seen(path: str, seen: set):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(seen)), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_history(path: str) -> list:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            return []
    return []


def save_history(path: str, entries: list):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def main():
    feeds = get_env_list("FEEDS")
    if not feeds:
        logging.error("Aucun flux RSS spécifié. Définissez la variable d'environnement FEEDS.")
        sys.exit(1)

    sentences = int(os.getenv("SENTENCES", "4"))
    max_per_feed = int(os.getenv("MAX_PER_FEED", "20"))
    timeout = int(os.getenv("TIMEOUT", "20"))
    out_dir = os.getenv("OUTPUT_DIR", "output")

    os.makedirs(out_dir, exist_ok=True)
    seen_path = os.path.join(out_dir, "seen.json")
    seen = load_seen(seen_path)

    items = []
    for feed_url in feeds:
        logging.info(f"Lecture du flux: {feed_url}")
        fp = feedparser.parse(feed_url)
        if fp.bozo and not fp.entries:
            logging.warning(f"Flux invalide ou inaccessible: {feed_url}")
            continue

        for entry in fp.entries[:max_per_feed]:
            title = entry.get("title", "").strip()

            # Lien
            link = entry.get("link", "").strip()
            if not link:
                continue
            orig = extract_original_url(link)

            # ----- RÉCUPÉRATION D'UN TEXTE "hint" DEPUIS LE RSS -----
            hint_html = ""
            # 1) summary
            if entry.get("summary"):
                hint_html = entry.get("summary")
            # 2) summary_detail.value
            elif entry.get("summary_detail") and entry["summary_detail"].get("value"):
                hint_html = entry["summary_detail"]["value"]
            # 3) content[0].value (très courant sur Google Alerts)
            elif entry.get("content") and isinstance(entry["content"], list) and entry["content"]:
                first = entry["content"][0]
                if isinstance(first, dict) and first.get("value"):
                    hint_html = first["value"]

            hint_text = html_to_text(hint_html)

            uid = hash_id(orig or link)
            if uid in seen:
                continue

            items.append({
                "title": title or "(Sans titre)",
                "link": orig or link,
                "source": domain_of(orig or link),
                "uid": uid,
                "hint": hint_text,  # <= on garde le texte RSS nettoyé
                "published": entry.get("published") or entry.get("updated") or "",
            })


        logging.info(f"{len(items)} nouvel(le)s article(s) à traiter.")
    results = []
    for it in items:
        url = it["link"]
        title = it["title"]
        hint = it.get("hint", "")  # texte RSS (fallback)

        try:
            full = fetch_text(url, timeout=timeout)
            base_text = full or hint or title  # <= priorité: article, sinon RSS, sinon titre

            summary = summarize_text(base_text, sentences=sentences) if base_text else ""
            if not summary:
                summary = "- (Résumé indisponible – texte non détecté)."

            results.append({**it, "summary": summary})
            seen.add(it["uid"])
            logging.info(f"OK: {title} [{it['source']}]")
        except Exception as e:
            logging.warning(f"Echec: {title} ({url}) -> {e}")

    # écriture des sorties
    today = datetime.now(timezone.utc).astimezone().date().isoformat()
    md_path = os.path.join(out_dir, f"{today}.md")
    latest_path = os.path.join(out_dir, "latest.md")
    history_json_path = os.path.join(out_dir, "all_articles.json")
    history_md_path = os.path.join(out_dir, "all_articles.md")

    history_entries = load_history(history_json_path)
    known_ids = {entry.get("uid") for entry in history_entries}
    updated_history = False
    for res in results:
        uid = res["uid"]
        if uid not in known_ids:
            history_entries.append({
                "uid": uid,
                "title": res["title"],
                "link": res["link"],
                "source": res.get("source", ""),
                "published": res.get("published", ""),
                "summary": res.get("summary", ""),
            })
            known_ids.add(uid)
            updated_history = True

    if updated_history or not os.path.exists(history_json_path):
        save_history(history_json_path, history_entries)

    def make_md(res, title):
        header = f"# {title}\n\n"
        if not res:
            return header + "_Aucun nouvel article._\n"
        parts = [header]
        for r in res:
            title = r.get('title', '(Sans titre)')
            link = r.get('link') or ''
            source = r.get('source') or ''
            published = r.get('published') or ''
            summary = (r.get('summary') or '').strip() or "- (Résumé indisponible)."
            meta_parts = []
            if source:
                meta_parts.append(f"Source: {source}")
            if published:
                meta_parts.append(f"Publié: {published}")
            meta = f"*{' | '.join(meta_parts)}*" if meta_parts else ""
            entry = f"## [{title}]({link})" if link else f"## {title}"
            if meta:
                entry += f"  \n{meta}"
            entry += f"\n\n{summary}\n"
            parts.append(entry)
        return "\n".join(parts)

    daily_md = make_md(results, f"Résumé Google Alerts – {today}")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(daily_md)
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(daily_md)

    history_for_md = list(reversed(history_entries))
    history_md = make_md(history_for_md, "Historique complet Google Alerts")
    with open(history_md_path, "w", encoding="utf-8") as f:
        f.write(history_md)

    save_seen(seen_path, seen)

    print(f"Créé: {md_path}\nAussi: {latest_path}\nArticles: {len(results)}")

if __name__ == "__main__":
    main()
