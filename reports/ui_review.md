# UI review — `streamlit_app_auto.py`

> plans.md item 5. Review of the Streamlit UI as committed in `05f6758`
> (the as-provided baseline). Line numbers refer to that version. Each issue
> lists the fix applied on branch RAG-2. Date: 2026-06-12.

## Verdict

As committed, the app **cannot start and cannot answer a single question** — initialization
always fails (issues 1–3), and even if it succeeded, the first search would crash (issue 4)
and the results display would crash (issue 5). Beyond the crashes, it rebuilds the entire
corpus on every cold start instead of loading the persisted index (issue 7) and defaults to a
model that is unusable on this CPU-only machine (issue 8).

## Blocking bugs (app cannot run)

1. **`streamlit_app_auto.py:67` — calls a method that doesn't exist.**
   `doc_processor.process_urls(urls)` — `DocumentProcessor` has no `process_urls`
   (see `src/document_ingestion/document_processor.py`: only `load_documents`,
   `split_documents`, `process`, all PDF-based). Always raises `AttributeError`, caught by the
   bare `except` → "Failed to intialized" → dead app.
   **Fix:** load the persisted Chroma index instead of processing documents at all.

2. **`streamlit_app_auto.py:71` — second nonexistent method.**
   `vector_store.create_retriever(documents)` — `VectorStore` has no `create_retriever`
   (it has `create_vectorstore` / `load_vectorstore` / `get_retriever`).
   **Fix:** `load_vectorstore()` + `get_retriever(k=4)`.

3. **`streamlit_app_auto.py:63` — empty corpus even if the methods existed.**
   `urls = Config.DEFAULT_URLS` is `[]` (`src/config/config.py:20`), so the app would index
   zero documents. The PIRLS corpus is PDFs, not URLs — this code was evidently copied from a
   different (web-corpus) project.
   **Fix:** same as 1 — use the persisted index built by `scripts/build_index.py`.

4. **`streamlit_app_auto.py:41-42` — session state deleted on every rerun.**
   ```python
   if "rag_system" in st.session_state:
       del st.session_state["rag_system"]
   ```
   Streamlit reruns the whole script on every interaction. Run 1 stores the system
   (line 99); run 2 (the user's first search) deletes it here, then line 114 reads
   `st.session_state.rag_system` → `AttributeError`. Every search would crash.
   **Fix:** delete these two lines; keep system state inside `@st.cache_resource` only.

5. **`streamlit_app_auto.py:136` — wrong result key.**
   `result['retrieved_docs']` — the advanced/CRAG graph state has no `retrieved_docs`; its
   key is `documents` (standard uses `retrieved_docs`, CRAG++ uses `final_contexts`).
   `KeyError` when rendering sources.
   **Fix:** per-architecture result extraction (same mapping as `scripts/run_generation.py`).

## Functional bugs (wrong behavior once it runs)

6. **`streamlit_app_auto.py:35-48` — `init_session_state()` is a no-op.**
   The function body is just a docstring; the four `if` blocks below it are dedented to module
   level. Works by accident (module code runs every rerun) but the call on line 88 does
   nothing, and the accidental module-level execution is what makes issue 4 fire every rerun.
   **Fix:** real function body (minus the deletion bug), called once.

7. **`streamlit_app_auto.py:63-71` — rebuilds the corpus on every cold start.**
   Re-processing and re-embedding the corpus takes ~15 min on CPU and recreates an index that
   already exists on disk. The persisted `chroma_db/` (6,771 chunks, mpnet) loads in seconds.
   **Fix:** `VectorStore.load_vectorstore()`; refuse to start with a clear message if the
   index is missing (pointing at `scripts/build_index.py`).

8. **`streamlit_app_auto.py:55` + `src/config/config.py:24` — unusable default model.**
   `Config.get_llm()` defaults to `deepseek-r1:8b` via local Ollama — slow on CPU and its
   `<think>` blocks pollute the UI unless stripped. The committed UI never strips them.
   **Fix:** model selector in the sidebar (default `llama3:8b`, measured ~18 s/answer on this
   machine) + `strip_reasoning` on display.

9. **Filename vs. behavior mismatch.** The file is `streamlit_app_auto.py` but it imports
   `graph_builder_adv` (CRAG) — the Auto architecture is not involved at all.
   **Fix:** architecture selector (Standard / CRAG / CRAG++); the file name is kept to avoid
   breaking any user bookmarks, with the selector making the actual architecture explicit.

## Usability gaps (improvements applied)

10. **No retrieved-context inspection** (plans.md item 6 asks for exactly this): sources were
    plain text dumps with no relevance signal.
    **Fix:** each retrieved chunk is shown with its **similarity score** to the question
    (`similarity_search_with_score`), source file and page, so retrieved text can be compared
    against the prompt directly.
11. **No latency/system feedback** beyond a single elapsed-time caption; no indication of
    which model/architecture produced the answer. **Fix:** answer card shows architecture,
    model, latency, and chunk count.
12. **History held forever** in session state (memory growth) and re-rendered in full.
    **Fix:** capped at the last 10 entries.
13. Minor: typos ("Intialized", "Simple CS"), `page_icon="RAG"` (not a valid icon),
    `st.text_area(disabled=True)` for read-only text where `st.markdown`/`st.code` is more
    appropriate. Fixed in passing.

## Notebook usability (the de-facto interface until now)

Not fixed in code (the notebooks are kept as the historical record of the old runs), but noted
for completeness: 16 near-duplicate notebooks differ only in model/system constants and
hard-coded `/content/drive/MyDrive/RAG/...` paths; setup cells must be run in a specific
order; results files are named by hand (`_2`, `(1)`, `.csv.csv` artifacts visible in
`results/`). The parameterized scripts in `scripts/` replace this workflow for the r3 runs.
