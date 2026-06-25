
# # # IEA • PIRLS RAG — Streamlit UI (ChatGPT-style with inline sources + persistent chats)
# # from pathlib import Path
# # import sys, time, uuid, re, datetime as dt
# # import streamlit as st

# # # ---------------------------------------------------------------------
# # # Import paths
# # # ---------------------------------------------------------------------
# # ROOT = Path(__file__).parent.resolve()
# # sys.path.extend([str(ROOT), str(ROOT / "src")])

# # try:
# #     from config import Config
# # except Exception:
# #     from src.config.config import Config
# # try:
# #     from document_processor import DocumentProcessor
# # except Exception:
# #     from src.document_ingestion.document_processor import DocumentProcessor
# # try:
# #     from vectorstore import VectorStore
# # except Exception:
# #     from src.vectorstore.vectorstore import VectorStore
# # try:
# #     from graph_builder import GraphBuilder
# # except Exception:
# #     from src.graph_builder.graph_builder import GraphBuilder

# # # NEW: disk persistence of chat sessions
# # from src.state.chat_store import get_user_id, load_sessions, save_sessions

# # # ---------------------------------------------------------------------
# # # Page config
# # # ---------------------------------------------------------------------
# # st.set_page_config(page_title="IEA • PIRLS Document Search", page_icon="📘", layout="wide")

# # # ---------------------------------------------------------------------
# # # Styles
# # # ---------------------------------------------------------------------
# # st.markdown("""
# # <style>
# # /* Layout paddings */
# # .block-container { padding-top: 80px; padding-bottom: 4.5rem; }

# # /* === Sticky header wrapper === */
# # #header-sticky { position: sticky; top: 0; z-index: 1000; background: #ffffff;
# #   margin: -12px 0 10px 0; padding: 10px 6px 6px 6px; border-bottom: 1px solid #e5e7eb;
# #   box-shadow: 0 4px 10px rgba(0,0,0,0.02); }

# # /* Top bar */
# # .topbar{ display:flex; align-items:center; gap:14px; padding:.4rem .25rem; margin:0; }
# # .topbar h3{ margin:0; font-size:1.1rem; font-weight:700; letter-spacing:.2px; }

# # /* Cards & meta */
# # .card{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:.8rem 1rem; }
# # .meta{ font-size:.78rem; color:#6b7280; margin:.25rem 0 .5rem; }

# # .pulse-dot{ display:inline-block; width:10px; height:10px; border-radius:50%;
# #   background:#22c55e; box-shadow:0 0 0 rgba(34,197,94,.7); animation:pulse 1.5s infinite; margin-right:6px; }
# # @keyframes pulse{ 0%{box-shadow:0 0 0 0 rgba(34,197,94,.7);}
# #   70%{box-shadow:0 0 0 10px rgba(34,197,94,0);} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0);} }

# # /* Footer */
# # .footer{ position:fixed; left:0; right:0; bottom:0; height:40px; background:#fafafa;
# #   border-top:1px solid #e5e7eb; display:flex; align-items:center; justify-content:center;
# #   font-size:.85rem; color:#6b7280; z-index:9999; }

# # /* Sidebar/session controls */
# # .session-row{ display:flex; align-items:center; gap:8px; background:#f9fafb;
# #   border:1px solid #e5e7eb; border-radius:12px; padding:.44rem .6rem; margin-bottom:.5rem; }
# # [data-testid="stSidebar"] { min-width: 373px !important; }
# # button[id^="sel_"]{ background:transparent !important; border:none !important; color:#111827 !important;
# #   text-align:left !important; font-weight:600 !important; padding:.15rem 0 !important; margin:0 !important; }
# # button[id^="sel_"]:hover{ color:#0b0f19 !important; }

# # /* Avoid chat input overlap on long threads */
# # .conversation-pad-bottom{ padding-bottom:120px; }

# # /* === GLOBAL UI BLOCKER (freezes all interactions while busy) === */
# # .ui-blocker{ position: fixed; inset: 0; z-index: 10001; background: rgba(255,255,255,.55);
# #   backdrop-filter: blur(1px); cursor: wait; }
# # .ui-blocker .inner{ position:absolute; left:50%; top:40%; transform:translate(-50%,-50%);
# #   background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px;
# #   box-shadow: 0 6px 20px rgba(0,0,0,.06); font: 500 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#111827; }

# # /* === BLUE SUBMIT BUTTON === */
# # [data-testid="stChatInput"] button { background-color: #2563eb !important; color: white !important; border: none !important;
# #   font-weight: 600 !important; border-radius: 9999px !important; padding: 0.45rem 1.2rem !important;
# #   transition: background-color 0.25s ease-in-out, transform 0.15s ease-in-out; }
# # [data-testid="stChatInput"] button:hover { background-color: #1d4ed8 !important; transform: translateY(-1px);
# #   box-shadow: 0 3px 8px rgba(29, 78, 216, 0.25); }
# # [data-testid="stChatInput"] button:active { background-color: #1e40af !important; transform: translateY(0);
# #   box-shadow: 0 1px 4px rgba(29, 78, 216, 0.3); }
# # </style>
# # """, unsafe_allow_html=True)

# # # ---------------------------------------------------------------------
# # # Utility: URL query param helpers (Streamlit >=1.29 has st.query_params)
# # # ---------------------------------------------------------------------
# # def _get_query_uid() -> str | None:
# #     try:
# #         # modern API
# #         qp = dict(st.query_params)
# #         return qp.get("uid")
# #     except Exception:
# #         # fallback to experimental
# #         params = st.experimental_get_query_params()
# #         return params.get("uid", [None])[0] if params else None

# # def _set_query_uid(uid: str):
# #     try:
# #         st.query_params["uid"] = uid
# #     except Exception:
# #         st.experimental_set_query_params(uid=uid)

# # # ---------------------------------------------------------------------
# # # State
# # # ---------------------------------------------------------------------
# # def _new_sid(): return uuid.uuid4().hex[:8]

# # def init_state():
# #     if "rag" not in st.session_state: st.session_state.rag = None
# #     if "ready" not in st.session_state: st.session_state.ready = False
# #     if "sessions" not in st.session_state:
# #         sid = _new_sid()
# #         st.session_state.sessions = {sid: {"title": "New chat", "turns": []}}
# #         st.session_state.active = sid
# #     if "busy" not in st.session_state: st.session_state.busy = False
# #     if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = ""
# #     if "open_sources" not in st.session_state: st.session_state.open_sources = set()
# #     if "editing_sid" not in st.session_state: st.session_state.editing_sid = None
# #     if "editing_value" not in st.session_state: st.session_state.editing_value = ""
# #     if "uid" not in st.session_state:
# #         st.session_state.uid = get_user_id(_get_query_uid())
# #         _set_query_uid(st.session_state.uid)
# #     if "loaded_from_disk" not in st.session_state:
# #         # Load previous sessions for this user
# #         persisted = load_sessions(st.session_state.uid)
# #         if persisted:
# #             st.session_state.sessions = persisted
# #             # ensure active exists
# #             if not st.session_state.sessions:
# #                 sid = _new_sid()
# #                 st.session_state.sessions = {sid: {"title": "New chat", "turns": []}}
# #             if "active" not in st.session_state or st.session_state.active not in st.session_state.sessions:
# #                 st.session_state.active = next(iter(st.session_state.sessions))
# #         st.session_state.loaded_from_disk = True

# # init_state()

# # def _persist():
# #     """Write sessions to disk for this user."""
# #     save_sessions(st.session_state.uid, st.session_state.sessions)

# # # ---------------------------------------------------------------------
# # # Boot (RAG pipeline)
# # # ---------------------------------------------------------------------
# # @st.cache_resource
# # def _boot():
# #     llm = Config.get_llm()
# #     dp = DocumentProcessor(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
# #     vs = VectorStore()
# #     docs = dp.process_pdf(["data"])
# #     vs.create_retriever(docs)
# #     gb = GraphBuilder(vs.get_retriever(), llm)
# #     gb.build()
# #     return gb, len(docs)

# # def refresh_index():
# #     try:
# #         st.cache_resource.clear()
# #     except:
# #         pass
# #     st.session_state.rag, _ = _boot()

# # if not st.session_state.ready:
# #     with st.spinner("Starting… indexing documents in /data"):
# #         st.session_state.rag, doc_count = _boot()
# #         st.session_state.ready = True
# #         st.success(f"System ready! ({doc_count} documents indexed)")

# # # ---------------------------------------------------------------------
# # # Sticky Header
# # # ---------------------------------------------------------------------
# # st.markdown('<div id="header-sticky">', unsafe_allow_html=True)
# # cols_top = st.columns([2,10,3], vertical_alignment="center")
# # with cols_top[0]:
# #     if (ROOT / "img" / "logo.png").exists():
# #         st.image(str(ROOT / "img" / "logo.png"), width=88)
# #     else:
# #         st.markdown(
# #             '<img alt="IEA Hamburg" src="https://www.iea.nl/sites/default/files/2020-05/IEA_Hamburg_logo_rgb.png" width="88"/>',
# #             unsafe_allow_html=True
# #         )
# # with cols_top[1]:
# #     st.markdown('<div class="topbar"><h3>IEA • PIRLS Document Search</h3></div>', unsafe_allow_html=True)
# # with cols_top[2]:
# #     if st.session_state.busy:
# #         st.markdown('<span class="pulse-dot"></span>Searching…', unsafe_allow_html=True)
# # st.markdown('</div>', unsafe_allow_html=True)

# # # ---------------------------------------------------------------------
# # # GLOBAL UI BLOCKER (render only when busy)
# # # ---------------------------------------------------------------------
# # if st.session_state.busy:
# #     st.markdown(
# #         '<div class="ui-blocker"><div class="inner">🔎 Searching… Please wait</div></div>',
# #         unsafe_allow_html=True
# #     )

# # # ---------------------------------------------------------------------
# # # Worker: execute pending prompt while UI is already frozen
# # # ---------------------------------------------------------------------
# # if st.session_state.busy and st.session_state.pending_prompt:
# #     try:
# #         with st.status("Searching…", expanded=True) as status:
# #             status.write("Retrieving relevant passages…")
# #             start = time.time()
# #             res = st.session_state.rag.run(st.session_state.pending_prompt)
# #             status.update(label="Generating answer…")
# #             elapsed = time.time() - start
# #     except Exception as e:
# #         with st.chat_message("assistant"):
# #             st.error(f"Error: {e}")
# #     else:
# #         turns_ref = st.session_state.sessions[st.session_state.active]["turns"]
# #         turns_ref.append({
# #             "q": st.session_state.pending_prompt,
# #             "a": res.get("answer", "No answer returned."),
# #             "time": elapsed,
# #             "retrieved_docs": res.get("retrieved_docs", []),
# #             "ts": dt.datetime.now().strftime("%H:%M")
# #         })
# #         _persist()
# #     finally:
# #         st.session_state.pending_prompt = ""
# #         st.session_state.busy = False
# #         st.rerun()

# # # ---------------------------------------------------------------------
# # # Sidebar
# # # ---------------------------------------------------------------------
# # with st.sidebar:
# #     if (ROOT / "img" / "logo.png").exists():
# #         st.image(str(ROOT / "img" / "logo.png"), use_container_width=True)
# #     else:
# #         st.markdown(
# #             """
# #             <div style="text-align:center; margin: 2px 0 10px;">
# #               <img src="https://www.iea.nl/sites/default/files/2020-05/IEA_Hamburg_logo_rgb.png" width="180" />
# #             </div>
# #             """,
# #             unsafe_allow_html=True
# #         )

# #     if st.button("＋ New chat", use_container_width=True, type="primary", disabled=st.session_state.busy):
# #         sid = _new_sid()
# #         st.session_state.sessions[sid] = {"title": "New chat", "turns": []}
# #         st.session_state.active = sid
# #         st.session_state.open_sources = set()
# #         _persist()
# #         st.rerun()

# #     st.markdown("---")

# #     # Session list (rename/delete)
# #     for sid, sess in list(st.session_state.sessions.items())[::-1]:
# #         c_title, c_btn1, c_btn2 = st.columns([8, 2, 2])
# #         with c_title:
# #             if st.session_state.editing_sid == sid:
# #                 st.session_state.editing_value = st.text_input(
# #                     "Rename",
# #                     value=st.session_state.editing_value or (sess["title"] or "Untitled"),
# #                     key=f"edit_in_{sid}_{int(time.time()*1000)}",
# #                     label_visibility="collapsed",
# #                     disabled=st.session_state.busy
# #                 )
# #             else:
# #                 if st.button(("🗂  " + (sess["title"] or "Untitled"))[:40],
# #                              key=f"sel_{sid}",
# #                              use_container_width=True,
# #                              disabled=st.session_state.busy):
# #                     st.session_state.active = sid
# #                     st.session_state.open_sources = set()
# #                     st.rerun()
# #         with c_btn1:
# #             if st.session_state.editing_sid == sid:
# #                 if st.button("✓", key=f"save_{sid}", help="Save", disabled=st.session_state.busy):
# #                     name = (st.session_state.editing_value or "").strip()
# #                     if name:
# #                         st.session_state.sessions[sid]["title"] = name
# #                     st.session_state.editing_sid = None
# #                     st.session_state.editing_value = ""
# #                     _persist()
# #                     st.rerun()
# #             else:
# #                 if st.button("✎", key=f"ren_{sid}", help="Rename", disabled=st.session_state.busy):
# #                     st.session_state.editing_sid = sid
# #                     st.session_state.editing_value = sess["title"]
# #                     st.rerun()
# #         with c_btn2:
# #             if st.session_state.editing_sid == sid:
# #                 if st.button("✕", key=f"cancel_{sid}", help="Cancel", disabled=st.session_state.busy):
# #                     st.session_state.editing_sid = None
# #                     st.session_state.editing_value = ""
# #                     st.rerun()
# #             else:
# #                 if st.button("🗑", key=f"del_{sid}", help="Delete", disabled=st.session_state.busy):
# #                     st.session_state.sessions.pop(sid, None)
# #                     if st.session_state.sessions:
# #                         st.session_state.active = next(iter(st.session_state.sessions))
# #                     else:
# #                         new_sid = _new_sid()
# #                         st.session_state.sessions = {new_sid: {"title": "New chat", "turns": []}}
# #                         st.session_state.active = new_sid
# #                     st.session_state.open_sources = set()
# #                     _persist()
# #                     st.rerun()

# #     st.markdown("---")
# #     st.markdown("#### Upload documents")
# #     uploaded = st.file_uploader(
# #         "PDF/TXT (drag or browse)",
# #         type=["pdf", "txt"],
# #         accept_multiple_files=True,
# #         disabled=st.session_state.busy,
# #         help="Files are indexed into the search. Large uploads may take a moment."
# #     )
# #     if uploaded:
# #         data_dir = ROOT / "data"
# #         data_dir.mkdir(parents=True, exist_ok=True)
# #         total_bytes = sum(len(f.getbuffer()) for f in uploaded)
# #         if total_bytes > 200 * 1024 * 1024:
# #             st.error("Upload limit exceeded (200MB).")
# #         else:
# #             for f in uploaded:
# #                 base = re.sub(r"[^A-Za-z0-9._-]", "_", f.name)
# #                 out = data_dir / f"{int(time.time()*1000)}_{base}"
# #                 with open(out, "wb") as w:
# #                     w.write(f.getbuffer())
# #             st.success("Files uploaded.")
# #             with st.spinner("Re-indexing…"):
# #                 refresh_index()
# #             st.info("Index refreshed. New documents are searchable.")

# # # ---------------------------------------------------------------------
# # # Main layout (chat history)
# # # ---------------------------------------------------------------------
# # st.markdown("##### Chat")
# # st.markdown('<div class="card" style="color:#2563eb;">Type below and press Enter — your message will appear here.</div>', unsafe_allow_html=True)

# # turns = st.session_state.sessions[st.session_state.active]["turns"]

# # for i, t in enumerate(turns):
# #     with st.chat_message("user"):
# #         st.write(t["q"])

# #     with st.chat_message("assistant"):
# #         st.write(t["a"])
# #         st.caption(f"Response time: {t['time']:.2f}s • {t['ts']}")

# #         label = "Show sources" if i not in st.session_state.open_sources else "Hide sources"
# #         if st.button(
# #             label,
# #             key=f"sources_{i}",
# #             disabled=st.session_state.busy,
# #             help="Disabled while searching…" if st.session_state.busy else None
# #         ):
# #             if i in st.session_state.open_sources:
# #                 st.session_state.open_sources.remove(i)
# #             else:
# #                 st.session_state.open_sources.add(i)
# #             st.rerun()

# #         if i in st.session_state.open_sources:
# #             docs = t.get("retrieved_docs") or []
# #             if not docs:
# #                 st.info("No sources returned.")
# #             else:
# #                 for j, d in enumerate(docs, 1):
# #                     meta = getattr(d, "metadata", {}) or {}
# #                     title = meta.get("title") or meta.get("source") or f"Document {j}"
# #                     page = meta.get("page")
# #                     header = f"{j}. {title}" + (f" — page {page}" if page else "")
# #                     with st.expander(header, expanded=(j == 1 and len(docs) <= 3)):
# #                         content = getattr(d, "page_content", "")
# #                         st.code((str(content) or "").strip()[:4000], language="markdown")
# #                         if meta:
# #                             st.caption(", ".join([f"{k}: {v}" for k, v in meta.items() if v]))

# # # ---------------------------------------------------------------------
# # # Input & Footer
# # # ---------------------------------------------------------------------
# # if not st.session_state.busy:
# #     prompt = st.chat_input("Ask anything…")
# # else:
# #     prompt = None
# #     st.info("Working on your last query…")

# # if prompt and not st.session_state.busy:
# #     st.session_state.pending_prompt = prompt.strip()
# #     st.session_state.busy = True
# #     st.rerun()

# # st.markdown('<div class="footer">© All rights reserved to IEA</div>', unsafe_allow_html=True)



# # IEA • PIRLS RAG — Streamlit UI (ChatGPT-style with inline sources + persistent chats)
# from pathlib import Path
# import sys, time, uuid, re, datetime as dt
# import streamlit as st

# # ---------------------------------------------------------------------
# # Import paths
# # ---------------------------------------------------------------------
# ROOT = Path(__file__).parent.resolve()
# sys.path.extend([str(ROOT), str(ROOT / "src")])

# try:
#     from config import Config
# except Exception:
#     from src.config.config import Config
# try:
#     from document_processor import DocumentProcessor
# except Exception:
#     from src.document_ingestion.document_processor import DocumentProcessor
# try:
#     from vectorstore import VectorStore
# except Exception:
#     from src.vectorstore.vectorstore import VectorStore
# try:
#     from graph_builder import GraphBuilder
# except Exception:
#     from src.graph_builder.graph_builder import GraphBuilder

# # NEW: disk persistence of chat sessions
# from src.state.chat_store import get_user_id, load_sessions, save_sessions

# # ---------------------------------------------------------------------
# # Page config
# # ---------------------------------------------------------------------
# st.set_page_config(page_title="IEA • PIRLS Document Search", page_icon="📘", layout="wide")

# # ---------------------------------------------------------------------
# # Styles
# # ---------------------------------------------------------------------
# st.markdown("""
# <style>
# /* Layout paddings */
# .block-container { padding-top: 80px; padding-bottom: 4.5rem; }

# /* === Sticky header wrapper === */
# #header-sticky { position: sticky; top: 0; z-index: 1000; background: #ffffff;
#   margin: -12px 0 10px 0; padding: 10px 6px 6px 6px; border-bottom: 1px solid #e5e7eb;
#   box-shadow: 0 4px 10px rgba(0,0,0,0.02); }

# /* Top bar */
# .topbar{ display:flex; align-items:center; gap:14px; padding:.4rem .25rem; margin:0; }
# .topbar h3{ margin:0; font-size:1.1rem; font-weight:700; letter-spacing:.2px; }

# /* Cards & meta */
# .card{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:.8rem 1rem; }
# .meta{ font-size:.78rem; color:#6b7280; margin:.25rem 0 .5rem; }

# .pulse-dot{ display:inline-block; width:10px; height:10px; border-radius:50%;
#   background:#22c55e; box-shadow:0 0 0 rgba(34,197,94,.7); animation:pulse 1.5s infinite; margin-right:6px; }
# @keyframes pulse{ 0%{box-shadow:0 0 0 0 rgba(34,197,94,.7);}
#   70%{box-shadow:0 0 0 10px rgba(34,197,94,0);} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0);} }

# /* Footer */
# .footer{ position:fixed; left:0; right:0; bottom:0; height:40px; background:#fafafa;
#   border-top:1px solid #e5e7eb; display:flex; align-items:center; justify-content:center;
#   font-size:.85rem; color:#6b7280; z-index:9999; }

# /* Sidebar/session controls */
# .session-row{ display:flex; align-items:center; gap:8px; background:#f9fafb;
#   border:1px solid #e5e7eb; border-radius:12px; padding:.44rem .6rem; margin-bottom:.5rem; }
# [data-testid="stSidebar"] { min-width: 373px !important; }
# button[id^="sel_"]{ background:transparent !important; border:none !important; color:#111827 !important;
#   text-align:left !important; font-weight:600 !important; padding:.15rem 0 !important; margin:0 !important; }
# button[id^="sel_"]:hover{ color:#0b0f19 !important; }

# /* Avoid chat input overlap on long threads */
# .conversation-pad-bottom{ padding-bottom:120px; }

# /* === GLOBAL UI BLOCKER (freezes all interactions while busy) === */
# .ui-blocker{ position: fixed; inset: 0; z-index: 10001; background: rgba(255,255,255,.55);
#   backdrop-filter: blur(1px); cursor: wait; }
# .ui-blocker .inner{ position:absolute; left:50%; top:40%; transform:translate(-50%,-50%);
#   background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px;
#   box-shadow: 0 6px 20px rgba(0,0,0,.06); font: 500 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#111827; }

# /* === BLUE SUBMIT BUTTON === */
# [data-testid="stChatInput"] button { background-color: #2563eb !important; color: white !important; border: none !important;
#   font-weight: 600 !important; border-radius: 9999px !important; padding: 0.45rem 1.2rem !important;
#   transition: background-color 0.25s ease-in-out, transform 0.15s ease-in-out; }
# [data-testid="stChatInput"] button:hover { background-color: #1d4ed8 !important; transform: translateY(-1px);
#   box-shadow: 0 3px 8px rgba(29, 78, 216, 0.25); }
# [data-testid="stChatInput"] button:active { background-color: #1e40af !important; transform: translateY(0);
#   box-shadow: 0 1px 4px rgba(29, 78, 216, 0.3); }
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------------------------------------------------
# # Utility: URL query param helpers (Streamlit >=1.29 has st.query_params)
# # ---------------------------------------------------------------------
# def _get_query_uid() -> str | None:
#     try:
#         qp = dict(st.query_params)
#         return qp.get("uid")
#     except Exception:
#         params = st.experimental_get_query_params()
#         return params.get("uid", [None])[0] if params else None

# def _set_query_uid(uid: str):
#     try:
#         st.query_params["uid"] = uid
#     except Exception:
#         st.experimental_set_query_params(uid=uid)

# # ---------------------------------------------------------------------
# # State
# # ---------------------------------------------------------------------
# def _new_sid(): return uuid.uuid4().hex[:8]

# def init_state():
#     if "rag" not in st.session_state: st.session_state.rag = None
#     if "ready" not in st.session_state: st.session_state.ready = False
#     if "sessions" not in st.session_state:
#         sid = _new_sid()
#         st.session_state.sessions = {sid: {"title": "New chat", "turns": []}}
#         st.session_state.active = sid
#     if "busy" not in st.session_state: st.session_state.busy = False
#     if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = ""
#     if "open_sources" not in st.session_state: st.session_state.open_sources = set()
#     if "editing_sid" not in st.session_state: st.session_state.editing_sid = None
#     if "editing_value" not in st.session_state: st.session_state.editing_value = ""
#     if "uid" not in st.session_state:
#         st.session_state.uid = get_user_id(_get_query_uid())
#         _set_query_uid(st.session_state.uid)
#     if "loaded_from_disk" not in st.session_state:
#         persisted = load_sessions(st.session_state.uid)
#         if persisted:
#             st.session_state.sessions = persisted
#             if not st.session_state.sessions:
#                 sid = _new_sid()
#                 st.session_state.sessions = {sid: {"title": "New chat", "turns": []}}
#             if "active" not in st.session_state or st.session_state.active not in st.session_state.sessions:
#                 st.session_state.active = next(iter(st.session_state.sessions))
#         st.session_state.loaded_from_disk = True

#     # NEW: flags to handle uploader freeze behavior
#     if "has_uploads" not in st.session_state: st.session_state.has_uploads = False
#     if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0

# init_state()

# def _persist():
#     """Write sessions to disk for this user."""
#     save_sessions(st.session_state.uid, st.session_state.sessions)

# # ---------------------------------------------------------------------
# # Boot (RAG pipeline)
# # ---------------------------------------------------------------------
# @st.cache_resource
# def _boot():
#     llm = Config.get_llm()
#     dp = DocumentProcessor(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
#     vs = VectorStore()
#     docs = dp.process_pdf(["data"])
#     vs.create_retriever(docs)
#     gb = GraphBuilder(vs.get_retriever(), llm)
#     gb.build()
#     return gb, len(docs)

# def refresh_index():
#     try:
#         st.cache_resource.clear()
#     except:
#         pass
#     st.session_state.rag, _ = _boot()

# if not st.session_state.ready:
#     with st.spinner("Starting… indexing documents in /data"):
#         st.session_state.rag, doc_count = _boot()
#         st.session_state.ready = True
#         st.success(f"System ready! ({doc_count} documents indexed)")

# # ---------------------------------------------------------------------
# # Sticky Header
# # ---------------------------------------------------------------------
# st.markdown('<div id="header-sticky">', unsafe_allow_html=True)
# cols_top = st.columns([2,10,3], vertical_alignment="center")
# with cols_top[0]:
#     if (ROOT / "img" / "logo.png").exists():
#         st.image(str(ROOT / "img" / "logo.png"), width=88)
#     else:
#         st.markdown(
#             '<img alt="IEA Hamburg" src="https://www.iea.nl/sites/default/files/2020-05/IEA_Hamburg_logo_rgb.png" width="88"/>',
#             unsafe_allow_html=True
#         )
# with cols_top[1]:
#     st.markdown('<div class="topbar"><h3>IEA • PIRLS Document Search</h3></div>', unsafe_allow_html=True)
# with cols_top[2]:
#     if st.session_state.busy:
#         st.markdown('<span class="pulse-dot"></span>Searching…', unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------------------------------------------------
# # GLOBAL UI BLOCKER (render only when busy)
# # ---------------------------------------------------------------------
# if st.session_state.busy:
#     st.markdown(
#         '<div class="ui-blocker"><div class="inner">🔎 Searching… Please wait</div></div>',
#         unsafe_allow_html=True
#     )

# # ---------------------------------------------------------------------
# # Worker: execute pending prompt while UI is already frozen
# # ---------------------------------------------------------------------
# if st.session_state.busy and st.session_state.pending_prompt:
#     try:
#         with st.status("Searching…", expanded=True) as status:
#             status.write("Retrieving relevant passages…")
#             start = time.time()
#             res = st.session_state.rag.run(st.session_state.pending_prompt)
#             status.update(label="Generating answer…")
#             elapsed = time.time() - start
#     except Exception as e:
#         with st.chat_message("assistant"):
#             st.error(f"Error: {e}")
#     else:
#         turns_ref = st.session_state.sessions[st.session_state.active]["turns"]
#         turns_ref.append({
#             "q": st.session_state.pending_prompt,
#             "a": res.get("answer", "No answer returned."),
#             "time": elapsed,
#             "retrieved_docs": res.get("retrieved_docs", []),
#             "ts": dt.datetime.now().strftime("%H:%M")
#         })
#         _persist()
#     finally:
#         st.session_state.pending_prompt = ""
#         st.session_state.busy = False
#         st.rerun()

# # ---------------------------------------------------------------------
# # Sidebar
# # ---------------------------------------------------------------------
# with st.sidebar:
#     if (ROOT / "img" / "logo.png").exists():
#         st.image(str(ROOT / "img" / "logo.png"), use_container_width=True)
#     else:
#         st.markdown(
#             """
#             <div style="text-align:center; margin: 2px 0 10px;">
#               <img src="https://www.iea.nl/sites/default/files/2020-05/IEA_Hamburg_logo_rgb.png" width="180" />
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#     if st.button("＋ New chat", use_container_width=True, type="primary",
#                  disabled=st.session_state.busy or st.session_state.has_uploads):
#         sid = _new_sid()
#         st.session_state.sessions[sid] = {"title": "New chat", "turns": []}
#         st.session_state.active = sid
#         st.session_state.open_sources = set()
#         _persist()
#         st.rerun()

#     st.markdown("---")

#     # Session list (rename/delete)
#     for sid, sess in list(st.session_state.sessions.items())[::-1]:
#         c_title, c_btn1, c_btn2 = st.columns([8, 2, 2])
#         with c_title:
#             if st.session_state.editing_sid == sid:
#                 st.session_state.editing_value = st.text_input(
#                     "Rename",
#                     value=st.session_state.editing_value or (sess["title"] or "Untitled"),
#                     key=f"edit_in_{sid}_{int(time.time()*1000)}",
#                     label_visibility="collapsed",
#                     disabled=st.session_state.busy or st.session_state.has_uploads
#                 )
#             else:
#                 if st.button(("🗂  " + (sess["title"] or "Untitled"))[:40],
#                              key=f"sel_{sid}",
#                              use_container_width=True,
#                              disabled=st.session_state.busy or st.session_state.has_uploads):
#                     st.session_state.active = sid
#                     st.session_state.open_sources = set()
#                     st.rerun()
#         with c_btn1:
#             if st.session_state.editing_sid == sid:
#                 if st.button("✓", key=f"save_{sid}", help="Save",
#                              disabled=st.session_state.busy or st.session_state.has_uploads):
#                     name = (st.session_state.editing_value or "").strip()
#                     if name:
#                         st.session_state.sessions[sid]["title"] = name
#                     st.session_state.editing_sid = None
#                     st.session_state.editing_value = ""
#                     _persist()
#                     st.rerun()
#             else:
#                 if st.button("✎", key=f"ren_{sid}", help="Rename",
#                              disabled=st.session_state.busy or st.session_state.has_uploads):
#                     st.session_state.editing_sid = sid
#                     st.session_state.editing_value = sess["title"]
#                     st.rerun()
#         with c_btn2:
#             if st.session_state.editing_sid == sid:
#                 if st.button("✕", key=f"cancel_{sid}", help="Cancel",
#                              disabled=st.session_state.busy or st.session_state.has_uploads):
#                     st.session_state.editing_sid = None
#                     st.session_state.editing_value = ""
#                     st.rerun()
#             else:
#                 if st.button("🗑", key=f"del_{sid}", help="Delete",
#                              disabled=st.session_state.busy or st.session_state.has_uploads):
#                     st.session_state.sessions.pop(sid, None)
#                     if st.session_state.sessions:
#                         st.session_state.active = next(iter(st.session_state.sessions))
#                     else:
#                         new_sid = _new_sid()
#                         st.session_state.sessions = {new_sid: {"title": "New chat", "turns": []}}
#                         st.session_state.active = new_sid
#                     st.session_state.open_sources = set()
#                     _persist()
#                     st.rerun()

    
#     # --- Upload documents section (final version) ---
#     st.markdown("---")
#     st.markdown("#### Upload documents")

#     upl_container = st.container()
#     with upl_container:
#         uploaded = st.file_uploader(
#             "PDF/TXT (drag or browse)",
#             type=["pdf", "txt"],
#             accept_multiple_files=True,
#             disabled=st.session_state.busy,
#             key=f"uploader_{st.session_state.uploader_key}",
#             help="Files here will block search until you index them."
#         )

#     # Track whether there are files present in uploader right now
#     st.session_state.has_uploads = bool(uploaded and len(uploaded) > 0)

#     # If uploads present, show warning + single 'Index' button
#     if st.session_state.has_uploads:
#         st.toast("Uploads detected. Please index them to continue searching.", icon="⚠️")
#         st.warning("📎 Files are present in the upload box. Index them to continue searching.")

#         # Only one column now — the indexing action
#         if st.button("📚 Index & add to search", use_container_width=True, type="primary",
#                     disabled=st.session_state.busy):
#             data_dir = ROOT / "data"
#             data_dir.mkdir(parents=True, exist_ok=True)
#             total_bytes = sum(len(f.getbuffer()) for f in uploaded)
#             if total_bytes > 200 * 1024 * 1024:
#                 st.error("Upload limit exceeded (200MB).")
#             else:
#                 for f in uploaded:
#                     base = re.sub(r"[^A-Za-z0-9._-]", "_", f.name)
#                     out = data_dir / f"{int(time.time()*1000)}_{base}"
#                     with open(out, "wb") as w:
#                         w.write(f.getbuffer())
#                 st.success("Files uploaded.")
#                 with st.spinner("Re-indexing…"):
#                     refresh_index()
#                 st.info("Index refreshed. New documents are searchable.")

 
#             # Clear uploader after processing
#             st.session_state.uploader_key += 1
#             st.session_state.has_uploads = False
#             st.rerun()


# # ---------------------------------------------------------------------
# # Main layout (chat history)
# # ---------------------------------------------------------------------
# st.markdown("##### Chat")
# st.markdown('<div class="card" style="color:#2563eb;">Type below and press Enter — your message will appear here.</div>', unsafe_allow_html=True)

# turns = st.session_state.sessions[st.session_state.active]["turns"]

# for i, t in enumerate(turns):
#     with st.chat_message("user"):
#         st.write(t["q"])

#     with st.chat_message("assistant"):
#         st.write(t["a"])
#         st.caption(f"Response time: {t['time']:.2f}s • {t['ts']}")

#         label = "Show sources" if i not in st.session_state.open_sources else "Hide sources"
#         if st.button(
#             label,
#             key=f"sources_{i}",
#             disabled=st.session_state.busy or st.session_state.has_uploads,
#             help="Disabled while searching…" if st.session_state.busy else None
#         ):
#             if i in st.session_state.open_sources:
#                 st.session_state.open_sources.remove(i)
#             else:
#                 st.session_state.open_sources.add(i)
#             st.rerun()

#         if i in st.session_state.open_sources:
#             docs = t.get("retrieved_docs") or []
#             if not docs:
#                 st.info("No sources returned.")
#             else:
#                 for j, d in enumerate(docs, 1):
#                     meta = getattr(d, "metadata", {}) or {}
#                     title = meta.get("title") or meta.get("source") or f"Document {j}"
#                     page = meta.get("page")
#                     header = f"{j}. {title}" + (f" — page {page}" if page else "")
#                     with st.expander(header, expanded=(j == 1 and len(docs) <= 3)):
#                         content = getattr(d, "page_content", "")
#                         st.code((str(content) or "").strip()[:4000], language="markdown")
#                         if meta:
#                             st.caption(", ".join([f"{k}: {v}" for k, v in meta.items() if v]))

# # ---------------------------------------------------------------------
# # Input & Footer
# # ---------------------------------------------------------------------
# if not st.session_state.busy and not st.session_state.has_uploads:
#     prompt = st.chat_input("Ask anything…", disabled=False)
# else:
#     prompt = None
#     if st.session_state.busy:
#         st.info("Working on your last query…")
#     elif st.session_state.has_uploads:
#         st.warning("Uploads are present. Clear the upload section to continue searching.")

# if prompt and not st.session_state.busy and not st.session_state.has_uploads:
#     st.session_state.pending_prompt = prompt.strip()
#     st.session_state.busy = True
#     st.rerun()

# st.markdown('<div class="footer">© All rights reserved to IEA</div>', unsafe_allow_html=True)



# IEA • PIRLS RAG — Streamlit UI (ChatGPT-style with inline sources + persistent chats)
from pathlib import Path
import sys, time, uuid, re, datetime as dt
import streamlit as st

# ---------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
sys.path.extend([str(ROOT), str(ROOT / "src")])

try:
    from config import Config
except Exception:
    from src.config.config import Config
try:
    from document_processor import DocumentProcessor
except Exception:
    from src.document_ingestion.document_processor import DocumentProcessor
try:
    from vectorstore import VectorStore
except Exception:
    from src.vectorstore.vectorstore import VectorStore
try:
    from graph_builder import GraphBuilder
except Exception:
    from src.graph_builder.graph_builder import GraphBuilder

# NEW: disk persistence of chat sessions
from src.state.chat_store import get_user_id, load_sessions, save_sessions

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="IEA • PIRLS Document Search", page_icon="📘", layout="wide")

# ---------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------
st.markdown("""
<style>
/* Layout paddings */
.block-container { padding-top: 80px; padding-bottom: 4.5rem; }

/* === Sticky header wrapper === */
#header-sticky { position: sticky; top: 0; z-index: 1000; background: #ffffff;
  margin: -12px 0 10px 0; padding: 10px 6px 6px 6px; border-bottom: 1px solid #e5e7eb;
  box-shadow: 0 4px 10px rgba(0,0,0,0.02); }

/* Top bar */
.topbar{ display:flex; align-items:center; gap:14px; padding:.4rem .25rem; margin:0; }
.topbar h3{ margin:0; font-size:1.1rem; font-weight:700; letter-spacing:.2px; }

/* Cards & meta */
.card{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:.8rem 1rem; }
.meta{ font-size:.78rem; color:#6b7280; margin:.25rem 0 .5rem; }

.pulse-dot{ display:inline-block; width:10px; height:10px; border-radius:50%;
  background:#22c55e; box-shadow:0 0 0 rgba(34,197,94,.7); animation:pulse 1.5s infinite; margin-right:6px; }
@keyframes pulse{ 0%{box-shadow:0 0 0 0 rgba(34,197,94,.7);}
  70%{box-shadow:0 0 0 10px rgba(34,197,94,0);} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0);} }

/* Footer */
.footer{ position:fixed; left:0; right:0; bottom:0; height:40px; background:#fafafa;
  border-top:1px solid #e5e7eb; display:flex; align-items:center; justify-content:center;
  font-size:.85rem; color:#6b7280; z-index:9999; }

/* Sidebar/session controls */
.session-row{ display:flex; align-items:center; gap:8px; background:#f9fafb;
  border:1px solid #e5e7eb; border-radius:12px; padding:.44rem .6rem; margin-bottom:.5rem; }
[data-testid="stSidebar"] { min-width: 373px !important; }
button[id^="sel_"]{ background:transparent !important; border:none !important; color:#111827 !important;
  text-align:left !important; font-weight:600 !important; padding:.15rem 0 !important; margin:0 !important; }
button[id^="sel_"]:hover{ color:#0b0f19 !important; }

/* Avoid chat input overlap on long threads */
.conversation-pad-bottom{ padding-bottom:120px; }

/* === GLOBAL UI BLOCKER (freezes all interactions while busy) === */
.ui-blocker{ position: fixed; inset: 0; z-index: 10001; background: rgba(255,255,255,.55);
  backdrop-filter: blur(1px); cursor: wait; }
.ui-blocker .inner{ position:absolute; left:50%; top:40%; transform:translate(-50%,-50%);
  background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px;
  box-shadow: 0 6px 20px rgba(0,0,0,.06); font: 500 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#111827; }

/* === BLUE SUBMIT BUTTON === */
[data-testid="stChatInput"] button { background-color: #2563eb !important; color: white !important; border: none !important;
  font-weight: 600 !important; border-radius: 9999px !important; padding: 0.45rem 1.2rem !important;
  transition: background-color 0.25s ease-in-out, transform 0.15s ease-in-out; }
[data-testid="stChatInput"] button:hover { background-color: #1d4ed8 !important; transform: translateY(-1px);
  box-shadow: 0 3px 8px rgba(29, 78, 216, 0.25); }
[data-testid="stChatInput"] button:active { background-color: #1e40af !important; transform: translateY(0);
  box-shadow: 0 1px 4px rgba(29, 78, 216, 0.3); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Utility: URL query param helpers (Streamlit >=1.29 has st.query_params)
# ---------------------------------------------------------------------
def _get_query_uid() -> str | None:
    try:
        qp = dict(st.query_params)
        return qp.get("uid")
    except Exception:
        params = st.experimental_get_query_params()
        return params.get("uid", [None])[0] if params else None

def _set_query_uid(uid: str):
    try:
        st.query_params["uid"] = uid
    except Exception:
        st.experimental_set_query_params(uid=uid)

# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------
def _new_sid(): return uuid.uuid4().hex[:8]

def init_state():
    if "rag" not in st.session_state: st.session_state.rag = None
    if "ready" not in st.session_state: st.session_state.ready = False
    if "sessions" not in st.session_state:
        sid = _new_sid()
        st.session_state.sessions = {sid: {"title": "New chat", "turns": []}}
        st.session_state.active = sid
    if "busy" not in st.session_state: st.session_state.busy = False
    if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = ""
    if "open_sources" not in st.session_state: st.session_state.open_sources = set()
    if "editing_sid" not in st.session_state: st.session_state.editing_sid = None
    if "editing_value" not in st.session_state: st.session_state.editing_value = ""
    if "uid" not in st.session_state:
        st.session_state.uid = get_user_id(_get_query_uid())
        _set_query_uid(st.session_state.uid)
    if "loaded_from_disk" not in st.session_state:
        persisted = load_sessions(st.session_state.uid)
        if persisted:
            st.session_state.sessions = persisted
            if not st.session_state.sessions:
                sid = _new_sid()
                st.session_state.sessions = {sid: {"title": "New chat", "turns": []}}
            if "active" not in st.session_state or st.session_state.active not in st.session_state.sessions:
                st.session_state.active = next(iter(st.session_state.sessions))
        st.session_state.loaded_from_disk = True

    # NEW: flags to handle uploader freeze and success message
    if "has_uploads" not in st.session_state: st.session_state.has_uploads = False
    if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0
    if "upload_success" not in st.session_state: st.session_state.upload_success = False  # ✅ init here

init_state()

def _persist():
    """Write sessions to disk for this user."""
    save_sessions(st.session_state.uid, st.session_state.sessions)

# ---------------------------------------------------------------------
# Boot (RAG pipeline)
# ---------------------------------------------------------------------
@st.cache_resource
def _boot():
    llm = Config.get_llm()
    dp = DocumentProcessor(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
    vs = VectorStore()
    docs = dp.process_pdf(["data"])
    vs.create_retriever(docs)
    gb = GraphBuilder(vs.get_retriever(), llm)
    gb.build()
    return gb, len(docs)

def refresh_index():
    try:
        st.cache_resource.clear()
    except:
        pass
    st.session_state.rag, _ = _boot()

if not st.session_state.ready:
    with st.spinner("Starting… indexing documents in /data"):
        st.session_state.rag, doc_count = _boot()
        st.session_state.ready = True
        st.success(f"System ready! ({doc_count} documents indexed)")

# ---------------------------------------------------------------------
# Sticky Header
# ---------------------------------------------------------------------
st.markdown('<div id="header-sticky">', unsafe_allow_html=True)
cols_top = st.columns([2,10,3], vertical_alignment="center")
with cols_top[0]:
    if (ROOT / "img" / "logo.png").exists():
        st.image(str(ROOT / "img" / "logo.png"), width=88)
    else:
        st.markdown(
            '<img alt="IEA Hamburg" src="https://www.iea.nl/sites/default/files/2020-05/IEA_Hamburg_logo_rgb.png" width="88"/>',
            unsafe_allow_html=True
        )
with cols_top[1]:
    st.markdown('<div class="topbar"><h3>IEA • PIRLS Document Search</h3></div>', unsafe_allow_html=True)
with cols_top[2]:
    if st.session_state.busy:
        st.markdown('<span class="pulse-dot"></span>Searching…', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# GLOBAL UI BLOCKER (render only when busy)
# ---------------------------------------------------------------------
if st.session_state.busy:
    st.markdown(
        '<div class="ui-blocker"><div class="inner">🔎 Searching… Please wait</div></div>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------
# Worker: execute pending prompt while UI is already frozen
# ---------------------------------------------------------------------
if st.session_state.busy and st.session_state.pending_prompt:
    try:
        with st.status("Searching…", expanded=True) as status:
            status.write("Retrieving relevant passages…")
            start = time.time()
            res = st.session_state.rag.run(st.session_state.pending_prompt)
            status.update(label="Generating answer…")
            elapsed = time.time() - start
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {e}")
    else:
        turns_ref = st.session_state.sessions[st.session_state.active]["turns"]
        turns_ref.append({
            "q": st.session_state.pending_prompt,
            "a": res.get("answer", "No answer returned."),
            "time": elapsed,
            "retrieved_docs": res.get("retrieved_docs", []),
            "ts": dt.datetime.now().strftime("%H:%M")
        })
        _persist()
    finally:
        st.session_state.pending_prompt = ""
        st.session_state.busy = False
        st.rerun()

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    if (ROOT / "img" / "logo.png").exists():
        st.image(str(ROOT / "img" / "logo.png"), use_container_width=True)
    else:
        st.markdown(
            """
            <div style="text-align:center; margin: 2px 0 10px;">
              <img src="https://www.iea.nl/sites/default/files/2020-05/IEA_Hamburg_logo_rgb.png" width="180" />
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.button("＋ New chat", use_container_width=True, type="primary",
                 disabled=st.session_state.busy or st.session_state.has_uploads):
        sid = _new_sid()
        st.session_state.sessions[sid] = {"title": "New chat", "turns": []}
        st.session_state.active = sid
        st.session_state.open_sources = set()
        _persist()
        st.rerun()

    st.markdown("---")
    # Session list (rename/delete)
    for sid, sess in list(st.session_state.sessions.items())[::-1]:
        c_title, c_btn1, c_btn2 = st.columns([8, 2, 2])
        with c_title:
            if st.session_state.editing_sid == sid:
                st.session_state.editing_value = st.text_input(
                    "Rename",
                    value=st.session_state.editing_value or (sess["title"] or "Untitled"),
                    key=f"edit_in_{sid}",
                    label_visibility="collapsed",
                    disabled=st.session_state.busy or st.session_state.has_uploads
                )
            else:
                if st.button(("🗂  " + (sess["title"] or "Untitled"))[:40],
                             key=f"sel_{sid}",
                             use_container_width=True,
                             disabled=st.session_state.busy or st.session_state.has_uploads):
                    st.session_state.active = sid
                    st.session_state.open_sources = set()
                    st.rerun()
        with c_btn1:
            if st.session_state.editing_sid == sid:
                if st.button("✓", key=f"save_{sid}", help="Save",
                             disabled=st.session_state.busy or st.session_state.has_uploads):
                    name = (st.session_state.editing_value or "").strip()
                    if name:
                        st.session_state.sessions[sid]["title"] = name
                    st.session_state.editing_sid = None
                    st.session_state.editing_value = ""
                    _persist()
                    st.rerun()
            else:
                if st.button("✎", key=f"ren_{sid}", help="Rename",
                             disabled=st.session_state.busy or st.session_state.has_uploads):
                    st.session_state.editing_sid = sid
                    st.session_state.editing_value = sess["title"]
                    st.rerun()
        with c_btn2:
            if st.session_state.editing_sid == sid:
                if st.button("✕", key=f"cancel_{sid}", help="Cancel",
                             disabled=st.session_state.busy or st.session_state.has_uploads):
                    st.session_state.editing_sid = None
                    st.session_state.editing_value = ""
                    st.rerun()
            else:
                if st.button("🗑", key=f"del_{sid}", help="Delete",
                             disabled=st.session_state.busy or st.session_state.has_uploads):
                    st.session_state.sessions.pop(sid, None)
                    if st.session_state.sessions:
                        st.session_state.active = next(iter(st.session_state.sessions))
                    else:
                        new_sid = _new_sid()
                        st.session_state.sessions = {new_sid: {"title": "New chat", "turns": []}}
                        st.session_state.active = new_sid
                    st.session_state.open_sources = set()
                    _persist()
                    st.rerun()

    st.markdown("---")
    st.markdown("#### Upload documents")

    # ---- Uploader with freeze behavior (no clear button) ----
    upl_container = st.container()
    with upl_container:
        uploaded = st.file_uploader(
            "PDF/TXT (drag or browse)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            disabled=st.session_state.busy,  # uploader allowed while not busy
            key=f"uploader_{st.session_state.uploader_key}",
            help="Files here will block search until you index them."
        )

    # ✅ Reset success message when new files appear
    if uploaded and len(uploaded) > 0:
        st.session_state.upload_success = False

    # Track whether there are files present in uploader right now
    st.session_state.has_uploads = bool(uploaded and len(uploaded) > 0)

    # If uploads present, show warning + single indexing action (no overlay)
    if st.session_state.has_uploads:
        st.toast("Uploads detected. Please index them to continue searching.", icon="⚠️")
        st.warning("📎 Files are present in the upload box. Index them to continue searching.")

        if st.button("📚 Index & add to search", use_container_width=True, type="primary",
                     disabled=st.session_state.busy):
            data_dir = ROOT / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            total_bytes = sum(len(f.getbuffer()) for f in uploaded)
            if total_bytes > 200 * 1024 * 1024:
                st.error("Upload limit exceeded (200MB).")
            else:
                for f in uploaded:
                    base = re.sub(r"[^A-Za-z0-9._-]", "_", f.name)
                    out = data_dir / f"{int(time.time()*1000)}_{base}"
                    with open(out, "wb") as w:
                        w.write(f.getbuffer())

                with st.spinner("Re-indexing…"):
                    refresh_index()

                # ✅ Set success flag AFTER re-indexing completes
                st.session_state.upload_success = True

            # Clear uploader after processing
            st.session_state.uploader_key += 1
            st.session_state.has_uploads = False
            st.rerun()

    # ✅ Display persistent success message below upload section
    if st.session_state.get("upload_success", False):
        st.success("✅ Files uploaded and indexed successfully! You can now search within them.")

    

# ---------------------------------------------------------------------
# Main layout (chat history)
# ---------------------------------------------------------------------
st.markdown("##### Chat")
st.markdown('<div class="card" style="color:#2563eb;">Type below and press Enter — your message will appear here.</div>', unsafe_allow_html=True)

turns = st.session_state.sessions[st.session_state.active]["turns"]

for i, t in enumerate(turns):
    with st.chat_message("user"):
        st.write(t["q"])

    with st.chat_message("assistant"):
        st.write(t["a"])
        st.caption(f"Response time: {t['time']:.2f}s • {t['ts']}")

        label = "Show sources" if i not in st.session_state.open_sources else "Hide sources"
        if st.button(
            label,
            key=f"sources_{i}",
            disabled=st.session_state.busy or st.session_state.has_uploads,
            help="Disabled while searching…" if st.session_state.busy else None
        ):
            if i in st.session_state.open_sources:
                st.session_state.open_sources.remove(i)
            else:
                st.session_state.open_sources.add(i)
            st.rerun()

        if i in st.session_state.open_sources:
            docs = t.get("retrieved_docs") or []
            if not docs:
                st.info("No sources returned.")
            else:
                for j, d in enumerate(docs, 1):
                    meta = getattr(d, "metadata", {}) or {}
                    title = meta.get("title") or meta.get("source") or f"Document {j}"
                    page = meta.get("page")
                    header = f"{j}. {title}" + (f" — page {page}" if page else "")
                    with st.expander(header, expanded=(j == 1 and len(docs) <= 3)):
                        content = getattr(d, "page_content", "")
                        st.code((str(content) or "").strip()[:4000], language="markdown")
                        if meta:
                            st.caption(", ".join([f"{k}: {v}" for k, v in meta.items() if v]))

# ---------------------------------------------------------------------
# Input & Footer
# ---------------------------------------------------------------------
if not st.session_state.busy and not st.session_state.has_uploads:
    prompt = st.chat_input("Ask anything…", disabled=False)
else:
    prompt = None
    if st.session_state.busy:
        st.info("Working on your last query…")
    elif st.session_state.has_uploads:
        st.warning("Uploads are present. Please index them to continue searching.")

if prompt and not st.session_state.busy and not st.session_state.has_uploads:
    st.session_state.pending_prompt = prompt.strip()
    st.session_state.busy = True
    st.rerun()

st.markdown('<div class="footer">© All rights reserved to IEA</div>', unsafe_allow_html=True)
