"""Smoke test: the repaired Streamlit UI answers end-to-end with all 3 architectures.

Standard runs through the full widget layer (streamlit.testing.v1.AppTest):
type a question, submit the form, assert the answer card and the
chunk/distance inspector rendered. CRAG and CRAG++ exercise the same app
functions directly (get_graph + extract) to keep CPU runtime reasonable.

Run:  .venv\\Scripts\\python tests\\smoke_ui.py   (needs chroma_db + local Ollama)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTION = ("Which month is set aside each year in the UAE for nationwide "
            "reading and literacy activities?")


def test_standard_through_widgets():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "streamlit_app_auto.py"), default_timeout=600)
    at.run()
    assert not at.exception, f"app failed to start: {at.exception}"

    at.text_input[0].set_value(QUESTION)
    at.button[0].set_value(True).run()
    assert not at.exception, f"search crashed: {at.exception}"

    assert at.success, "no answer card rendered"
    answer = at.success[0].value
    assert answer.strip(), "empty answer"
    assert "march" in answer.lower(), f"unexpected answer: {answer[:200]}"

    markdown_text = " ".join(str(m.value) for m in at.markdown)
    assert "distance" in markdown_text.lower(), "chunk/distance inspector not rendered"
    print(f"PASS Standard via widgets: answer={answer[:60]!r}, inspector rendered")


def test_graph_functions(architecture):
    import streamlit_app_auto as app

    graph = app.get_graph(architecture, "llama3:8b")
    result = graph.run(QUESTION)
    answer, docs = app.extract(architecture, result)
    assert answer.strip(), f"{architecture}: empty answer"
    assert docs, f"{architecture}: no context documents"
    print(f"PASS {architecture}: answer={answer[:60]!r}, {len(docs)} chunks")


def main():
    test_standard_through_widgets()
    test_graph_functions("CRAG")
    test_graph_functions("CRAG++")
    print("ALL STAGE-5 UI SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
