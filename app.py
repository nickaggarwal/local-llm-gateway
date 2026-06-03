"""Streamlit UI for the local LLM gateway.

Run: streamlit run app.py
"""

import os
import tempfile

import streamlit as st

import gateway
from registry import TASKS

st.set_page_config(page_title="Local LLM Gateway", page_icon="🧠", layout="centered")

st.title("🧠 Local LLM Gateway")
st.caption("Pick a task — the best local model is chosen, downloaded if needed, and run on your laptop.")

ram = gateway.available_ram_gb()
st.sidebar.metric("Detected RAM", f"{ram:.0f} GB")
st.sidebar.write("Models run locally via Ollama. Nothing leaves your machine.")

task_name = st.selectbox(
    "Task",
    options=list(TASKS),
    format_func=lambda n: f"{n} — {TASKS[n].description}",
)
task = TASKS[task_name]
selected_model = task.pick_model(ram)
st.info(f"**Kind:** {task.kind}  •  **Model:** `{selected_model}`")

override = st.sidebar.text_input("Force model (optional)", placeholder=selected_model)

# Reset chat history when the task changes.
if st.session_state.get("active_task") != task_name:
    st.session_state["active_task"] = task_name
    st.session_state["history"] = []
if st.sidebar.button("Clear chat"):
    st.session_state["history"] = []


def run_and_render(prompt: str, image_path: str | None) -> None:
    """Run the gateway for one turn and append the result to chat history."""
    status = st.status("Working...", expanded=True)

    def on_progress(msg: str) -> None:
        status.write(msg if len(msg) < 200 else msg[:200])

    try:
        with status:
            result = gateway.run(
                task_name, prompt=prompt, image_path=image_path,
                model=(override or None), on_progress=on_progress,
            )
        status.update(label=f"Done · {result['model']}", state="complete", expanded=False)
        if "embedding" in result:
            emb = result["embedding"]
            answer = f"{len(emb)}-dimensional embedding\n\n```python\n{emb[:12]} ...\n```"
        else:
            answer = result["text"]
    except Exception as e:  # noqa: BLE001
        status.update(label="Error", state="error")
        answer = f"⚠️ {e}"

    st.session_state["history"].append({"role": "assistant", "content": answer})


# --- Embed task: simple one-shot input (chat doesn't apply) ---
if task.kind == "embed":
    text = st.text_area("Text to embed", height=140)
    if st.button("Run", type="primary", use_container_width=True) and text.strip():
        run_and_render(text, None)
        msg = st.session_state["history"][-1]["content"]
        st.subheader("Output")
        st.markdown(msg)
        st.download_button("Download vector", msg, file_name="embedding.txt")
    st.stop()

# --- Vision tasks: upload an image, then chat directions about what to extract ---
image_path = None
if task.kind == "vision":
    uploaded = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"]
    )
    if uploaded:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)
        # Write and close the handle so the path is readable on Windows too.
        suffix = os.path.splitext(uploaded.name)[1] or ".png"
        fd, image_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as fh:
            fh.write(uploaded.getvalue())

# --- Chat history ---
for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input: send extraction directions / prompts ---
placeholder = (
    "Tell me what to extract (e.g. 'just the total', 'the table as markdown')..."
    if task.kind == "vision"
    else "Type your message..."
)
user_msg = st.chat_input(placeholder)

if user_msg:
    if task.kind == "vision" and not image_path:
        st.warning("Please upload an image first.")
    else:
        st.session_state["history"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            run_and_render(user_msg, image_path)
            st.markdown(st.session_state["history"][-1]["content"])

# Helpful starting hint for vision tasks.
if task.kind == "vision" and not st.session_state["history"]:
    st.caption(
        "Default direction if you just say 'extract': "
        + (gateway.OCR_PROMPT if task_name == "ocr" else "describe the image.")
    )
