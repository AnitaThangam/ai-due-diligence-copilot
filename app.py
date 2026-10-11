import os
import streamlit as st
from dotenv import load_dotenv
from src.documents import extract_pdf_pages, chunk_pages
from src.rag import build_index, retrieve_chunks, generate_answer, reset_collection

load_dotenv()
st.set_page_config(page_title="AI Due Diligence Copilot", page_icon="📊", layout="wide")

st.title("📊 AI Due Diligence Copilot")
st.caption("Explore financial and business documents with source-backed AI answers.")
st.warning("Portfolio prototype only. This tool does not provide investment, legal, or accounting advice. Verify every conclusion against the original documents.")

with st.sidebar:
    st.header("1. Add documents")
    uploaded_files = st.file_uploader(
        "Upload annual reports, filings, or investor presentations (PDF)",
        type=["pdf"], accept_multiple_files=True
    )
    st.caption("Start with the synthetic sample PDFs in data/sample_pdfs.")
    process = st.button("Process uploaded PDFs", type="primary", use_container_width=True)
    if st.button("Clear local document index", use_container_width=True):
        reset_collection()
        st.session_state["indexed_files"] = []
        st.success("Local index cleared. Process your documents again.")

if "indexed_files" not in st.session_state:
    st.session_state["indexed_files"] = []

if process:
    if not uploaded_files:
        st.error("Upload at least one PDF first.")
    else:
        with st.spinner("Extracting text, creating embeddings, and indexing document chunks..."):
            all_chunks = []
            for pdf in uploaded_files:
                if pdf.size > 20 * 1024 * 1024:
                    st.error(f"{pdf.name} exceeds the 20 MB limit and was skipped.")
                    continue
                pages = extract_pdf_pages(pdf.getvalue(), pdf.name)
                all_chunks.extend(chunk_pages(pages))
            if not all_chunks:
                st.error("No selectable text was found. Scanned PDFs need OCR, which this starter version does not include.")
            else:
                count = build_index(all_chunks)
                st.session_state["indexed_files"] = sorted(set(c["source"] for c in all_chunks))
                st.success(f"Indexed {count} text chunks from {len(st.session_state['indexed_files'])} PDF(s).")

st.subheader("2. Ask a due diligence question")
question = st.text_input(
    "Question",
    placeholder="Example: What risks does management mention about operating costs?"
)
top_k = st.slider("Number of source passages to retrieve", min_value=2, max_value=8, value=4)

if st.button("Analyze documents", type="primary"):
    if not question.strip():
        st.error("Enter a question first.")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Add it to your .env file and restart Streamlit.")
    else:
        try:
            with st.spinner("Searching the indexed documents..."):
                results = retrieve_chunks(question, top_k=top_k)
            if not results:
                st.info("No indexed passages found. Upload and process PDFs first.")
            else:
                with st.spinner("Generating a source-grounded answer..."):
                    answer = generate_answer(question, results)
                st.markdown("### Answer")
                st.write(answer)
                st.markdown("### Retrieved evidence")
                for i, item in enumerate(results, start=1):
                    with st.expander(f"{i}. {item['source']} — page {item['page']} (distance: {item['distance']:.3f})"):
                        st.write(item["text"])
        except Exception as exc:
            st.error("The request failed. Check your API key, internet connection, and terminal error.")
            st.exception(exc)

st.divider()
st.caption("Prototype note: answers depend on the quality of the uploaded documents and retrieval. A citation indicates retrieved context, not a guarantee that the conclusion is correct.")
