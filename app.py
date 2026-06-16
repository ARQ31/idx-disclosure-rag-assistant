import streamlit as st
import numpy as np
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="IDX Disclosure RAG Assistant",
    page_icon="📊",
    layout="wide"
)


# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
    /* === Google Fonts === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* === Global === */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #151B28 50%, #0E1117 100%);
    }

    /* === Hide default header & footer === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* === Hero Header === */
    .hero-container {
        background: linear-gradient(135deg, #6C63FF 0%, #4F46E5 40%, #7C3AED 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(108, 99, 255, 0.25);
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    .hero-title {
        color: #FFFFFF;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        line-height: 1.15;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
        letter-spacing: 0.3px;
    }

    /* === Glass Card === */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(108, 99, 255, 0.3);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.1);
        transform: translateY(-2px);
    }

    .glass-card h4 {
        color: #FAFAFA;
        font-weight: 600;
        margin-bottom: 0.6rem;
        font-size: 1rem;
    }

    .glass-card p {
        color: rgba(250, 250, 250, 0.7);
        font-size: 0.92rem;
        line-height: 1.65;
        margin: 0;
    }

    /* === Disclaimer Banner === */
    .disclaimer-banner {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.08) 0%, rgba(245, 158, 11, 0.05) 100%);
        border: 1px solid rgba(251, 191, 36, 0.25);
        border-left: 4px solid #F59E0B;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    }

    .disclaimer-banner .disclaimer-label {
        color: #FBBF24;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }


    /* === About Section === */
    .about-section {
        background: rgba(108, 99, 255, 0.06);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
    }

    .about-section h4 {
        color: #A5B4FC;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.6rem;
        letter-spacing: 0.3px;
    }

    .about-section p {
        color: rgba(250, 250, 250, 0.72);
        font-size: 0.9rem;
        line-height: 1.7;
        margin: 0;
    }

    /* === Feature Pills === */
    .feature-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }

    .feature-pill {
        background: rgba(108, 99, 255, 0.12);
        border: 1px solid rgba(108, 99, 255, 0.25);
        color: #C4B5FD;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.35rem 0.85rem;
        border-radius: 50px;
        letter-spacing: 0.2px;
    }

    /* === Section Headers === */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.5rem;
        margin-top: 1.5rem;
        border-left: 8px solid #7C3AED;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
    }

    .section-header .section-icon {
        font-size: 1.3rem;
    }

    .section-header h3 {
        color: #FAFAFA;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.2px;
    }

    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(108, 99, 255, 0.4) 0%, rgba(108, 99, 255, 0) 100%);
        margin-bottom: 1.2rem;
        border: none;
    }

    /* === Metric Cards === */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.3rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: rgba(108, 99, 255, 0.3);
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .metric-card .metric-value {
        color: #A5B4FC;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .metric-card .metric-label {
        color: rgba(250, 250, 250, 0.55);
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* === Sidebar Styling === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141926 0%, #0E1117 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label {
        color: rgba(250, 250, 250, 0.8) !important;
        font-weight: 500;
    }

    .sidebar-brand {
        text-align: center;
        padding: 1rem 0 1.2rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 1.2rem;
    }

    .sidebar-brand .brand-icon {
        font-size: 2rem;
        display: block;
        margin-bottom: 0.3rem;
    }

    .sidebar-brand .brand-text {
        color: rgba(250, 250, 250, 0.5);
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    /* === Buttons === */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.2px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 99, 255, 0.45);
        background: linear-gradient(135deg, #7C73FF 0%, #5F56F5 100%);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* === File Uploader === */
    section[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(108, 99, 255, 0.25);
        border-radius: 16px;
        padding: 1rem;
        transition: all 0.3s ease;
    }

    section[data-testid="stFileUploader"]:hover {
        border-color: rgba(108, 99, 255, 0.5);
        background: rgba(108, 99, 255, 0.03);
    }

    /* === Expander === */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        font-weight: 500;
    }

    /* === Text Area === */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }

    .stTextArea textarea:focus {
        border-color: #6C63FF;
        box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.2);
    }

    /* === Select Box === */
    .stSelectbox > div > div {
        border-radius: 10px;
    }

    /* === How it Works Cards === */
    .how-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .how-step .step-number {
        background: linear-gradient(135deg, #6C63FF, #4F46E5);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
    }

    .how-step .step-content {
        color: rgba(250, 250, 250, 0.72);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .how-step .step-content strong {
        color: #C4B5FD;
    }

    /* === Footer === */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
    }

    .app-footer p {
        color: rgba(250, 250, 250, 0.3);
        font-size: 0.78rem;
        margin: 0;
    }

    /* === Smooth scrollbar === */
    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(108, 99, 255, 0.3);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(108, 99, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# =========================
# Load Groq API key
# =========================
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = ""


# =========================
# Helper functions
# =========================
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return model


def format_table_as_text(table):
    """Convert a pdfplumber table into readable text lines."""
    if not table or len(table) == 0:
        return ""

    lines = []
    headers = table[0] if table[0] else []

    for row in table:
        # Clean each cell
        cells = []
        for cell in row:
            if cell is None:
                cells.append("")
            else:
                cells.append(str(cell).strip())

        # Skip completely empty rows
        if all(c == "" for c in cells):
            continue

        # Join cells with separator
        line = "  |  ".join(c for c in cells if c)
        if line.strip():
            lines.append(line)

    return "\n".join(lines)


def extract_text_from_pdf(uploaded_file):
    all_pages = []

    if HAS_PDFPLUMBER:
        # Use pdfplumber for better table and layout extraction
        with pdfplumber.open(uploaded_file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_texts = []

                # Extract tables separately for structured data
                tables = page.extract_tables()
                table_text = ""
                if tables:
                    for table in tables:
                        table_text += format_table_as_text(table) + "\n\n"

                # Extract full page text
                text = page.extract_text()

                if table_text.strip():
                    page_texts.append(table_text.strip())

                if text and len(text.strip()) > 20:
                    page_texts.append(text)

                combined = "\n\n".join(page_texts)

                if combined and len(combined.strip()) > 20:
                    all_pages.append({
                        "page": page_number,
                        "text": combined
                    })

        # Reset file position for potential re-reads
        uploaded_file.seek(0)
    else:
        # Fallback to pypdf
        reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text(extraction_mode="layout")

            if not text or len(text.strip()) < 20:
                text = page.extract_text()

            if text and len(text.strip()) > 20:
                all_pages.append({
                    "page": page_number,
                    "text": text
                })

    return all_pages


def clean_financial_text(text):
    """Clean and normalize financial document text for better chunking and retrieval."""

    # Normalize line breaks
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that are just dashes, underscores, or decorative separators
        if re.match(r"^[-_=\.\s\|]+$", line):
            continue

        # Skip very short lines that are likely noise (page numbers, etc.)
        if len(line) < 3 and not line.isdigit():
            continue

        # Normalize excessive whitespace within line but preserve some structure
        # Convert 3+ spaces to a tab-like separator for table alignment
        line = re.sub(r"\s{4,}", "  →  ", line)
        line = re.sub(r"\s{2,3}", " ", line)

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Add line breaks before major financial keywords when they follow numbers
    financial_keywords = [
        "Total", "Jumlah", "Laba", "Rugi", "Pendapatan", "Beban",
        "Aset", "Liabilitas", "Ekuitas", "Revenue", "Income", "Profit",
        "Loss", "Assets", "Liabilities", "Equity", "Net", "Gross",
        "Operating", "Comprehensive", "Earnings", "Dividend",
        "Cash", "Kas", "Piutang", "Receivable", "Hutang", "Payable"
    ]

    for keyword in financial_keywords:
        text = re.sub(
            rf"(\d)\s*({keyword})",
            rf"\1\n{keyword}",
            text,
            flags=re.IGNORECASE
        )

    # Remove duplicate consecutive lines
    final_lines = []
    prev_line = None
    for line in text.split("\n"):
        if line.strip() != prev_line:
            final_lines.append(line)
            prev_line = line.strip()

    return "\n".join(final_lines)


def split_text_into_chunks(pages, chunk_size=1000, overlap=200):
    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        # Apply financial text cleaning
        text = clean_financial_text(text)

        # Split by paragraphs first, then merge into chunks
        paragraphs = text.split("\n")
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(paragraph) + 1 > chunk_size and len(current_chunk.strip()) > 100:
                chunks.append({
                    "page": page_number,
                    "text": current_chunk.strip()
                })
                # Keep overlap by taking the last portion of the current chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n" + paragraph
            else:
                current_chunk += "\n" + paragraph if current_chunk else paragraph

        # Don't forget the last chunk
        if len(current_chunk.strip()) > 100:
            chunks.append({
                "page": page_number,
                "text": current_chunk.strip()
            })

    return chunks


def create_embeddings(chunks, model):
    texts = []

    for chunk in chunks:
        texts.append(chunk["text"])

    embeddings = model.encode(texts)
    return embeddings


def search_relevant_chunks(question, chunks, embeddings, model, top_k=5):
    question_embedding = model.encode([question])

    similarity_scores = cosine_similarity(question_embedding, embeddings)[0]
    top_indexes = similarity_scores.argsort()[::-1][:top_k]

    relevant_chunks = []

    for index in top_indexes:
        relevant_chunks.append({
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": similarity_scores[index]
        })

    return relevant_chunks


def call_groq(prompt, api_key, model_name):
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Kamu adalah asisten analis dokumen emiten dan keterbukaan informasi pasar modal Indonesia."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    return answer


def generate_answer(question, relevant_chunks, api_key, model_name):
    context_text = ""

    for i, chunk in enumerate(relevant_chunks, start=1):
        context_text += f"\n[Halaman {chunk['page']}]\n"
        context_text += chunk["text"]
        context_text += "\n---\n"

    prompt = f"""
Kamu adalah asisten RAG profesional untuk menganalisis dokumen emiten yang terdaftar di Bursa Efek Indonesia.

Instruksi penting:
1. Jawab pertanyaan user HANYA berdasarkan konteks dokumen yang diberikan di bawah.
2. Gunakan bahasa Indonesia yang jelas, profesional, dan mudah dipahami.
3. Jangan mengarang angka, fakta, atau informasi yang tidak ada di konteks.
4. Jika informasi tidak ditemukan di konteks, katakan dengan jelas bahwa informasi tersebut tidak ditemukan.
5. Jika ada angka keuangan, tampilkan dengan format yang rapi (gunakan Rp untuk Rupiah, sertakan satuan juta/miliar jika relevan).
6. WAJIB cantumkan referensi halaman sumber dalam jawaban, misalnya "(Halaman 14)" di setiap klaim penting.
7. Jika data berasal dari tabel yang formatnya kurang rapi, usahakan interpretasikan data tersebut secara logis.
8. Jawab dengan detail yang cukup — bukan hanya satu kalimat. Berikan penjelasan dan konteks jika diperlukan.
9. Di akhir jawaban, buat bagian "📄 Referensi Halaman" yang merangkum halaman-halaman sumber yang digunakan.

Konteks dokumen:
{context_text}

Pertanyaan user:
{question}

Berikan jawaban yang terstruktur dan informatif:
"""

    answer = call_groq(prompt, api_key, model_name)
    return answer


def prepare_summary_context(chunks, max_chunks=12, max_characters=12000):
    selected_chunks = chunks[:max_chunks]

    summary_context = ""

    for chunk in selected_chunks:
        summary_context += f"\n[Page {chunk['page']}]\n"
        summary_context += chunk["text"]
        summary_context += "\n"

    summary_context = summary_context[:max_characters]

    return summary_context


def generate_disclosure_summary(chunks, api_key, model_name):
    summary_context = prepare_summary_context(chunks)

    prompt = f"""
Kamu adalah asisten analis keterbukaan informasi emiten.

Tugas kamu adalah membaca konteks dokumen berikut dan membuat analisis ringkas untuk investor atau analis.
Jawaban harus berdasarkan isi dokumen, tidak boleh mengarang angka atau fakta.
Jangan memberikan rekomendasi beli, jual, atau tahan saham.
Gunakan bahasa Indonesia yang profesional tetapi mudah dipahami.

Konteks dokumen:
{summary_context}

Buat output dengan format berikut:

## 1. Detected Document Type
Tentukan jenis dokumen ini.
Contoh jenis dokumen: Financial Statement, Annual Report, Public Expose, Corporate Action Disclosure, Dividend Announcement, Buyback Disclosure, atau Other Disclosure.
Berikan alasan singkat.

## 2. Key Takeaways
Buat 5 poin penting dari dokumen ini untuk investor atau analis.

## 3. Potential Risk Signals
Sebutkan potensi sinyal risiko yang muncul di dokumen.
Gunakan kategori seperti Financial Risk, Legal Risk, Operational Risk, Debt Risk, Liquidity Risk, Corporate Action Risk, atau Market Risk.
Jika tidak ada sinyal risiko yang jelas, katakan bahwa tidak ada sinyal risiko yang jelas berdasarkan konteks yang tersedia.

## 4. Investor Follow-up Checklist
Buat daftar pertanyaan lanjutan yang sebaiknya dicek investor atau analis setelah membaca dokumen ini.
Pertanyaan harus berupa checklist analitis, bukan rekomendasi investasi.
"""

    answer = call_groq(prompt, api_key, model_name)
    return answer


# =========================
# Sidebar
# =========================
st.sidebar.markdown("""
<div class="sidebar-brand">
    <span class="brand-icon">📊</span>
    <span class="brand-text">IDX RAG Assistant</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Settings")

if groq_api_key == "":
    st.sidebar.warning("Groq API Key belum ditemukan.")
    st.sidebar.caption("Tambahkan GROQ_API_KEY di .streamlit/secrets.toml")
else:
    st.sidebar.success("✅ Groq API Key berhasil dimuat.")


# =========================
# Model selection
# =========================
model_options = {
    "llama-3.1-8b-instant": "Llama 3.1 8B — ⚡ Quick & Cheap",
    "openai/gpt-oss-20b": "GPT OSS 20B — ⚖️ Balance",
    "llama-3.3-70b-versatile": "Llama 3.3 70B — 🎯 Most Accurate",
    "openai/gpt-oss-120b": "GPT OSS 120B — 🧠 Smarter"
}

model_name = st.sidebar.selectbox(
    "🤖 Groq Model Name",
    options=list(model_options.keys()),
    format_func=lambda model: model_options[model],
    index=0,
    help="Pilih model LLM yang akan digunakan untuk membuat jawaban."
)

# Short description for the selected model
model_descriptions = {
    "llama-3.1-8b-instant": "💡 Paling cepat & hemat quota. Cocok untuk pertanyaan sederhana.",
    "openai/gpt-oss-20b": "💡 Keseimbangan antara kecepatan dan kualitas jawaban.",
    "llama-3.3-70b-versatile": "💡 Jawaban lebih detail & akurat, tapi lebih boros quota.",
    "openai/gpt-oss-120b": "💡 Kualitas terbaik untuk analisis kompleks. Paling boros quota."
}
st.sidebar.caption(model_descriptions[model_name])

# with st.sidebar.expander("📖 Model Guide"):
#     st.markdown("""
#     **Llama 3.1 8B Instant**  
#     Model paling ringan dan cepat. Cocok untuk penggunaan default, demo portfolio, dan menjaga penggunaan quota tetap aman.

#     **GPT OSS 20B**  
#     Opsi tengah. Kualitas jawaban biasanya lebih baik dari model kecil, tetapi masih lebih ringan dibanding model besar.

#     **Llama 3.3 70B Versatile**  
#     Cocok untuk ringkasan dokumen, risk signal, dan analisis yang lebih kompleks. Kualitas lebih bagus, tetapi penggunaan quota lebih besar.

#     **GPT OSS 120B**  
#     Cocok untuk analisis yang lebih berat dan kompleks. Sebaiknya dipakai hanya untuk demo terbatas karena lebih boros quota.
#     """)

st.sidebar.markdown("---")

# Set default number of retrieved context chunks
top_k = 5

# show_context = st.sidebar.checkbox(
#     "🔬 Debug: Tampilkan retrieved context",
#     value=False,
#     help="Aktifkan untuk melihat potongan dokumen mentah yang diambil oleh sistem. Fitur ini untuk keperluan debugging."
# )

# =========================
# Hero Header
# =========================
st.markdown("""
<div class="hero-container">
    <div class="hero-icon">📊</div>
    <h1 class="hero-title">IDX Disclosure RAG Assistant</h1>
    <p class="hero-subtitle">AI-Powered Analysis for Indonesia Stock Exchange Disclosure Documents</p>
</div>
""", unsafe_allow_html=True)


# =========================
# About Section
# =========================
st.markdown("""
<div class="about-section">
    <h4>📌 Tentang Project Ini</h4>
    <p>
        Aplikasi ini adalah prototype <strong>Retrieval-Augmented Generation (RAG) Assistant</strong> 
        yang dirancang untuk membantu investor dan analis membaca serta menganalisis dokumen 
        keterbukaan informasi atau laporan keuangan emiten yang terdaftar di Bursa Efek Indonesia (IDX). 
        Upload file PDF, dan sistem akan secara otomatis memproses dokumen tersebut untuk menghasilkan 
        ringkasan AI, mendeteksi potensi risk signal, membuat investor follow-up checklist, 
        dan menjawab pertanyaan berbasis isi dokumen.
    </p>
    <div class="feature-pills">
        <span class="feature-pill">📄 PDF Processing</span>
        <span class="feature-pill">🔍 Semantic Search</span>
        <span class="feature-pill">🤖 AI Summary</span>
        <span class="feature-pill">⚠️ Risk Signal Detection</span>
        <span class="feature-pill">❓ Document Q&A</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# Disclaimer
# =========================
st.markdown("""
<div class="disclaimer-banner">
    <div class="disclaimer-label">⚠️ Disclaimer Investasi</div>
    <p>
        Aplikasi ini dibuat untuk keperluan <strong>edukasi dan portfolio AI Engineer</strong>. 
        Seluruh hasil analisis yang dihasilkan oleh sistem ini <strong>bukan merupakan rekomendasi 
        investasi</strong> — termasuk bukan rekomendasi untuk membeli, menjual, atau menahan saham tertentu. 
        Jawaban dihasilkan sepenuhnya berdasarkan dokumen yang diunggah dan kemampuan model AI, 
        sehingga mungkin tidak lengkap atau akurat. Gunakan informasi ini sebagai referensi awal, 
        dan selalu konsultasikan keputusan investasi Anda kepada penasihat keuangan profesional.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# Upload PDF
# =========================
st.markdown("""
<div class="section-header">
    <h3>Upload Dokumen</h3>
</div>
<hr class="section-divider">
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload PDF laporan keuangan atau keterbukaan informasi emiten",
    type=["pdf"],
    help="Format yang didukung: PDF. Pastikan PDF memiliki teks yang bisa diseleksi (bukan hasil scan gambar)."
)

if uploaded_file is None:
    st.markdown("""
    <div class="glass-card">
        <h4>📤 Belum ada dokumen</h4>
        <p>Silakan upload file PDF terlebih dahulu untuk mulai menganalisis dokumen. 
        Anda bisa menggunakan laporan keuangan, prospektus, atau dokumen keterbukaan informasi emiten lainnya.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# =========================
# Process PDF
# =========================
with st.spinner("⏳ Membaca PDF dan membuat embedding..."):
    pages = extract_text_from_pdf(uploaded_file)

    if len(pages) == 0:
        st.error("Teks dari PDF tidak berhasil dibaca. Coba gunakan PDF lain yang teksnya bisa diseleksi.")
        st.stop()

    chunks = split_text_into_chunks(pages)

    if len(chunks) == 0:
        st.error("Chunk dokumen tidak berhasil dibuat. Coba gunakan dokumen yang memiliki teks lebih jelas.")
        st.stop()

    embedding_model = load_embedding_model()
    embeddings = create_embeddings(chunks, embedding_model)

st.success("✅ PDF berhasil diproses dan embedding telah dibuat!")


# =========================
# Document overview
# =========================
st.markdown("""
<div class="section-header">
    <h3>Document Overview</h3>
</div>
<hr class="section-divider">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(pages)}</div>
        <div class="metric-label">Total Pages</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(chunks)}</div>
        <div class="metric-label">Total Chunks</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">MiniLM</div>
        <div class="metric-label">Embedding Model</div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# AI Disclosure Summary
# =========================
st.markdown("""
<div class="section-header">
    <h3>AI Disclosure Summary</h3>
</div>
<hr class="section-divider">
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <h4>📝 Ringkasan Otomatis</h4>
    <p>Fitur ini akan membuat ringkasan otomatis berisi jenis dokumen, poin penting untuk investor, 
    potensi risk signal, dan checklist pertanyaan lanjutan. Klik tombol di bawah untuk memulai analisis.</p>
</div>
""", unsafe_allow_html=True)

summary_button = st.button("🚀 Generate AI Disclosure Summary")

if summary_button:
    if groq_api_key == "":
        st.warning("Groq API Key belum tersedia, jadi summary belum bisa dibuat.")
    else:
        with st.spinner("🧠 Membuat AI Disclosure Summary..."):
            try:
                summary_result = generate_disclosure_summary(
                    chunks=chunks,
                    api_key=groq_api_key,
                    model_name=model_name
                )

                st.markdown(summary_result)

            except Exception as e:
                st.error("Terjadi error saat membuat AI Disclosure Summary.")
                st.write(e)


# =========================
# Ask the document
# =========================
st.markdown("""
<div class="section-header">
    <h3>Ask the Document</h3>
</div>
<hr class="section-divider">
""", unsafe_allow_html=True)

example_question = st.pills(
    "💡 Pertanyaan Cepat (Klik untuk langsung analisa)",
    options=[
        "Ringkas isi dokumen ini dalam 5 poin penting.",
        "Apa potensi risiko yang disebutkan dalam dokumen ini?",
        "Berapa laba bersih perusahaan dan apa penyebab perubahannya?",
        "Apakah dokumen ini membahas aksi korporasi (seperti buyback/dividen)?"
    ]
)

question_input = st.text_area(
    "✍️ Atau tulis pertanyaan spesifik kamu di sini:",
    placeholder="Ketik pertanyaan kamu lalu klik tombol Generate Answer di bawah..."
)

# Priority: if a quick pill is selected, use that. Otherwise use the manual input.
question = example_question if example_question else question_input

ask_button = st.button("🔍 Generate Answer")

# Trigger generation if the user clicked a pill OR clicked the generate button
if ask_button or example_question:
    if not question or question.strip() == "":
        st.warning("⚠️ Pertanyaan tidak boleh kosong. Silakan tulis pertanyaan atau pilih dari Pertanyaan Cepat.")
        st.stop()

    relevant_chunks = search_relevant_chunks(
        question=question,
        chunks=chunks,
        embeddings=embeddings,
        model=embedding_model,
        top_k=top_k
    )

    # if show_context:
    #     st.markdown("""
    #     <div class="section-header">
    #         <span class="section-icon">📎</span>
    #         <h3>Retrieved Context</h3>
    #     </div>
    #     <hr class="section-divider">
    #     """, unsafe_allow_html=True)

    #     for i, chunk in enumerate(relevant_chunks, start=1):
    #         score_pct = chunk['score'] * 100
    #         score_color = "🟢" if score_pct > 60 else "🟡" if score_pct > 40 else "🔴"
    #         with st.expander(f"{score_color} Context {i} | Page {chunk['page']} | Relevance: {score_pct:.1f}%"):
    #             # Use st.code for monospace display that preserves formatting
    #             st.code(chunk["text"], language=None)

    st.markdown("""
    <div class="section-header">
        <span class="section-icon">💡</span>
        <h3>Answer</h3>
    </div>
    <hr class="section-divider">
    """, unsafe_allow_html=True)

    if groq_api_key == "":
        st.warning(
            "Groq API Key belum tersedia. Sistem sudah berhasil mengambil konteks yang relevan, "
            "tetapi belum bisa membuat jawaban dengan LLM."
        )

        st.write("Konteks paling relevan yang ditemukan:")
        st.write(relevant_chunks[0]["text"])

    else:
        with st.spinner("🤖 Membuat jawaban menggunakan LLM..."):
            try:
                answer = generate_answer(
                    question=question,
                    relevant_chunks=relevant_chunks,
                    api_key=groq_api_key,
                    model_name=model_name
                )

                st.markdown(answer)

                source_pages = []

                for chunk in relevant_chunks:
                    source_pages.append(chunk["page"])

                source_pages = sorted(list(set(source_pages)))
                pages_str = ", ".join([f"Halaman {p}" for p in source_pages])

                st.info(f"📄 **Sumber konteks:** {pages_str} — Diambil dari {len(relevant_chunks)} bagian dokumen yang paling relevan.")

            except Exception as e:
                st.error("Terjadi error saat membuat jawaban.")
                st.write(e)


# =========================
# How it works
# =========================
st.markdown("""
<div class="section-header">
    <h3>How This RAG System Works</h3>
</div>
<hr class="section-divider">
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <div class="how-step">
        <div class="step-number">1</div>
        <div class="step-content"><strong>PDF Extraction</strong> — Dokumen PDF diekstrak menjadi teks per halaman menggunakan parser.</div>
    </div>
    <div class="how-step">
        <div class="step-number">2</div>
        <div class="step-content"><strong>Text Chunking</strong> — Teks dipecah menjadi potongan-potongan kecil (chunks) dengan overlap untuk menjaga konteks.</div>
    </div>
    <div class="how-step">
        <div class="step-number">3</div>
        <div class="step-content"><strong>Embedding Generation</strong> — Setiap chunk diubah menjadi vektor embedding menggunakan Sentence Transformer.</div>
    </div>
    <div class="how-step">
        <div class="step-number">4</div>
        <div class="step-content"><strong>Semantic Search</strong> — Pertanyaan user dicocokkan dengan chunk yang paling relevan menggunakan cosine similarity.</div>
    </div>
    <div class="how-step">
        <div class="step-number">5</div>
        <div class="step-content"><strong>LLM Response</strong> — Chunk relevan diberikan ke LLM sebagai konteks untuk menghasilkan jawaban yang akurat.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# # =========================
# # Disclaimer (bottom)
# # =========================
# st.markdown("""
# <div class="section-header">
#     <span class="section-icon">📜</span>
#     <h3>Disclaimer</h3>
# </div>
# <hr class="section-divider">
# """, unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <p>
        Aplikasi ini hanya digunakan untuk tujuan <strong>edukasi dan portfolio</strong>. 
        Hasil analisis <strong>tidak boleh dianggap sebagai rekomendasi investasi</strong>. 
        Sistem ini juga tidak menjamin seluruh informasi penting dalam dokumen berhasil diekstrak, 
        terutama jika PDF memiliki format tabel kompleks atau hasil scan gambar. 
        Selalu lakukan verifikasi mandiri dan konsultasikan dengan penasihat keuangan profesional 
        sebelum mengambil keputusan investasi.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# Footer
# =========================
st.markdown("""
<div class="app-footer">
    <p>IDX Disclosure RAG Assistant • Built for AI Engineer Portfolio • Powered by Groq, Sentence Transformers & Streamlit</p>
</div>
""", unsafe_allow_html=True)