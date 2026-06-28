import os
import json
import re
import nltk
import pandas as pd
from fpdf import FPDF
import io

import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 0. Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="AI Quiz Generator", page_icon="📝", layout="wide")

# NLTK downloads (Menangkap exception umum agar tidak memicu AttributeError)
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except Exception:
    nltk.download('punkt_tab', quiet=True)

# Mengambil API keys dari Environment Variables Secrets
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not GROQ_API_KEY:
    st.error("🔑 GROQ_API_KEY belum diatur di Space Secrets!")
    st.stop()
if not HF_TOKEN:
    st.error("🔑 HUGGINGFACEHUB_API_TOKEN belum diatur di Space Secrets!")
    st.stop()

# Inisialisasi LLM & Embeddings secara Caching agar aplikasi cepat
@st.cache_resource
def init_models():
    llm_model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1, groq_api_key=GROQ_API_KEY)
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm_model, embed_model

llm, embeddings_model = init_models()


# --- 1. Fungsi Pembuat Prompt ---
def get_prompt_template(tingkat_kesulitan, tipe_soal, jumlah_soal, topik_terpakai="Belum ada"):
    if tingkat_kesulitan.upper() == "HOTS":
        panduan_kognitif = """Wajib Menggunakan Level Kognitif HOTS (C4/C5/C6):
- DILARANG KERAS menggunakan kata tanya berbasis hafalan baku seperti "Apa yang dimaksud", "Apa pengertian", atau kalimat berujung "adalah".
- WAJIB menyediakan STIMULUS nyata: studi kasus industri, skenario dilematis, cuplikan data/persentase, atau troubleshooting eror sistem.
- Pertanyaan harus memaksa siswa melakukan analisis komparatif, mendiagnosis masalah, atau mengevaluasi keputusan terbaik."""
    else:
        panduan_kognitif = """Wajib Menggunakan Level Kognitif LOTS (C1/C2/C3):
- Fokus pada pengujian ingatan, pemahaman dasar konsep, fungsi fitur, komponen, atau fakta literal yang tertulis di materi."""

    template = f"""Anda adalah seorang Profesor Senior Pembuat Soal Ujian Akademik dan Pakar Psikometri tingkat tinggi. Task Anda adalah merancang {jumlah_soal} butir soal ujian berkategori {tingkat_kesulitan} dengan tipe "{tipe_soal}" berdasarkan secara ketat hanya pada Konteks Materi yang disediakan.
{panduan_kognitif}
Panduan Kualitas Penjelasan & Semantic:
- Untuk key "alasan", jelaskan konsep dengan kata-kata Anda sendiri, pastikan mudah dipahami, dan berikan alasan yang **komprehensif namun padat**.
- Prioritaskan kejelasan dan presisi dalam alasan, sintetis dari informasi kontekstual yang relevan.
- Pastikan jawaban secara akurat mencerminkan pemahaman mendalam tentang konsep, bukan hanya pengulangan frasa dari teks sumber.
Aturan Validasi Format & Proteksi Redundansi Mutlak:
1. JAWABAN HARUS BERUPA VALID JSON ARRAY SAJA, tanpa teks pembuka, tanpa teks penutup, dan tanpa markdown block seperti ```json.
2. Setiap objek dalam array harus memiliki key wajib dengan huruf kecil: "no", "tipe", "tingkat", "soal", "pilihan", "kunci", "alasan", "sumber_halaman", dan "topik_bahasan".
3. Pada key "topik_bahasan", isi dengan 1 atau 2 kata kunci/istilah teknis inti (ditulis dengan huruf kecil) yang menjadi fokus utama soal yang Anda buat ini.
4. Jika tipe berupa "Pilihan Ganda (PG)", key "pilihan" wajib memiliki sub-key "A", "B", "C", "D". Jika tipe berupa "Isian Singkat" atau "Benar atau Salah", key "pilihan" wajib bernilai null.
⚠️ ATURAN ANTI-DUPLIKASI MUTLAK:
Topik/istilah teknis berikut ini SUDAH digunakan sebelumnya: [{topik_terpakai}].
Anda DILARANG KERAS membuat soal yang membahas atau berfokus pada istilah/topik yang ada di dalam daftar tersebut! Cari sub-bab, frasa kalimat, atau fungsionalitas lain dari Konteks Materi yang benar-benar belum tersentuh.
Konteks Materi:
{{context}}
Output JSON Array:"""
    return PromptTemplate(input_variables=["context"], template=template)


# --- 2. Fungsi Pembantu Pembuat PDF ---
def generate_pdf_bytes(df):
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 14)
            self.cell(0, 10, 'Soal Ujian Generator AI', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.set_font('Helvetica', '', 10)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    for index, row in df.iterrows():
        pdf.set_font('Helvetica', 'B', 11)
        pdf.multi_cell(0, 6, f"No Soal: {row['no']}".encode('latin-1', 'ignore').decode('latin-1'))
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 5, f"Tipe Soal: {row['tipe']} ({row['tingkat']})".encode('latin-1', 'ignore').decode('latin-1'))
        pdf.multi_cell(0, 5, f"Pertanyaan: {row['soal']}".encode('latin-1', 'ignore').decode('latin-1'))

        if 'pilihan' in row and row['pilihan'] is not None:
            if isinstance(row['pilihan'], dict):
                pdf.multi_cell(0, 5, "Pilihan Jawaban:")
                for key, value in row['pilihan'].items():
                    pdf.multi_cell(0, 5, f"  {key}. {value}".encode('latin-1', 'ignore').decode('latin-1'))
            else:
                pdf.multi_cell(0, 5, f"Pilihan: {row['pilihan']}".encode('latin-1', 'ignore').decode('latin-1'))

        pdf.multi_cell(0, 5, f"Kunci Jawaban: {row['kunci']}".encode('latin-1', 'ignore').decode('latin-1'))
        pdf.multi_cell(0, 5, f"Alasan: {row['alasan']}".encode('latin-1', 'ignore').decode('latin-1'))
        pdf.multi_cell(0, 5, f"Sumber Halaman: {row['sumber_halaman']}".encode('latin-1', 'ignore').decode('latin-1'))
        pdf.multi_cell(0, 5, f"Topik Bahasan: {row['topik_bahasan']}".encode('latin-1', 'ignore').decode('latin-1'))
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')


# --- 3. Streamlit Interface / Layout ---
st.title("🤖 AI-Powered Quiz Generator dari PDF")
st.markdown("Unggah file PDF materi Anda, tentukan jumlah target, lalu pilih kombinasi tipe dan tingkat kesulitan soal.")

# Data Kombinasi Soal
all_combinations = [
    {"tipe": "Pilihan Ganda (PG)", "tingkat": "LOTS"},
    {"tipe": "Pilihan Ganda (PG)", "tingkat": "HOTS"},
    {"tipe": "Isian Singkat", "tingkat": "LOTS"},
    {"tipe": "Isian Singkat", "tingkat": "HOTS"},
    {"tipe": "Benar atau Salah", "tingkat": "LOTS"},
    {"tipe": "Benar atau Salah", "tingkat": "HOTS"}
]
display_options = [f"{c['tipe']} ({c['tingkat']})" for c in all_combinations]

# Layout Input menggunakan dua kolom Streamlit
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Unggah File PDF Materi", type=["pdf"])
    
    # Fitur Kustomisasi Jumlah Banyak Soal Per Kategori
    num_questions = st.slider(
        "Jumlah Soal Baru (Per Kategori Terpilih)",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Menentukan berapa banyak butir soal yang dibuat untuk setiap opsi kombinasi yang Anda centang."
    )

with col2:
    selected_options = st.multiselect(
        "Pilih Tipe & Tingkat Kesulitan Soal",
        options=display_options,
        default=display_options,
        help="Pilih minimal satu kombinasi untuk mulai membangkitkan kuis."
    )

# Tombol Aksi Utama
generate_btn = st.button("🚀 Buat Soal", type="primary")

# --- 4. Core Pipeline Engine (Saat Tombol Diklik) ---
if generate_btn:
    if uploaded_file is None:
        st.warning("⚠️ Silakan unggah file PDF terlebih dahulu!")
    elif not selected_options:
        st.warning("⚠️ Mohon pilih minimal satu kombinasi tipe soal pada kotak pilihan!")
    else:
        with st.spinner("⏳ Sedang menganalisis PDF, membangun Vector Store (RAG) dan memproses soal dengan AI Groq..."):
            try:
                # Simpan berkas upload sementara untuk dibaca PyPDFLoader
                temp_pdf_path = f"temp_{uploaded_file.name}"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # RAG Pipeline
                loader = PyPDFLoader(temp_pdf_path)
                docs_raw = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs_raw)

                vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
                retriever = vector_store.as_retriever(search_kwargs={"k": 8})

                # Otomatis mendeteksi topik utama halaman pertama
                sampel_materi = docs_raw[0].page_content[:1000]
                prompt_detektor = f"Baca teks berikut dan berikan 2 atau 3 kata kunci topik utamanya saja tanpa basa-basi: {sampel_materi}"
                topik_otomatis = llm.invoke(prompt_detektor).content.strip()

                docs_konteks = retriever.invoke(topik_otomatis)
                context_text = "\n\n".join(doc.page_content for doc in docs_konteks)

                semua_soal_terbuat = []
                daftar_topik_terpakai = []
                nomor_urut = 1

                # Filter opsi kombinasi pilihan user
                actual_selected_combinations = []
                for opt in selected_options:
                    for comb in all_combinations:
                        if f"{comb['tipe']} ({comb['tingkat']})" == opt:
                            actual_selected_combinations.append(comb)

                # Loop generasi soal via LLM Chain
                for kombinasi in actual_selected_combinations:
                    tipe_saat_ini = kombinasi["tipe"]
                    tingkat_saat_ini = kombinasi["tingkat"]

                    string_topik_terpakai = ", ".join(daftar_topik_terpakai) if daftar_topik_terpakai else "Belum ada"

                    # Menyuntikkan kustomisasi jumlah soal ke template prompt
                    prompt_template = get_prompt_template(
                        tingkat_kesulitan=tingkat_saat_ini,
                        tipe_soal=tipe_saat_ini,
                        jumlah_soal=num_questions,
                        topik_terpakai=string_topik_terpakai
                    )

                    rag_chain = (
                        {"context": lambda x: context_text}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )

                    raw_output = rag_chain.invoke({})

                    # Ekstraksi array JSON secara aman
                    match = re.search(r'\[\s*\{.*\}\s*\]', raw_output, re.DOTALL)
                    json_clean = match.group(0) if match else raw_output

                    try:
                        data_soal = json.loads(json_clean)
                        for item in data_soal:
                            item["no"] = nomor_urut
                            semua_soal_terbuat.append(item)

                            topik_dari_ai = item.get("topik_bahasan", "").strip().lower()
                            if topik_dari_ai and topik_dari_ai not in daftar_topik_terpakai:
                                daftar_topik_terpakai.append(topik_dari_ai)

                            nomor_urut += 1
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON untuk nomor {nomor_urut}: {e}")

                # Bersihkan file PDF temp setelah selesai diproses
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

                # Tampilkan hasil jika pembuatan sukses
                if semua_soal_terbuat:
                    st.success(f"🎉 Sukses membuat total {len(semua_soal_terbuat)} butir soal!")
                    
                    df_soal = pd.DataFrame(semua_soal_terbuat)
                    
                    # 1. Tampilkan pratinjau sebagai Teks Biasa/Markdown yang rapi (BUKAN TABEL)
                    st.subheader("📝 Pratinjau Soal")
                    
                    for item in semua_soal_terbuat:
                        with st.container():
                            st.markdown(f"### **Soal No. {item['no']}**")
                            st.markdown(f"**Tipe:** {item['tipe']} | **Tingkat Kesulitan:** {item['tingkat']}")
                            st.markdown(f"❓ **Pertanyaan:**\n{item['soal']}")
                            
                            # Jika ada pilihan ganda, tampilkan opsinya berderet ke bawah
                            if item.get('pilihan') and isinstance(item['pilihan'], dict):
                                st.markdown("**Pilihan Jawaban:**")
                                for opsi, teks_opsi in item['pilihan'].items():
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**{opsi}.** {teks_opsi}")
                            
                            st.markdown(f"✅ **Kunci Jawaban:** {item['kunci']}")
                            st.markdown(f"💡 **Alasan/Penjelasan:**\n*{item['alasan']}*")
                            st.markdown(f"📖 *Sumber: Halaman {item['sumber_halaman']} (Topik: {item['topik_bahasan']})*")
                            st.markdown("---") # Garis pembatas antar soal

                    # 2. Render Dokumen ke Memory Buffer untuk Tombol Unduhan Langsung
                    st.subheader("📥 Pusat Unduhan Dokumen")
                    
                    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

                    # Export JSON
                    json_bytes = json.dumps(semua_soal_terbuat, indent=2, ensure_ascii=False).encode('utf-8')
                    with dl_col1:
                        st.download_button("💾 Unduh format JSON", data=json_bytes, file_name="output_soal.json", mime="application/json")

                    # Export Excel
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        df_soal.to_excel(writer, index=False, sheet_name='Quiz_AI')
                    excel_bytes = excel_buffer.getvalue()
                    with dl_col2:
                        st.download_button("📊 Unduh format Excel", data=excel_bytes, file_name="output_soal.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    # Export TXT
                    txt_buffer = io.StringIO()
                    for index, row in df_soal.iterrows():
                        for col, value in row.items():
                            txt_buffer.write(f"{col}: {value}\n")
                        txt_buffer.write("\n" + "="*50 + "\n\n")
                    txt_bytes = txt_buffer.getvalue().encode('utf-8')
                    with dl_col3:
                        st.download_button("📝 Unduh format TXT", data=txt_bytes, file_name="output_soal.txt", mime="text/plain")

                    # Export PDF
                    pdf_bytes = generate_pdf_bytes(df_soal)
                    with dl_col4:
                        st.download_button("📕 Unduh format PDF", data=pdf_bytes, file_name="output_soal.pdf", mime="application/pdf")

                else:
                    st.error("❌ Gagal memetakan keluaran AI. Silakan klik ulang tombol 'Buat Soal'.")

            except Exception as error:
                st.error(f"💥 Terjadi kesalahan sistem: {str(error)}")
