import streamlit as st
import os
import json
import re
import io
import tempfile
import pandas as pd
from fpdf import FPDF
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Quiz Generator", page_icon="📚", layout="centered")
st.title("📚 RAG Quiz Generator")
st.write("Kelompok TextPy - Natural Language Processing (NLP)")

st.sidebar.header("⚙️ Pengaturan Kuis")
tipe_kesulitan = st.sidebar.radio("Tingkat Kesulitan:", ["LOTS (Hafalan/Teoretis)", "HOTS (Analisis/Penalaran)"])
jenis_soal = st.sidebar.selectbox("Jenis Soal:", ["Pilihan Ganda (PG)", "Isian Singkat", "Benar / Salah"])
jumlah_soal = st.sidebar.slider("Jumlah Soal yang Dibuat:", min_value=1, max_value=10, value=3)

TEMPLATE_PROMPT = '''Kamu adalah seorang dosen senior dan ahli penilai evaluasi pendidikan di Indonesia.
Tugasmu adalah menyusun {jumlah_soal} butir soal ujian mandiri bermutu tinggi dengan tipe "{tipe_soal}" dan tingkat kesulitan "{tingkat_kesulitan}" menggunakan rujukan Konteks Materi di bawah ini.

Konteks Materi:
{context}

PANDUAN KUALITAS SOAL (PENTING):
1. Jika tingkat kesulitan adalah "HOTS", soal HARUS menguji kemampuan berpikir tingkat tinggi (Analisis/C4, Evaluasi/C5, atau Kreasi/C6).
2. Hindari pertanyaan tingkat rendah yang hanya menanyakan definisi, hafalan kata, atau ingatan literal teks (LOTS).
3. Gunakan stimulus berupa studi kasus, analisis skenario, hubungan sebab-akibat, atau pemecahan masalah konkret dari materi teks.

Ketentuan Pembuatan Soal Berdasarkan Tipe:
1. Tipe "Pilihan Ganda (PG)": Susun 1 pertanyaan berbasis masalah, sertakan objek pilihan berisi 4 opsi unik (A, B, C, D). Pastikan 3 opsi pengecoh terlihat ilmiah, sangat masuk akal, dan menantang. Kunci jawaban diisi huruf indeksnya saja (A/B/C/D).
2. Tipe "Isian Singkat": Susun pertanyaan langsung yang membutuhkan jawaban analitis pendek dan pasti. Bagian "pilihan" WAJIB diisi null.
3. Tipe "Benar / Salah": Susun sebuah pernyataan analisis yang menuntut pembaca menilai kebenaran logika materi. Kunci jawaban diisi string "Benar" atau "Salah". Bagian "pilihan" WAJIB diisi null.

Aturan Variasi & Anti-Pengulangan:
- Setiap butir soal HARUS menguji sub-topik, konsep spesifik, atau sudut pandang yang berbeda. Dilarang keras membuat 2 soal dengan esensi pertanyaan yang mirip.
- Pilihan pengecoh pada tipe PG dilarang menggunakan kalimat yang sama secara berulang di nomor soal yang berbeda.

Ketentuan Output:
- Seluruh teks soal, pilihan, alasan, dan instruksi wajib menggunakan Bahasa Indonesia yang baku, akademis, dan formal.
- Bagian "sumber_halaman" diisi dengan nomor halaman riil tempat materi tersebut ditemukan di teks (contoh: "Halaman 12"). Jika tidak terdeteksi, tulis "-".
- Format jawaban akhir WAJIB berupa RAW JSON ARRAY yang valid, bersih, tanpa teks sapaan pembuka, tanpa penutup, dan TANPA bungkus markdown seperti ```json ... ```.

Struktur JSON mutlak yang wajib kamu ikuti:
[
  {{
    "no": 1,
    "tipe": "{tipe_soal}",
    "soal": "Teks stimulus kasus/pertanyaan analisis...",
    "pilihan": {{"A": "Teks opsi A", "B": "Teks opsi B", "C": "Teks opsi C", "D": "Teks opsi D"}},
    "kunci": "A",
    "alasan": "Penjelasan ilmiah komprehensif mengapa opsi tersebut benar dan mengapa pengecoh lainnya salah...",
    "sumber_halaman": "Halaman X"
  }}
]

Pertanyaan: Buatkan soal dari materi di atas sesuai kriteria dan format JSON tersebut.
Jawaban:'''

def get_prompt_template(tingkat_kesulitan, tipe_soal, jumlah_soal):
    return PromptTemplate(
        input_variables=["context"],
        template=TEMPLATE_PROMPT
    ).partial(
        tingkat_kesulitan=tingkat_kesulitan,
        tipe_soal=tipe_soal,
        jumlah_soal=str(jumlah_soal)
    )

def inisialisasi_rag_retriever(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_documents(docs)
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = FAISS.from_documents(chunks, embed_model)
    return vector_db.as_retriever(search_kwargs={"k": 3})

def ekstrak_json_bersih(text):
    cleaned_text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'```\s*', '', cleaned_text)
    cleaned_text = cleaned_text.strip()

    try:
        json.loads(cleaned_text)
        return cleaned_text
    except:
        pass

    match_array = re.search(r'\[\s*\{.*\}\s*\]', cleaned_text, re.DOTALL)
    if match_array:
        try:
            json.loads(match_array.group(0))
            return match_array.group(0)
        except:
            pass

    match_single = re.search(r'\{\s*.*\}\s*', cleaned_text, re.DOTALL)
    if match_single:
        try:
            extracted = f"[{match_single.group(0)}]"
            json.loads(extracted)
            return extracted
        except:
            pass

    return "[]"

# ============================================================
# UPLOAD FILE PDF
# ============================================================
uploaded_file = st.file_uploader("Unggah Materi PDF", type=["pdf"])

retriever = None

if uploaded_file is not None:
    # Simpan ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    st.info(f"📄 File '{uploaded_file.name}' berhasil diupload! ({uploaded_file.size/1024:.1f} KB)")

    with st.spinner("Mengindeks dokumen PDF..."):
        try:
            retriever = inisialisasi_rag_retriever(temp_path)
            st.success(f"✨ File '{uploaded_file.name}' siap diproses!")
        except Exception as e:
            st.error(f"Gagal memproses dokumen: {e}")
            retriever = None

if retriever is not None and st.button("🚀 Generate Soal Sekarang"):
    with st.spinner("LLM sedang meracik soal ujian..."):
        try:
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
            raw_llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                task="conversational",
                temperature=0.4,
                max_new_tokens=2000,
                huggingfacehub_api_token=hf_token,
                timeout=300
            )
            llm = ChatHuggingFace(llm=raw_llm)
            prompt = get_prompt_template(tipe_kesulitan, jenis_soal, jumlah_soal)
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
            hasil = rag_chain.invoke({"input": f"Buatkan {jumlah_soal} soal."})
            raw_output = hasil.get('answer', '[]')
            json_bersih = ekstrak_json_bersih(raw_output)
            daftar_soal = json.loads(json_bersih)

            txt_display = "=== HASIL GENERASI SOAL ===\n"
            txt_display += f"Tipe: {jenis_soal} | Tingkat Kesulitan: {tipe_kesulitan}\n\n"

            for item in daftar_soal:
                txt_display += f"Soal No. {item.get('no', 1)}\n"
                txt_display += f"Pertanyaan: {item.get('soal')}\n"

                pilihan = item.get('pilihan')
                if pilihan and isinstance(pilihan, dict) and jenis_soal == "Pilihan Ganda (PG)":
                    for opsi, teks in pilihan.items():
                        if teks:
                            txt_display += f"  {opsi}. {teks}\n"

                txt_display += f"Kunci Jawaban: {item.get('kunci')}\n"
                txt_display += f"Alasan: {item.get('alasan')}\n"
                txt_display += f"Sumber: {item.get('sumber_halaman', 'Tidak terdeteksi')}\n"
                txt_display += "-" * 50 + "\n\n"

            st.session_state['daftar_soal'] = daftar_soal
            st.session_state['txt_display'] = txt_display
            st.session_state['generate_success'] = True

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

if st.session_state.get('generate_success'):
    st.success("✨ Soal berhasil dibuat!")
    st.text_area("Hasil Soal", value=st.session_state['txt_display'], height=400)

    st.subheader("📥 Opsi Unduh")
    daftar_soal = st.session_state['daftar_soal']
    txt_display = st.session_state['txt_display']
    filename_base = "hasil_soal"

    col1, col2, col3, col4 = st.columns(4)
    col1.download_button(label="📄 Unduh .TXT", data=txt_display, file_name=f"{filename_base}.txt", mime="text/plain")
    json_string = json.dumps(daftar_soal, indent=2, ensure_ascii=False)
    col2.download_button(label="💾 Unduh .JSON", data=json_string, file_name=f"{filename_base}.json", mime="application/json")

    excel_rows = []
    for item in daftar_soal:
        pilihan = item.get('pilihan', {}) if isinstance(item.get('pilihan'), dict) else {}
        excel_rows.append({
            "No": item.get('no'),
            "Pertanyaan": item.get('soal'),
            "Pilihan A": pilihan.get('A', '-'),
            "Pilihan B": pilihan.get('B', '-'),
            "Pilihan C": pilihan.get('C', '-'),
            "Pilihan D": pilihan.get('D', '-'),
            "Kunci": item.get('kunci'),
            "Alasan": item.get('alasan'),
            "Sumber": item.get('sumber_halaman', '-')
        })
    df = pd.DataFrame(excel_rows)
    buffer_excel = io.BytesIO()
    with pd.ExcelWriter(buffer_excel, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Bank Soal AI')
    col3.download_button(
        label="📊 Unduh .EXCEL",
        data=buffer_excel.getvalue(),
        file_name=f"{filename_base}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=11)
        clean_pdf_text = txt_display.encode('latin-1', 'replace').decode('latin-1')
        for baris in clean_pdf_text.split('\n'):
            pdf.multi_cell(0, 6, txt=baris)
        pdf_output = pdf.output(dest='S').encode('latin-1')
        col4.download_button(
            label="📕 Unduh .PDF",
            data=pdf_output,
            file_name=f"{filename_base}.pdf",
            mime="application/pdf"
        )
    except Exception as pdf_err:
        col4.warning("PDF gagal dimuat.")