---
title: AI Powered Quiz Generator RAG LLM
emoji: 📝
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

🤖 AI-Powered Quiz Generator dari Dokumen PDF (RAG + LLM)
Aplikasi ini adalah sistem Pembangkit Kuis Otomatis (Automated Question Generator) yang memanfaatkan teknologi Retrieval-Augmented Generation (RAG) dan model inferensi berkecepatan tinggi dari Groq untuk merancang paket soal ujian akademik yang kontekstual, kredibel, dan tervalidasi langsung dari materi pembelajaran berformat PDF.

Sistem dilengkapi dengan algoritma pelacakan topik (anti-duplication memory) guna memastikan variasi istilah teknis pada setiap soal tidak redundan.

🚀 Fitur Utama & Arsitektur RAG
PDF Semantic Chunking: Ekstraksi dokumen materi menggunakan PyPDFLoader dan pemecahan teks pintar dengan RecursiveCharacterTextSplitter (chunk size: 900, overlap: 200).
FAISS Vector Storage: Pemetaan representasi vektor teks dokumen menggunakan model embedding publik sentence-transformers/all-MiniLM-L6-v2.
Automatic Topic Detection: Sistem membaca halaman awal dokumen secara otomatis untuk mendeteksi kata kunci topik utama sebelum memicu pipeline pencarian dokumen kontekstual.
Multi-Format Export Engine: Setelah paket soal berhasil dirangkai, data kuis dikonversi ke dalam 4 tipe ekstensi file unduhan sekaligus: JSON, Excel, TXT, dan PDF.
🛠️ Dokumentasi Antarmuka UI (Streamlit Interface)
Antarmuka aplikasi dibangun menggunakan framework Streamlit dengan tata letak satu halaman interaktif yang dibagi menjadi beberapa komponen fungsional berikut:

1. Panel Atas & Pengaturan Konfigurasi (Input Workspace)
Unggah File PDF Materi (st.file_uploader): Komponen seret-dan-lepas (drag-and-drop) yang menerima satu dokumen PDF rujukan utama (ground truth). Unggahan berkas ini langsung memicu inisialisasi rantai pembuatan indeks vektor ke database lokal FAISS.
Grup Kotak Centang Kombinasi (st.checkbox / st.multiselect): Menyediakan pilihan kombinasi silang antara tipe soal (Pilihan Ganda, Isian Singkat, Benar atau Salah) dan tingkat kesulitan kognitif (LOTS atau HOTS).
2. Tombol Aksi Utama (Execution Trigger)
Tombol Aksentuasi (st.button): Tombol eksekusi utama dengan label "🚀 Buat Soal". Ketika diklik, tombol ini memicu fungsi pemrosesan RAG Chain dan interaksi LLM Groq, dibungkus di dalam animasi pemuatan berbasis st.spinner().
3. Panel Status & Tabel Pratinjau (Data Preview Dashboard)
Kotak Status Proses (st.info / st.success): Menampilkan indikator teks real-time mengenai tahapan pemrosesan atau pesan galat jika berkas PDF belum diunggah.
Tabel Soal Digital (st.dataframe): Menampilkan pratinjau data soal terstruktur yang berhasil dirangkai oleh LLM dalam bentuk baris dan kolom interaktif (Nomor, Tipe, Tingkat, Soal, Pilihan, Kunci, Alasan, Sumber Halaman, Topik).
4. Pusat Unduhan Dokumen (Export & Download Hub)
Tombol Unduh Otomatis (st.download_button): Terdiri dari 4 tombol unduhan terpisah (JSON, Excel, TXT, PDF). Komponen ini bersifat dinamis dan hanya akan muncul ke permukaan layar pengguna setelah file buffer biner sukses digenerasikan pada memori lokal server, sehingga mencegah pengunduhan file kosong.
