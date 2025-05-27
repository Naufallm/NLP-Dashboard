
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# import plotly.graph_objects as go # Uncomment jika digunakan
from sklearn.metrics import confusion_matrix
import pickle
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
import os

# --- Konfigurasi Path & Konstanta ---
DATA_DIR = "dashboard_files" 
W2V_VECTOR_SIZE_DASH = 100 # Harus sama dengan W2V_VECTOR_SIZE di skrip persiapan

# --- Fungsi Pemuatan Data (dengan caching) ---
@st.cache_data
def load_predictions_data():
    path = os.path.join(DATA_DIR, "dashboard_all_predictions.csv")
    if not os.path.exists(path):
        st.error(f"File tidak ditemukan: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_metrics_summary():
    path = os.path.join(DATA_DIR, "dashboard_metrics_summary.csv")
    if not os.path.exists(path):
        st.error(f"File tidak ditemukan: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_error_analysis():
    path = os.path.join(DATA_DIR, 'dashboard_error_analysis.xlsx')
    if not os.path.exists(path):
        st.error(f"File tidak ditemukan: {path}")
        return {}
    try:
        xls = pd.ExcelFile(path)
        return {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
    except Exception as e:
        st.error(f"Gagal memuat file analisis error '{path}': {e}")
        return {}

@st.cache_resource
def load_label_encoder():
    path = os.path.join(DATA_DIR, "label_encoder.pkl")
    if not os.path.exists(path):
        st.error(f"File tidak ditemukan: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model_pickle(model_filename):
    path = os.path.join(DATA_DIR, model_filename)
    if not os.path.exists(path):
        st.error(f"File model tidak ditemukan: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_w2v_gensim_model(model_filename="word2vec.model"):
    path = os.path.join(DATA_DIR, model_filename)
    if not os.path.exists(path):
        st.warning(f"File Word2Vec tidak ditemukan: {path}. Fitur prediksi Word2Vec mungkin tidak berfungsi.")
        return None
    return Word2Vec.load(path)

# --- Inisialisasi Aplikasi Utama ---
st.set_page_config(layout="wide", page_title="Dashboard Clickbait")
st.title("📊 Dashboard Analisis Klasifikasi Berita Clickbait")

# --- Memuat Data dan Objek Penting ---
predictions_df = load_predictions_data()
metrics_df = load_metrics_summary()
error_analysis_dfs = load_error_analysis()
label_encoder = load_label_encoder()

if predictions_df.empty or metrics_df.empty or label_encoder is None:
    st.error("Gagal memuat data penting. Pastikan skrip persiapan data (Tahap 2) berjalan sukses dan semua file output ada di folder 'dashboard_files'.")
    st.stop()

# Membuat mapping label dari LabelEncoder
label_mapping = {i: label_encoder.classes_[i] for i in range(len(label_encoder.classes_))}
clickbait_label_text_found = next((cls_text for cls_text in label_encoder.classes_ if 'clickbait' in cls_text.lower()), None)

if clickbait_label_text_found:
    CLICKBAIT_CLASS_INDEX = label_encoder.transform([clickbait_label_text_found])[0]
else:
    st.warning("Label 'clickbait' tidak terdeteksi di LabelEncoder. Menggunakan kelas indeks 1 (atau 0 jika hanya ada 1 kelas) sebagai default untuk clickbait. Harap periksa label Anda.")
    CLICKBAIT_CLASS_INDEX = 1 if len(label_encoder.classes_) > 1 else 0

available_model_keys = metrics_df['Model_Key'].unique().tolist()
available_model_display_names = metrics_df['Model_Display_Name'].unique().tolist()

# --- Sidebar Navigasi ---
st.sidebar.header("Menu Navigasi")
selected_page = st.sidebar.radio("Pilih Halaman:",
                                 ["Ringkasan Kinerja", "Detail Model & Confusion Matrix", "Analisis Kesalahan", "Analisis Fitur", "Prediksi Langsung"])

# --- Konten Halaman ---
if selected_page == "Ringkasan Kinerja":
    st.header("🚀 Ringkasan Kinerja Model")
    st.subheader("Tabel Perbandingan Metrik")
    display_metrics = metrics_df.set_index('Model_Display_Name')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
    st.dataframe(display_metrics.style.format("{:.4f}"))

    metric_to_plot = st.selectbox("Pilih Metrik untuk Grafik Perbandingan:",
                                  ['Accuracy', 'Precision', 'Recall', 'F1-Score'], key="ringkasan_metric_select")
    if metric_to_plot:
        fig_metrics_comp = px.bar(metrics_df, x='Model_Display_Name', y=metric_to_plot,
                                    color='Model_Display_Name', title=f"Perbandingan {metric_to_plot} Antar Model",
                                    text_auto='.4f', labels={'Model_Display_Name': 'Model'})
        fig_metrics_comp.update_layout(xaxis_title="Model", yaxis_title=metric_to_plot, showlegend=False)
        st.plotly_chart(fig_metrics_comp, use_container_width=True)

    st.subheader("Distribusi Label Aktual (Data Uji)")
    if 'true_label_text' in predictions_df.columns:
        true_label_counts = predictions_df['true_label_text'].value_counts()
        fig_dist = px.pie(values=true_label_counts.values, names=true_label_counts.index,
                          title="Distribusi Label Aktual pada Data Uji", hole=0.3)
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.warning("Kolom 'true_label_text' tidak ditemukan untuk menampilkan distribusi label.")

elif selected_page == "Detail Model & Confusion Matrix":
    st.header("🔎 Detail Model dan Confusion Matrix")
    selected_model_display = st.selectbox("Pilih Model:", available_model_display_names, key="detail_model_select")

    if selected_model_display:
        model_key = metrics_df[metrics_df['Model_Display_Name'] == selected_model_display]['Model_Key'].iloc[0]
        model_metrics = metrics_df[metrics_df['Model_Key'] == model_key].iloc[0]

        st.subheader(f"Metrik untuk: {selected_model_display}")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
        m_col2.metric("Precision", f"{model_metrics['Precision']:.4f}")
        m_col3.metric("Recall", f"{model_metrics['Recall']:.4f}")
        m_col4.metric("F1-Score", f"{model_metrics['F1-Score']:.4f}")

        pred_col_num = f"pred_numeric_{model_key}"
        if pred_col_num in predictions_df.columns and 'true_label_numeric' in predictions_df.columns:
            y_true = predictions_df['true_label_numeric']
            y_pred = predictions_df[pred_col_num]
            
            cm = confusion_matrix(y_true, y_pred)
            cm_labels = [label_mapping.get(i, str(i)) for i in sorted(np.unique(np.concatenate((y_true, y_pred))))]

            fig_cm = px.imshow(cm, labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                               x=cm_labels, y=cm_labels, text_auto=True,
                               title=f"Confusion Matrix untuk {selected_model_display}",
                               color_continuous_scale=px.colors.sequential.Blues)
            fig_cm.update_layout(xaxis_title="Label Prediksi", yaxis_title="Label Aktual")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning(f"Kolom prediksi atau label asli tidak ditemukan untuk model {selected_model_display}.")


elif selected_page == "Analisis Kesalahan":
    st.header("🧐 Analisis Kesalahan Prediksi")
    selected_model_display_err = st.selectbox("Pilih Model untuk Analisis Kesalahan:", available_model_display_names, key="error_model_select")

    if selected_model_display_err:
        model_key_err = metrics_df[metrics_df['Model_Display_Name'] == selected_model_display_err]['Model_Key'].iloc[0]
        if model_key_err in error_analysis_dfs:
            st.subheader(f"Contoh Prediksi Salah oleh: {selected_model_display_err}")
            df_errors = error_analysis_dfs[model_key_err]
            if not df_errors.empty:
                st.dataframe(df_errors.style.set_properties(**{'white-space': 'pre-wrap', 'text-align': 'left'}), use_container_width=True)
            else:
                st.success(f"Tidak ada kesalahan prediksi yang tercatat untuk model {selected_model_display_err}!")
        else:
            st.warning(f"Data analisis kesalahan untuk model '{model_key_err}' tidak tersedia.")

elif selected_page == "Analisis Fitur":
    st.header("🔬 Analisis Fitur (Feature Importance)")
    feature_model_options = [name for name in available_model_display_names if "DT" in name and ("BoW" in name or "TFIDF" in name)]

    if not feature_model_options:
        st.warning("Analisis fitur saat ini hanya tersedia untuk model Decision Tree dengan BoW/TF-IDF.")
    else:
        selected_model_feat = st.selectbox("Pilih Model (DT - BoW/TFIDF):", feature_model_options, key="feat_model_select")
        if selected_model_feat:
            model_key_feat = metrics_df[metrics_df['Model_Display_Name'] == selected_model_feat]['Model_Key'].iloc[0]
            model_pkl_filename = f"{model_key_feat.lower()}.pkl"
            
            vectorizer_pkl_filename = ""
            if "BoW" in model_key_feat: vectorizer_pkl_filename = "bow_vectorizer.pkl"
            elif "TFIDF" in model_key_feat: vectorizer_pkl_filename = "tfidf_vectorizer.pkl"

            model_obj = load_model_pickle(model_pkl_filename)
            vectorizer_obj = load_model_pickle(vectorizer_pkl_filename)

            if model_obj and vectorizer_obj and hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
                feature_names = vectorizer_obj.get_feature_names_out()
                
                feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False).head(20)

                fig_imp = px.bar(feat_imp_df, x='importance', y='feature', orientation='h',
                                 title=f"Top 20 Fitur Penting untuk {selected_model_feat}",
                                 color='importance', color_continuous_scale='Viridis')
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning(f"Tidak dapat menampilkan feature importance. Pastikan model dan vectorizer ({model_pkl_filename}, {vectorizer_pkl_filename}) ada dan model memiliki 'feature_importances_'.")

elif selected_page == "Prediksi Langsung":
    st.header("🔮 Prediksi Berita Baru")
    selected_model_pred_disp = st.selectbox("Pilih Model untuk Prediksi:", available_model_display_names, key="live_pred_model_select")
    news_input = st.text_area("Masukkan Judul Berita:", height=100, key="live_pred_input")

    if st.button("Prediksi Clickbait", key="live_pred_button"):
        if not news_input.strip():
            st.warning("Harap masukkan judul berita.")
        elif not selected_model_pred_disp:
            st.warning("Harap pilih model.")
        else:
            model_key_pred = metrics_df[metrics_df['Model_Display_Name'] == selected_model_pred_disp]['Model_Key'].iloc[0]
            model_pkl_pred = f"{model_key_pred.lower()}.pkl"
            model_to_use = load_model_pickle(model_pkl_pred)
            
            if not model_to_use:
                st.error(f"Gagal memuat model {model_pkl_pred} untuk prediksi.")
            else:
                transformed_input = None
                if "Original" in model_key_pred:
                    def extract_basic_features_live(text_list):
                        feats = []
                        for t_str in text_list:
                            s = str(t_str)
                            feats.append([len(s.split()), len(s)])
                        return np.array(feats)
                    transformed_input = extract_basic_features_live([news_input])
                elif "BoW" in model_key_pred:
                    vectorizer = load_model_pickle("bow_vectorizer.pkl")
                    if vectorizer: transformed_input = vectorizer.transform([news_input])
                elif "TFIDF" in model_key_pred:
                    vectorizer = load_model_pickle("tfidf_vectorizer.pkl")
                    if vectorizer: transformed_input = vectorizer.transform([news_input])
                elif "Word2Vec" in model_key_pred:
                    w2v_g_model = load_w2v_gensim_model()
                    if w2v_g_model:
                        tokens = word_tokenize(news_input.lower())
                        def get_w2v_embedding_live(tok_list, w2v_m, vec_size):
                            embs = [w2v_m.wv[w] for w in tok_list if w in w2v_m.wv]
                            return np.mean(embs, axis=0) if embs else np.zeros(vec_size)
                        transformed_input = np.array([get_w2v_embedding_live(tokens, w2v_g_model, W2V_VECTOR_SIZE_DASH)])
                        transformed_input = np.nan_to_num(transformed_input)
                
                if transformed_input is not None:
                    pred_num = model_to_use.predict(transformed_input)[0]
                    pred_text = label_mapping.get(pred_num, f"Label_Unknown ({pred_num})")
                    
                    st.subheader("Hasil Prediksi:")
                    if pred_num == CLICKBAIT_CLASS_INDEX:
                        st.error(f"**{pred_text}**")
                    else:
                        st.success(f"**{pred_text}**")

                    if hasattr(model_to_use, "predict_proba"):
                        pred_proba = model_to_use.predict_proba(transformed_input)[0]
                        st.write("Skor Kepercayaan:")
                        for i in range(len(label_encoder.classes_)):
                            class_label = label_encoder.classes_[i]
                            prob_val = pred_proba[i]
                            st.write(f"{class_label}: {prob_val*100:.2f}%")
                            st.progress(float(prob_val))
                else:
                    st.error("Gagal mentransformasi input untuk model yang dipilih. Vectorizer atau model Word2Vec mungkin tidak termuat.")
