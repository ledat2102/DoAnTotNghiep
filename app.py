import streamlit as st
import pandas as pd
import joblib
import re
import os
import urllib.request
from pyvi import ViTokenizer
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from fpdf import FPDF

# ==========================================================
# 0. TỰ ĐỘNG CẤU HÌNH FONT TIẾNG VIỆT (LINK GỐC GOOGLE - CHỐNG 404)
# ==========================================================
FONT_REGULAR = "Roboto-Regular.ttf"
FONT_BOLD = "Roboto-Bold.ttf"

if not os.path.exists(FONT_REGULAR) or not os.path.exists(FONT_BOLD):
    with st.spinner("⏳ Đang tải Font tiếng Việt gốc từ Google (Chỉ tải 1 lần duy nhất)..."):
        try:
            import ssl
            # Bỏ qua chứng chỉ SSL cứng nhắc của Windows
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Sử dụng Link Raw từ Source gốc của Google Fonts (Bao không chết)
            url_reg = "https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf"
            url_bold = "https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Bold.ttf"
            
            urllib.request.urlretrieve(url_reg, FONT_REGULAR)
            urllib.request.urlretrieve(url_bold, FONT_BOLD)
            st.toast("✅ Đã tải Font thành công! Tính năng Xuất PDF đã sẵn sàng.")
        except Exception as e:
            st.error(f"Lỗi tải Font: {e}. Bạn hãy tải file arial.ttf bỏ vào chung thư mục nhé!")
# ==========================================
# 1. CẤU HÌNH TRANG VÀ LOAD MÔ HÌNH (CACHED)
# ==========================================
st.set_page_config(page_title="Hệ thống Phân tích Cảm xúc Shopee", layout="wide")

@st.cache_resource
def load_models():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('svm_model.pkl')
    return vectorizer, model

vectorizer, model = load_models()

# ==========================================
# 2. CÁC HÀM XỬ LÝ LÕI
# ==========================================
def tien_xu_ly(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ViTokenizer.tokenize(text)

# 1. DANH SÁCH TỪ ĐỆM CẦN LỌC (Bổ sung thêm các hư từ gây nhiễu cụm)
STOP_WORDS_CUSTOM = [
    "cái", "thì", "là", "mà", "có", "cho", "của", "và", "với", "mình", "shop", 
    "ơi", "này", "khi", "nha", "nhé", "ạ", "rồi", "nữa", "đã", "đang", "được", 
    "các", "những", "thấy", "cũng", "sản_phẩm", "mọi_người", "như_thế", "một_số",
    "để", "nửa", "bao_giờ", "đến", "đi", "về", "ra", "lên", "xuống", "một", "hai"
]

# 2. TỪ ĐIỂN SẮC THÁI DẠNG TỪ GHÉP PYVI
TICH_CUC_LEXICON = ["đẹp", "tốt", "nhanh", "rẻ", "xịn", "thơm", "ưng", "tuyệt", "chuẩn", "xuất_sắc", "ok", "dày", "nhiệt_tình", "thích", "mềm", "mịn"]
TIEU_CUC_LEXICON = ["tệ", "chê", "xấu", "chậm", "rách", "mỏng", "đắt", "lỗi", "dỏm", "không", "ko", "k", "chưa", "thất_vọng", "chán", "nhầm", "ngắn", "cũ", "rão", "kém", "thái_độ"]

def tim_tu_khoa_nhieu_nhat(text_series, loai_cam_xuc, top_n=5):
    clean_sentences = [str(text).strip() for text in text_series]
    if not clean_sentences: return []

    # Quét cụm từ dài từ 1 đến 4 từ
    vectorizer = CountVectorizer(ngram_range=(3, 4), token_pattern=r'(?u)\b\w+\b', stop_words=STOP_WORDS_CUSTOM)
    try:
        X = vectorizer.fit_transform(clean_sentences)
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        
        # BƯỚC 1: LỌC QUA TỪ ĐIỂN SẮC THÁI VÀ ÉP ĐỘ DÀI PHẢI TỪ 2 TỪ TRỞ LÊN
        ket_qua_tam = []
        for cum_tu, freq in words_freq:
            cum_tu_dep = cum_tu.replace('_', ' ')
            # ĐIỀU KIỆN TIÊN QUYẾT: Loại bỏ hoàn toàn từ đơn lẻ (như: nhanh, tốt, đẹp đứng một mình)
            if len(cum_tu_dep.split()) < 2:
                continue
                
            words_in_cum_tu = cum_tu.split()
            if loai_cam_xuc == "Tích cực":
                co_khen = any(w in words_in_cum_tu for w in TICH_CUC_LEXICON)
                co_che = any(w in words_in_cum_tu for w in TIEU_CUC_LEXICON)
                if co_khen and not co_che:
                    ket_qua_tam.append((cum_tu_dep, freq))
                    
            elif loai_cam_xuc == "Tiêu cực":
                co_che = any(w in words_in_cum_tu for w in TIEU_CUC_LEXICON)
                if co_che:
                    ket_qua_tam.append((cum_tu_dep, freq))

        # BƯỚC 2: THUẬT TOÁN ĐỘ TRÙNG LẬP TẬP HỢP TỪ (Token-Set Overlap)
        ket_qua_sach = []
        for cum_tu, freq in ket_qua_tam:
            words_set = set(cum_tu.split())
            is_trung_lap = False
            
            for chosen_cum, _ in ket_qua_sach:
                chosen_set = set(chosen_cum.split())
                giao = words_set.intersection(chosen_set)
                
                # Tính tỉ lệ trùng lặp dựa trên cụm ngắn hơn
                ti_le_trung = len(giao) / min(len(words_set), len(chosen_set))
                
                # Nếu trùng nhau từ 50% số từ trở lên -> coi như cùng bản chất, loại bỏ cụm đi sau
                if ti_le_trung >= 0.5:
                    is_trung_lap = True
                    break
            
            if not is_trung_lap:
                ket_qua_sach.append((cum_tu, freq))
                
        # Sắp xếp và cắt lấy đúng Top 5 cụm từ chất lượng nhất
        return sorted(ket_qua_sach, key=lambda x: x[1], reverse=True)[:top_n]
    except ValueError:
        return []

# ==========================================================
# 2.5 HÀM SINH BÁO CÁO PDF TRỰC TIẾP (DÙNG CHO PHẦN TẢI FILE)
# ==========================================================
def tao_file_pdf(df_result, top_khen, top_che):
    tong_so = len(df_result)
    so_tich_cuc = len(df_result[df_result['Nhãn'] == 'Tích cực'])
    so_tieu_cuc = len(df_result[df_result['Nhãn'] == 'Tiêu cực'])
    so_trung_tinh = len(df_result[df_result['Nhãn'] == 'Trung tính'])
    
    tl_tich_cuc = round((so_tich_cuc / tong_so) * 100, 1) if tong_so > 0 else 0
    tl_tieu_cuc = round((so_tieu_cuc / tong_so) * 100, 1) if tong_so > 0 else 0
    tl_trung_tinh = round((so_trung_tinh / tong_so) * 100, 1) if tong_so > 0 else 0

    pdf = FPDF()
    pdf.add_page()
    
    # Cấu hình font chữ Unicode Roboto
    pdf.add_font("Roboto", "", FONT_REGULAR, uni=True)
    pdf.add_font("Roboto", "B", FONT_BOLD, uni=True)
    
    # Tiêu đề báo cáo
    pdf.set_font("Roboto", "B", size=16)
    pdf.set_text_color(238, 77, 45) # Màu cam Shopee
    pdf.cell(0, 12, "BÁO CÁO PHÂN TÍCH CẢM XÚC PHẢN HỒI KHÁCH HÀNG", ln=True, align='C')
    
    # Metadata thông tin đề tài
    pdf.set_font("Roboto", "", size=10)
    pdf.set_text_color(80, 80, 80)
    pdf.ln(10)
    
    # Khối 1: Tổng quan số liệu thống kê
    pdf.set_font("Roboto", "B", size=13)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. Chỉ số thống kê số lượng phản hồi phân lớp:", ln=True)
    
    pdf.set_font("Roboto", "", size=11)
    pdf.cell(0, 8, f"   • Tổng số mẫu văn bản đã xử lý quét hệ thống: {tong_so} bình luận", ln=True)
    pdf.cell(0, 8, f"   • Phân lớp Tích cực (Khen): {tl_tich_cuc}% ({so_tich_cuc} câu)", ln=True)
    pdf.cell(0, 8, f"   • Phân lớp Trung tính (Lấp lửng): {tl_trung_tinh}% ({so_trung_tinh} câu)", ln=True)
    pdf.cell(0, 8, f"   • Phân lớp Tiêu cực (Chê): {tl_tieu_cuc}% ({so_tieu_cuc} câu)", ln=True)
    pdf.ln(8)
    
    # Khối 2: Top từ khóa khen nhiều nhất
    pdf.set_font("Roboto", "B", size=13)
    pdf.cell(0, 10, "2. Top 5 cấu trúc cụm khía cạnh mang sắc thái TÍCH CỰC:", ln=True)
    pdf.set_font("Roboto", "", size=11)
    if top_khen:
        for idx, k in enumerate(top_khen, 1):
            pdf.cell(0, 8, f"     {idx}. {k[0].upper()} (Xuất hiện: {k[1]} lần)", ln=True)
    else:
        pdf.cell(0, 8, "     (Không phát hiện cấu trúc đặc trưng vượt ngưỡng tần suất)", ln=True)
    pdf.ln(6)
    
    # Khối 3: Top từ khóa chê nhiều nhất
    pdf.set_font("Roboto", "B", size=13)
    pdf.cell(0, 10, "3. Top 5 cấu trúc cụm khía cạnh mang sắc thái TIÊU CỰC:", ln=True)
    pdf.set_font("Roboto", "", size=11)
    if top_che:
        for idx, c in enumerate(top_che, 1):
            pdf.cell(0, 8, f"     {idx}. {c[0].upper()} (Xuất hiện: {c[1]} lần)", ln=True)
    else:
        pdf.cell(0, 8, "     (Không phát hiện cấu trúc đặc trưng vượt ngưỡng tần suất)", ln=True)
        
    return bytes(pdf.output())

# ==========================================
# 3. GIAO DIỆN WEB
# ==========================================
st.title("🚀 Hệ thống Phân tích cảm xúc của bình luận sản phẩm ")
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Phân tích câu đơn", "📊 Phân tích hàng loạt (File)"])

# --- TAB 1: PHÂN TÍCH ĐƠN LẺ ---
with tab1:
    user_input = st.text_area("Nhập bình luận sản phẩm vào đây:", placeholder="...")
    if st.button("Phân tích ngay"):
        if user_input:
            with st.spinner("Đang phân tích..."):
                clean_text = tien_xu_ly(user_input)
                vec = vectorizer.transform([clean_text])
                prediction = model.predict(vec)[0]
                
                label_map = {0: ("Tiêu cực 😡", "red"), 1: ("Trung tính 😐", "orange"), 2: ("Tích cực 😍", "green")}
                label, color = label_map[prediction]
                
                st.subheader("Kết quả dự đoán:")
                st.markdown(f"<h2 style='color: {color};'>{label}</h2>", unsafe_allow_html=True)
        else:
            st.warning("Vui lòng nhập nội dung!")

# --- TAB 2: PHÂN TÍCH HÀNG LOẠT ---
with tab2:
    uploaded_file = st.file_uploader("Tải file Excel hoặc CSV chứa bình luận", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)
        
        st.write("Dữ liệu đã tải lên (5 dòng đầu):")
        st.dataframe(df_upload.head())
        
        column_to_analyze = st.selectbox("Chọn cột chứa nội dung bình luận:", df_upload.columns)
        
        if st.button("Bắt đầu phân tích file"):
            with st.spinner(f"Đang xử lý {len(df_upload)} bình luận..."):
                df_upload['clean_text_temp'] = df_upload[column_to_analyze].apply(tien_xu_ly)
                X_vec = vectorizer.transform(df_upload['clean_text_temp'])
                preds = model.predict(X_vec)
                
                df_upload['Dự đoán'] = preds
                df_upload['Nhãn'] = df_upload['Dự đoán'].map({0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"})
                
                st.markdown("---")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Thống kê tổng quan")
                    fig = px.pie(df_upload, names='Nhãn', color='Nhãn',
                                 color_discrete_map={'Tích cực':'green', 'Trung tính':'orange', 'Tiêu cực':'red'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Bảng kết quả chi tiết")
                    st.dataframe(df_upload[[column_to_analyze, 'Nhãn']], use_container_width=True)
                
                st.markdown("---")
                st.subheader("🔍 Khách hàng đang Khen/Chê điều gì nhất?")
                
                col_khen, col_che = st.columns([1, 1])
                
                df_tich_cuc = df_upload[df_upload['Nhãn'] == "Tích cực"]
                df_tieu_cuc = df_upload[df_upload['Nhãn'] == "Tiêu cực"]
                
                top_khen = tim_tu_khoa_nhieu_nhat(df_tich_cuc['clean_text_temp'], "Tích cực") if not df_tich_cuc.empty else []
                top_che = tim_tu_khoa_nhieu_nhat(df_tieu_cuc['clean_text_temp'], "Tiêu cực") if not df_tieu_cuc.empty else []
                
                with col_khen:
                    st.markdown("<h4 style='text-align: center; color: green;'>Top 5 Lý do KHEN (Tích cực)</h4>", unsafe_allow_html=True)
                    if top_khen:
                        df_khen = pd.DataFrame(top_khen, columns=['Từ khóa', 'Số lần xuất hiện'])
                        fig_khen = px.bar(df_khen, x='Số lần xuất hiện', y='Từ khóa', orientation='h', color_discrete_sequence=['green'])
                        fig_khen.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_khen, use_container_width=True)
                    else:
                        st.info("Không tìm thấy từ khóa nổi bật.")

                with col_che:
                    st.markdown("<h4 style='text-align: center; color: red;'>Top 5 Lý do CHÊ (Tiêu cực)</h4>", unsafe_allow_html=True)
                    if top_che:
                        df_che = pd.DataFrame(top_che, columns=['Từ khóa', 'Số lần xuất hiện'])
                        fig_che = px.bar(df_che, x='Số lần xuất hiện', y='Từ khóa', orientation='h', color_discrete_sequence=['red'])
                        fig_che.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_che, use_container_width=True)
                    else:
                        st.info("Không tìm thấy từ khóa nổi bật.")

                # ==========================================================
                # KHU VỰC THÊM NÚT XUẤT FILE PDF TRỰC TIẾP
                # ==========================================================
                st.markdown("---")
                st.subheader("📥 Xuất kết quả phân tích")
                
                
                
                # Nút tải trực tiếp file báo cáo PDF mới thêm vào
                if os.path.exists(FONT_REGULAR):
                    pdf_bytes = tao_file_pdf(df_upload, top_khen, top_che)
                    st.download_button(
                        label="📄 Tải trực tiếp file Báo cáo PDF",
                        data=pdf_bytes,
                        file_name="Bao_Cao_Phan_Tich_Shopee.pdf",
                        mime="application/pdf"
                    )