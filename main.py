import pymongo
import pandas as pd
import re
import joblib 
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# 1. KẾT NỐI VÀ LẤY DỮ LIỆU
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["shoppe_sentiment"] 
collection = db["comments"]
cursor = collection.find()
df = pd.DataFrame(list(cursor))

print("1. Đã lấy thành công", len(df), "bình luận!")
print("Các cột dữ liệu đang có là:", df.columns.tolist())

# 2. GÁN NHÃN CẢM XÚC DỰA TRÊN SỐ SAO (RATING)
def gan_nhan(rating):
    if rating <= 2.0:
        return 0
    elif rating == 3.0:
        return 1
    else:
        return 2
    
df['label'] = df['rat'].apply(gan_nhan)

# 3. TIỀN XỬ LÝ VÀ TÁCH TỪ TIẾNG VIỆT
# Tạo từ điển chuẩn hóa Teencode / Viết tắt hay gặp trên Shopee
TEENCODE_DICT = {
    "sp": "sản phẩm", 
    "ko": "không", "k": "không", "khong": "không", "kh": "không", "hok": "không",
    "đc": "được", "dc": "được", "dk": "được",
    "vs": "với", "mik": "mình", "m": "mình",
    "trg": "trong", "nx": "nhận xét", "đg": "đánh giá",
    "cx": "cũng", "auth": "chính hãng", "fake": "giả",
    "rep": "trả lời", "ib": "nhắn tin",
    "tl": "trả lời", "sz": "kích cỡ", "size": "kích cỡ",
    "thik": "thích", "ntn": "như thế này", "chuẫn": "chuẩn"
}

def tien_xu_ly(text):
    text = str(text).lower()
    
    # Xóa dấu câu và ký tự đặc biệt
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # TÁCH TỪ VÀ CHUẨN HÓA VIẾT TẮT
    words = text.split()
    words_clean = []
    for w in words:
        # Nếu từ có trong từ điển teencode thì đổi thành từ chuẩn, nếu không thì giữ nguyên
        if w in TEENCODE_DICT:
            words_clean.append(TEENCODE_DICT[w])
        else:
            words_clean.append(w)
            
    # Ghép lại thành câu
    text = ' '.join(words_clean)
    
    # Cho PyVi nối từ ghép (vd: sản phẩm -> sản_phẩm, không đẹp -> không đẹp)
    text = ViTokenizer.tokenize(text)
    return text

# ÁP DỤNG HÀM LÀM SẠCH 
ten_cot_chu = "com" 
df['clean_text'] = df[ten_cot_chu].apply(tien_xu_ly)

# 4. XEM KẾT QUẢ SAU KHI XỬ LÝ
print("\n--- KẾT QUẢ SAU KHI LÀM SẠCH VÀ GÁN NHÃN ---")
print(df[[ten_cot_chu, 'clean_text', 'label']].head(10))

# 5. CHUẨN BỊ DỮ LIỆU ĐỂ HUẤN LUYỆN 
df = df.dropna(subset=['clean_text', 'label'])
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nSố lượng câu để AI học (Train): {len(X_train)}")
print(f"Số lượng câu để test AI (Test): {len(X_test)}")

# 6. BIẾN CHỮ THÀNH SỐ BẰNG TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),   # Bao gồm Unigram (1), Bigram (2), và Trigram (3)
    min_df=3,             # Lọc bỏ rác: Cụm từ nào xuất hiện dưới 3 lần thì vứt
    max_features=10000    # Giới hạn 10.000 cụm từ có giá trị nhất để nhẹ RAM
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test) 

# # 6. BIẾN CHỮ THÀNH SỐ BẰNG PHO-BERT (DEEP LEARNING)
# print("\nĐang tải 'não bộ' PhoBERT ...")
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
# phobert = AutoModel.from_pretrained("vinai/phobert-base")

# def get_phobert_embeddings(text_series, batch_size=32):

#     text_list = text_series.tolist()
#     all_features = []
    
#     for i in range(0, len(text_list), batch_size):
#         batch = text_list[i : i + batch_size]
        
#         encoded_inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
        
#         with torch.no_grad():
#             outputs = phobert(**encoded_inputs)
#             cls_vectors = outputs.last_hidden_state[:, 0, :]
#             all_features.append(cls_vectors.numpy())
            
#     return np.vstack(all_features)

# print("Đang nhúng (embedding) tập Train bằng PhoBERT... (Sẽ mất khoảng 1-2 phút tùy máy)")
# X_train_vec = get_phobert_embeddings(X_train)

# print("Đang nhúng (embedding) tập Test bằng PhoBERT...")
# X_test_vec = get_phobert_embeddings(X_test)

# print(f"-> Kích thước ma trận Train: {X_train_vec.shape}")

# 7. HUẤN LUYỆN MÔ HÌNH Support Vector Machine VÀ CHẤM ĐIỂM
print("\nĐang huấn luyện mô hình Support Vector Machine ")

model = LinearSVC(class_weight='balanced', random_state=42)
model.fit(X_train_vec, y_train)

# print("\nĐang huấn luyện Naive Bayes...")
# model = MultinomialNB()
# model.fit(X_train_vec, y_train)

# print("\nĐang huấn luyện Logistic Regression...")
# model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
# model.fit(X_train_vec, y_train)
# print("--- ĐÃ HUẤN LUYỆN XONG! ---")

y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n=> ĐỘ CHÍNH XÁC TỔNG THỂ (Accuracy): {acc * 100:.2f}%")

print(f"2. Balanced Accuracy (Chính xác cân bằng): {bal_acc * 100:.2f}%")

print(f"3. Macro F1-Score (Điểm F1 cào bằng)  : {macro_f1 * 100:.2f}%")

print(classification_report(y_test, y_pred))

# === COPY THÊM ĐOẠN NÀY VÀO DƯỚI CÙNG PHẦN 7 ===
print("\nĐang xuất ảnh Ma trận nhầm lẫn (Confusion Matrix)...")

# 1. Tính toán ma trận
cm = confusion_matrix(y_test, y_pred)

# 2. Thiết lập kích thước và vẽ hình
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Tiêu cực', 'Trung tính', 'Tích cực'],
            yticklabels=['Tiêu cực', 'Trung tính', 'Tích cực'])

# 3. Đặt tên các trục cho dễ hiểu
plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Nhãn thực tế (True Label)', fontsize=12)
plt.xlabel('Nhãn dự đoán (Predicted Label)', fontsize=12)

# 4. Lưu trực tiếp ra file ảnh nét căng (dpi=300) để dán vào Word
plt.savefig('confusion_matrix_svm.png', dpi=300, bbox_inches='tight')
print("-> ĐÃ LƯU ẢNH THÀNH CÔNG: Mở thư mục code để xem file 'confusion_matrix_svm.png'")

plt.show() # Nếu bạn muốn lúc chạy code nó hiện cái bảng lên màn hình luôn thì bỏ dấu # ở đầu đi
# ===============================================

# 8. TEST 
def doan_cam_xuc(text):
    clean_text = tien_xu_ly(text)
    vec = vectorizer.transform([clean_text])
    # text_series = pd.Series([clean_text])
    # vec = get_phobert_embeddings(text_series)
    pred = model.predict(vec)
    
    nhan = {0: "Tiêu cực 😡", 1: "Trung tính 😐", 2: "Tích cực 😍"}
    return nhan[pred[0]]

print("\n--- VÍ DỤ: ---")
cau_1 = "sản phẩm rách nát, thái độ shop quá tệ, sẽ không quay lại"
print(f"Câu: '{cau_1}' \n=> AI đoán: {doan_cam_xuc(cau_1)}\n")

cau_2 = "Chất vải rất tệ, không như quảng cáo"
print(f"Câu: '{cau_2}' \n=> AI đoán: {doan_cam_xuc(cau_2)}\n")

cau_3 = "áo rất xinh, mặc tôn dáng, giao hàng thần tốc luôn 10 điểm"
print(f"Câu: '{cau_3}' \n=> AI đoán: {doan_cam_xuc(cau_3)}\n")

cau_4 = "hàng đẹp đúng như mô tả, shop tư vấn rất nhiệt tình"
print(f"Câu: '{cau_4}' \n=> AI đoán: {doan_cam_xuc(cau_4)}\n")

cau_5 = "chất vải tạm ổn, không quá xuất sắc nhưng với giá này thì chấp nhận được"
print(f"Câu: '{cau_5}' \n=> AI đoán: {doan_cam_xuc(cau_5)}\n")

cau_6 = "áo đẹp nhưng hơi mỏng"
print(f"Câu: '{cau_6}' \n=> AI đoán: {doan_cam_xuc(cau_6)}\n")

# 9. LƯU MÔ HÌNH LẠI ĐỂ DÙNG CHO HỆ THỐNG
print("Đang lưu trữ mô hình của AI...")
joblib.dump(model, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("--- HOÀN TẤT! Đã lưu 2 file 'svm_model.pkl' và 'tfidf_vectorizer.pkl' vào thư mục gốc ---")