# Xây dựng hệ thống phân tích cảm xúc từ bình luận trên các sàn thương mại điện tử

## Giới thiệu
Dự án xây dựng mô hình phân tích cảm xúc từ các bình luận sản phẩm thương mại điện tử bằng Machine Learning. 
Mục tiêu là phân loại bình luận thành các nhóm:
- Tích cực
- Tiêu cực
- Trung tính

## Chức năng chính
- Làm sạch và tiền xử lý dữ liệu văn bản tiếng Việt
- Gán nhãn dữ liệu cảm xúc
- Huấn luyện mô hình phân loại bằng thuật toán SVM
- Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score

## Công nghệ sử dụng
- Python
- Pandas
- Scikit-learn
- NLP
- TF-IDF
- Streamlit

## Dataset
Dataset gồm các bình luận sản phẩm thương mại điện tử được gán nhãn cảm xúc.

## Kết quả
- Accuracy: 88,2%
- Macro F1-score: 79,54%
- Mô hình SVM cho kết quả tốt trên dữ liệu tiếng Việt


