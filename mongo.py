import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- 1. CẤU HÌNH FILE CSV & GIỚI HẠN ---
CSV_FILENAME = "du_lieu_binh_luan_shopee.csv"
MAX_COMMENTS = 2102 # THÊM GIỚI HẠN DỪNG TẠI 5000 BÌNH LUẬN

# --- 2. CẤU HÌNH SELENIUM ---
print("Đang kết nối trình duyệt Chrome...")
options = webdriver.ChromeOptions()
options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

options.add_argument("--disable-blink-features=AutomationControlled")

# Khởi tạo WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Link sản phẩm bạn cần lấy dữ liệu
URL = "https://shopee.vn/Áo-Sơ-Mi-Nam-Trơn-Tay-Dài-GEN-ALPHA-cổ-Đức-vải-chéo-mềm-mịn-hạn-chế-nhăn-nhiều-màu-dễ-mặc-i.637683645.20767146799"

total_saved = 0 
page = 1

# Mở file CSV ở chế độ ghi ('w') với mã hóa utf-8-sig để Excel không bị lỗi tiếng Việt
with open(CSV_FILENAME, mode='w', encoding='utf-8-sig', newline='') as file:
    # Khai báo các cột (Headers)
    fieldnames = ['username', 'timestamp', 'comment_raw']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # Ghi dòng tiêu đề vào file
    writer.writeheader()

    try:
        print("Đang điều khiển Chrome thật truy cập trang sản phẩm...")
        if URL not in driver.current_url:
            driver.get(URL)
        
        print("Đang cuộn trang để tìm bình luận...")
        for i in range(8):
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(2) 
            
        print("Đang chờ phần bình luận xuất hiện...")
        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@class, 'shopee-product-rating')]"))
        )
        print("Đã load xong phần bình luận. Bắt đầu bóc tách dữ liệu...")
        
        # --- BẬT BỘ LỌC CÓ BÌNH LUẬN ---
        try:
            print("Đang bật bộ lọc 'Có Bình luận' để tối ưu hóa...")
            filter_buttons = driver.find_elements(By.CSS_SELECTOR, ".product-rating-overview__filter")
            for btn in filter_buttons:
                if "bình luận" in btn.text.lower():
                    driver.execute_script("arguments[0].click();", btn)
                    print("-> Đã bật lọc thành công! Chờ 3 giây để Shopee tải lại danh sách...")
                    time.sleep(3)
                    break
        except Exception as e:
            print("Không thể tự động bấm nút lọc, sẽ cào theo danh sách mặc định.")

        # --- VÒNG LẶP CÀO DỮ LIỆU ---
        while True:
            print(f"--- Đang cào dữ liệu trang {page} ---")
            time.sleep(3)
            
            # CẬP NHẬT CHÌA KHÓA MỚI: Dùng class "meQyXP" mà bạn vừa tìm được
            content_elements = driver.find_elements(By.CSS_SELECTOR, ".meQyXP")
            
            print(f" -> Đã tìm thấy {len(content_elements)} nội dung bình luận.")
            
            comments_batch = []
            
            for content in content_elements:
                try:
                    # Lấy text content trực tiếp từ khối nội dung này
                    raw_text = content.get_attribute("textContent")
                    
                    if not raw_text:
                         continue
                         
                    # Làm sạch: Gộp dòng, xóa khoảng trắng thừa
                    clean_text = raw_text.strip().replace('\n', ' | ')
                    
                    # Bỏ qua nếu text quá ngắn
                    if len(clean_text) < 5:
                         continue
                         
                    doc = {
                        "username": "User", # Tạm thời điền User
                        "timestamp": "Time", # Tạm thời điền Time
                        "comment_raw": clean_text
                    }
                    comments_batch.append(doc)
                except Exception as e:
                    pass

            # Ghi trực tiếp mẻ dữ liệu vừa lấy được vào file CSV
            if comments_batch:
                writer.writerows(comments_batch)
                total_saved += len(comments_batch)
                print(f" -> Đã lưu thêm {len(comments_batch)} bình luận vào file CSV. Tổng: {total_saved}/{MAX_COMMENTS}")
            else:
                print(" -> Không tìm thấy bình luận hợp lệ trên trang này.")

            # --- KIỂM TRA ĐIỀU KIỆN DỪNG ---
            if total_saved >= MAX_COMMENTS:
                print(f"\n[THÔNG BÁO] Đã đạt đủ số lượng {MAX_COMMENTS} bình luận. Quá trình cào dữ liệu sẽ dừng lại.")
                break

            # Tìm nút "Trang tiếp theo"
            try:
                next_button = driver.find_element(By.XPATH, "//button[contains(@class, 'shopee-icon-button--right')]")
                
                if not next_button.is_enabled() or "shopee-icon-button--disabled" in next_button.get_attribute("class"):
                    print("Đã đến trang cuối cùng.")
                    break
                    
                driver.execute_script("arguments[0].click();", next_button)
                page += 1
                time.sleep(3) 
                
            except Exception as e:
                print("Không tìm thấy nút Next hoặc đã hết trang.")
                break

    except Exception as e:
        print(f"\n[LỖI] Có lỗi xảy ra trong quá trình cào giao diện: {e}")

    finally:
        print(f"\n[HOÀN THÀNH] Đã lưu tổng cộng {total_saved} bình luận vào file {CSV_FILENAME}.")