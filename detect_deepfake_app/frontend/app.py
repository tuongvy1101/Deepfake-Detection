import base64
import streamlit as st
import requests
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Cấu hình
BACKEND_URL = "http://localhost:8000/predict"  # Sửa endpoint đúng

# Streamlit cấu hình giao diện
st.set_page_config(layout="wide")
st.title("🧠 Ứng dụng Phát hiện Deepfake")

# Tạo 3 tab
tab1, tab2, tab3 = st.tabs([
    "🔍 Phân tích & Phát hiện Deepfake",
    "🧪 So sánh giữa các phiên bản",
    "📊 Thống kê hiệu suất mô hình"
])

# ========== TAB 1 ==========
with tab1:
    st.header("📌 Phân tích và phát hiện Deepfake")

    uploaded_file = st.file_uploader("Chọn ảnh (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.image(img, caption="Ảnh đã chọn", width=700)

        # Debug thông tin file
        # st.write(f"File name: {uploaded_file.name}")
        # st.write(f"File type: {uploaded_file.type}")
        # st.write(f"File size: {uploaded_file.size} bytes")

        if st.button("Phân tích"):
            with st.spinner("Đang gửi ảnh tới mô hình..."):
                try:
                    # Resize ảnh về 224x224 trước khi gửi
                    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                    img_byte_arr = BytesIO()
                    img_resized.save(img_byte_arr, format='JPEG')
                    img_data = img_byte_arr.getvalue()

                    # Gửi file với content_type rõ ràng
                    content_type = "image/jpeg"
                    res = requests.post(
                        BACKEND_URL,
                        files={"file": (uploaded_file.name, img_data, content_type)}
                    )

                    res.raise_for_status()  # Ném lỗi nếu status không phải 200
                    result = res.json()

                    st.success("✅ Phân tích thành công!")
                    confidence = result['confidence']
                    real_prob = confidence['real']
                    fake_prob = confidence['fake']
                    st.write(f"**Xác suất Real:** {real_prob:.2%}")
                    st.write(f"**Xác suất Fake:** {fake_prob:.2%}")

                    # Biểu đồ với matplotlib
                    fig, ax = plt.subplots(figsize=(6,3))  # Kích thước nhỏ gọn
                    labels = ['Real', 'Fake']  # Nhãn nằm ngang
                    values = [real_prob, fake_prob]
                    colors = ["#6AC268", "#BF3232"]  # Màu xanh nhạt và đỏ
                    ax.bar(labels, values, color=colors)
                    ax.set_ylabel('Xác suất')
                    ax.set_title('Xác suất phân loại Deepfake')
                    ax.set_ylim(0, 1)  # Giới hạn trục y từ 0 đến 1
                    ax.tick_params(axis='x', rotation=0, labelsize=8)  # Nhãn x nằm ngang
                    ax.tick_params(axis='y', labelsize=6)  # Giảm kích thước nhãn trục y
                    ax.grid(False)  # Loại bỏ lưới để giống st.bar_chart
                    ax.spines['top'].set_visible(False)  # Ẩn viền trên
                    ax.spines['right'].set_visible(False)  # Ẩn viền phải
                    plt.tight_layout(pad=0.5)  # Giảm padding để thu nhỏ nền trắng

                    # Giới hạn chiều rộng và căn giữa bằng CSS
                    st.markdown(
                        """
                        <style>
                        .chart-container {
                            display: flex;
                            justify-content: center;
                            max-width: 250px;  /* Giới hạn chiều rộng */
                            margin: 0 auto;
                        }
                        </style>
                        <div class="chart-container">
                        """,
                        unsafe_allow_html=True
                    )
                    st.pyplot(fig)
                    st.markdown("</div>", unsafe_allow_html=True)

                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ Lỗi khi gửi ảnh đến server: {str(e)}")
                    st.write(f"Response: {res.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Lỗi kết nối đến server: {str(e)}")
                except KeyError as e:
                    st.error(f"❌ Dữ liệu trả về không đúng định dạng: {str(e)}")
                    st.write(f"Response: {res.text}")
                except Exception as e:
                    st.error(f"❌ Lỗi không xác định: {str(e)}")

# ========== TAB 2 ==========
with tab2:
    st.header("🔍 So sánh giữa các phiên bản ảnh")
    st.subheader("🧪 So sánh ảnh gốc, deepfake và ảnh thêm nhiễu đối kháng")

    IMAGE_PATH = "images"

    # Hàng 1: Ảnh gốc
    st.markdown("### Ảnh gốc")
    col_left, col_center, col_right = st.columns([2, 1, 2])
    with col_center:
        st.image(os.path.join(IMAGE_PATH, "original.jpg"), width=300)
        st.markdown("<div style='text-align:center; font-size:18px; font-weight:bold'>Original</div>", unsafe_allow_html=True)

    # Hàng 2: 5 ảnh deepfake
    st.markdown("### Ảnh Deepfake")
    df_names = ["neuraltextures.jpg", "faceswap.jpg", "face2face.jpg", "deepfakes.jpg", "genai.jpg"]
    df_captions = ["NeuralTextures", "FaceSwap", "Face2Face", "Deepfakes", "GenAI"]
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(os.path.join(IMAGE_PATH, df_names[i]), width=300)
            st.markdown(f"<div style='text-align:center; font-size:16px; font-weight:bold'>{df_captions[i]}</div>", unsafe_allow_html=True)

    # Hàng 3: 5 ảnh deepfake + adversarial noise
    st.markdown("### Ảnh Deepfake thêm nhiễu đối kháng")
    adv_names = ["neuraltextures_adv.jpg", "faceswap_adv.jpg", "face2face_adv.jpg", "deepfakes_adv.jpg", "genai_adv.jpg"]
    adv_captions = [cap + " + Noise" for cap in df_captions]
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(os.path.join(IMAGE_PATH, adv_names[i]), width=300)
            st.markdown(f"<div style='text-align:center; font-size:16px; font-weight:bold'>{adv_captions[i]}</div>", unsafe_allow_html=True)

with tab3:
    st.subheader("📊 Hiệu suất mô hình phát hiện Deepfake")
    METRIC_PATH = "metrics"

    # --- Kết quả trên FF++ ---
    st.markdown("## Kết quả huấn luyện trên tập FF++")
    col1, col2, col3 = st.columns([1, 2, 1])  # Tỷ lệ 1:2:1
    with col2:  # Cột giữa chứa ROC
        roc_path = os.path.join(METRIC_PATH, "ffpp_roc.png")
        st.image(roc_path, width=800)
        st.markdown("<div style='text-align:center; font-size:20px; font-weight:bold'; padding-bottom: 20px>ROC Curve</div>", unsafe_allow_html=True)

    # Thêm khoảng cách
    st.write("")
    st.write("")
    st.write("")

    col4, col5 = st.columns(2)
    with col4:
        st.image(os.path.join(METRIC_PATH, "ffpp_acc.png"), use_container_width=True)
        st.markdown("<div style='text-align:center; font-size:20px; font-weight:bold'>Train and Validation Accuracy</div>", unsafe_allow_html=True)
    with col5:
        st.image(os.path.join(METRIC_PATH, "ffpp_loss.png"), use_container_width=True)
        st.markdown("<div style='text-align:center; font-size:20px; font-weight:bold'>Train and Validation Loss</div>", unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.markdown("### 📊 Chỉ số trên tập FF++")
    col6, col7, col8 = st.columns(3)
    with col6:
        st.metric(label="Accuracy", value="53.0%")
    with col7:
        st.metric(label="Precision", value="73.8%")
    with col8:
        st.metric(label="Recall", value="37.8%")

    st.markdown("---")

    # --- Kết quả trên tập FF++ + ảnh GenAI + nhiễu ---
    st.markdown("## 🧪 Kết quả trên tập kết hợp (FF++ + ảnh GenAI + nhiễu đối kháng)")
    col9, col10, col11 = st.columns([1, 2, 1])  # Tỷ lệ 1:2:1
    with col10:  # Cột giữa chứa ROC
        roc_path = os.path.join(METRIC_PATH, "fusion_roc.png")
        st.image(roc_path, width=800)
        st.markdown("<div style='text-align:center; font-size:20px; font-weight:bold'; padding-bottom: 20px>ROC Curve</div>", unsafe_allow_html=True)

    # Thêm khoảng cách
    st.write("")
    st.write("")
    st.write("")

    col12, col13 = st.columns(2)
    with col12:
        st.image(os.path.join(METRIC_PATH, "fusion_acc.png"), use_container_width=True)
        st.markdown("<div style='text-align:center; font-size:20px; font-weight:bold'>Train and Validation Accuracy</div>", unsafe_allow_html=True)
    with col13:
        st.image(os.path.join(METRIC_PATH, "fusion_loss.png"), use_container_width=True)
        st.markdown("<div style='text-align:center; font-size:20px; font-weight:bold'>Train and Validation Loss</div>", unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.markdown("### 🧪 Chỉ số trên tập kết hợp (FF++ + GenAI + Noise)")
    col14, col15, col16 = st.columns(3)
    with col14:
        st.metric(label="Accuracy", value="70.0%")
    with col15:
        st.metric(label="Precision", value="73.8%")
    with col16:
        st.metric(label="Recall", value="62.0%")