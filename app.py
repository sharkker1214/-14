import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import random

# 設定網頁標題與寬度
st.set_page_config(page_title="UAV 緊急通訊覆蓋系統", layout="wide")

# 1. 載入輕量化 YOLOv8 模型 (使用 cache 避免重複載入)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 2. 定義特徵擷取函數 (針對真實圖片)
def get_people_coordinates(image_path):
    results = model(image_path, conf=0.15) 
    coords = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in [0, 2, 5, 7]: # 人、汽車、公車、卡車
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                coords.append([center_x, center_y])
    return coords

# ================= 新增：虛擬災區生成器 =================
def generate_virtual_disaster(width=1200, height=800, num_hotspots=5, people_per_hotspot=30, scattered=20):
    """
    利用高斯分佈 (Gaussian Distribution) 模擬災民聚集熱點
    """
    coords = []
    # 建立一張全白的畫布來模擬地圖
    img_array = np.ones((height, width, 3), dtype=np.uint8) * 240 
    
    # 畫上一些模擬的街道網格背景
    for i in range(0, width, 100):
        cv2.line(img_array, (i, 0), (i, height), (220, 220, 220), 1)
    for i in range(0, height, 100):
        cv2.line(img_array, (0, i), (width, i), (220, 220, 220), 1)

    # 1. 生成高斯聚集熱點 (模擬避難所/空曠處人群)
    for _ in range(num_hotspots):
        # 隨機決定熱點中心
        cx = random.randint(100, width - 100)
        cy = random.randint(100, height - 100)
        
        # 每個熱點的擴散程度 (標準差)
        spread = random.uniform(20, 60) 
        
        for _ in range(people_per_hotspot):
            # 使用常態分佈產生座標
            x = int(np.random.normal(cx, spread))
            y = int(np.random.normal(cy, spread))
            
            # 確保座標不出界
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            coords.append([x, y])
            # 在畫布上畫出代表災民的小黑點
            cv2.circle(img_array, (x, y), 3, (50, 50, 50), -1)

    # 2. 生成均勻分佈的離散災民 (模擬迷失或落單者)
    for _ in range(scattered):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        coords.append([x, y])
        cv2.circle(img_array, (x, y), 3, (100, 100, 100), -1)

    return coords, img_array
# =========================================================

# 3. 定義分群與通訊覆蓋函數
def calculate_uav_positions(coords, eps=150, min_samples=3):
    if len(coords) == 0:
        return []
        
    X = np.array(coords)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    
    uav_targets = []
    unique_labels = set(labels)
    
    for k in unique_labels:
        if k == -1:
            continue 
            
        class_member_mask = (labels == k)
        cluster_points = X[class_member_mask]
        
        centroid = cluster_points.mean(axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        radius = distances.max() * 1.2 
        
        uav_targets.append({
            'center': (int(centroid[0]), int(centroid[1])),
            'radius': int(radius),
            'people_covered': len(cluster_points)
        })
        
    return uav_targets

# 4. Streamlit 網頁介面設計
st.title("🚁 UAV 緊急通訊動態覆蓋模擬系統")
st.markdown("結合邊緣視覺辨識與高斯拓撲模擬，動態規劃無人機最佳通訊懸停座標。")

# 側邊欄：模式切換與參數設定
with st.sidebar:
    st.header("⚙️ 系統模式與參數")
    
    # 模式切換
    mode = st.radio("選擇測試模式", ["純數學拓撲模擬 (虛擬生成)", "邊緣視覺辨識 (上傳圖片)"])
    st.markdown("---")
    
    # 共用參數
    st.subheader("通訊覆蓋演算法參數 (DBSCAN)")
    eps_val = st.slider("叢集距離閾值 (通訊延展度)", min_value=50, max_value=300, value=120, step=10)
    min_samples_val = st.slider("啟動 UAV 最低熱點人數", min_value=1, max_value=10, value=4)
    
    st.markdown("---")
    # 虛擬模擬專用參數
    if mode == "純數學拓撲模擬 (虛擬生成)":
        st.subheader("虛擬災區生成參數")
        num_hotspots = st.slider("災民聚集熱點數量", 1, 10, 4)
        people_per_hotspot = st.slider("每個熱點平均人數", 10, 100, 30)
        scattered = st.slider("隨機落單災民數量", 0, 100, 20)
        if st.button("🎲 重新生成災區拓撲"):
            pass # 按下按鈕會自動觸發 Streamlit 重新載入畫面

# 主畫面邏輯
col1, col2 = st.columns(2)

if mode == "邊緣視覺辨識 (上傳圖片)":
    uploaded_file = st.file_uploader("請上傳災難現場空拍圖 (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB')) 
        
        with col1:
            st.subheader("📡 原始空拍影像")
            st.image(image, use_container_width=True)
            
        with st.spinner('YOLO 特徵擷取中...'):
            temp_path = "temp.jpg"
            image.convert('RGB').save(temp_path)
            coords = get_people_coordinates(temp_path)
            uav_targets = calculate_uav_positions(coords, eps=eps_val, min_samples=min_samples_val)
else:
    # 虛擬模擬模式
    with col1:
        st.subheader("🗺️ 虛擬災區拓撲圖 (高斯分佈)")
        with st.spinner('生成數學拓撲模型中...'):
            coords, img_array = generate_virtual_disaster(
                num_hotspots=num_hotspots, 
                people_per_hotspot=people_per_hotspot, 
                scattered=scattered
            )
            st.image(img_array, use_container_width=True)
            uav_targets = calculate_uav_positions(coords, eps=eps_val, min_samples=min_samples_val)

# 右半邊：通訊覆蓋繪製 (兩種模式共用這段邏輯)
if (mode == "純數學拓撲模擬 (虛擬生成)") or (mode == "邊緣視覺辨識 (上傳圖片)" and uploaded_file is not None):
    with col2:
        st.subheader("🎯 無人機動態覆蓋分析圖")
        result_img = img_array.copy()
        total_covered = 0
        
        for target in uav_targets:
            center = target['center']
            radius = target['radius']
            total_covered += target['people_covered']
            
            # 畫出綠色通訊覆蓋範圍與紅色中心
            cv2.circle(result_img, center, radius, (0, 200, 0), 3)
            cv2.circle(result_img, center, 8, (255, 0, 0), -1)
            
        st.image(result_img, use_container_width=True)
        
    # 底部數據看板
    st.markdown("---")
    st.subheader("📊 系統效能分析")
    m1, m2, m3 = st.columns(3)
    m1.metric("當前災區總目標數", f"{len(coords)} 個")
    m2.metric("最佳化佈署無人機數量", f"{len(uav_targets)} 架")
    if len(coords) > 0:
        coverage_rate = (total_covered / len(coords)) * 100
        m3.metric("通訊需求覆蓋率", f"{coverage_rate:.1f} %")