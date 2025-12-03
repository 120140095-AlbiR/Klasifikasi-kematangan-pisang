import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================== CONFIG & UTILS ====================
st.set_page_config(page_title="Klasifikasi Kematangan Pisang", layout="centered")

def gray_world(img_bgr):
    """Gray World White Balance (Sama persis dengan source)"""
    img = img_bgr.astype(np.float32)
    mean_b = img[:, :, 0].mean()
    mean_g = img[:, :, 1].mean()
    mean_r = img[:, :, 2].mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    scale_b = mean_gray / (mean_b + 1e-8)
    scale_g = mean_gray / (mean_g + 1e-8)
    scale_r = mean_gray / (mean_r + 1e-8)

    img[:, :, 0] *= scale_b
    img[:, :, 1] *= scale_g
    img[:, :, 2] *= scale_r

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def show_img_streamlit(img, title=None):
    """
    Helper untuk menampilkan gambar di Streamlit,
    menggantikan fungsi show_img() matplotlib di script asli.
    """
    if title:
        st.caption(f"**{title}**")
    
    # Handle Grayscale vs BGR
    if len(img.shape) == 3:
        # Convert BGR to RGB for display
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_display, use_container_width=True)
    else:
        # Grayscale
        st.image(img, use_container_width=True, channels='GRAY')

def create_color_sample(hue_std, bstar_std, size=(100, 100)):
    """
    Fungsi generate sample warna (Sama persis dengan source)
    """
    hue_cv = int(hue_std / 2)

    # Atur Saturation dan Value berdasarkan kategori
    if hue_std > 95:  # Mentah
        sat = 70; val = 70
    elif hue_std < 87:  # Matang Penuh
        sat = 85; val = 95
    else:  # Setengah Matang
        sat = 75; val = 85

    hsv_img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    hsv_img[:, :, 0] = hue_cv
    hsv_img[:, :, 1] = int(sat * 2.55)
    hsv_img[:, :, 2] = int(val * 2.55)

    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return bgr_img

# ==================== MAIN APP ====================

st.title("🍌 Klasifikasi Kematangan Pisang")
st.markdown("*Web UI Sistem Deteksi Kematangan Pisang menggunakan Rule-Based Classification dan Gray World White Balance*")
st.divider()

# 1. Upload File (Menggantikan hardcoded path)
uploaded_file = st.file_uploader("Upload Gambar Pisang Mentah/Setengah Matang/Matang", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # ==================== AKUISISI GAMBAR ====================
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img0 = cv2.imdecode(file_bytes, 1)
    
    st.text(f"Gambar yang dipakai: {uploaded_file.name}")
    show_img_streamlit(img0, 'Gambar Asli')

    # ==================== PRA-PEMROSESAN ====================
    
    # 3.4.1 Resize
    st.markdown("### Tahap Resize")
    target_size = 640
    resized = cv2.resize(img0, (target_size, target_size), interpolation=cv2.INTER_AREA)
    show_img_streamlit(resized, f'Hasil Resize ke {target_size}x{target_size}')
    st.text(f"Ukuran gambar setelah resize: {resized.shape}")

    # 3.4.2 Koreksi Kecerahan
    st.markdown("### Tahap Koreksi Kecerahan")
    gw = gray_world(resized)
    show_img_streamlit(gw, 'Setelah Gray World White Balance')

    # 3.4.3 Filtering
    st.markdown("### Tahap Filtering")
    blur = cv2.GaussianBlur(gw, (5, 5), 0)
    show_img_streamlit(blur, 'Setelah Gaussian Blur (5x5)')

    # ==================== KONVERSI RUANG WARNA ====================
    st.markdown("### Tahap Konversi Ruang Warna")
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)

    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    c1, c2 = st.columns(2)
    with c1: show_img_streamlit(hsv, 'Gambar dalam Ruang Warna HSV')
    with c2: show_img_streamlit(lab, 'Gambar dalam Ruang Warna Lab')
    
    c3, c4, c5 = st.columns(3)
    with c3: show_img_streamlit(H, 'Channel Hue (H)')
    with c4: show_img_streamlit(S, 'Channel Saturation (S)')
    with c5: show_img_streamlit(B, 'Channel b* (Lab)')

    # ==================== SEGMENTASI OBJEK BUAH ====================
    st.markdown("### Tahap Segmentasi HSV Thresholding")

    lower_hsv = np.array([15, 40, 40])
    upper_hsv = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    show_img_streamlit(mask, 'Mask Awal (HSV Thresholding)')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    show_img_streamlit(mask_closed, 'Mask Setelah Morphological Closing')

    segmented = cv2.bitwise_and(blur, blur, mask=mask_closed)
    show_img_streamlit(segmented, 'Hasil Segmentasi (Objek Buah Saja)')

    # ==================== ANALISIS WARNA ====================
    st.markdown("### Tahap Analisis Warna")

    mask_bool = mask_closed.astype(bool)

    if mask_bool.any():
        mean_hue = float(H[mask_bool].mean())
        std_hue = float(H[mask_bool].std())
        mean_bstar = float(B[mask_bool].mean())
        std_bstar = float(B[mask_bool].std())
        pixel_count = np.sum(mask_bool)
    else:
        st.warning("PERINGATAN: Tidak ada piksel yang tersegmentasi!")
        mean_hue, std_hue, mean_bstar, std_bstar, pixel_count = 0, 0, 0, 0, 0

    st.code(f"""
Jumlah piksel area buah: {pixel_count}
Mean Hue (area tersegmentasi): {mean_hue:.2f}° (std: {std_hue:.2f})
Mean b* (area tersegmentasi): {mean_bstar:.2f} (std: {std_bstar:.2f})
    """)

    # Histogram
    if mask_bool.any():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(H[mask_bool].flatten(), bins=50, color='orange', alpha=0.7)
        axes[0].axvline(mean_hue, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_hue:.1f}')
        axes[0].set_xlabel('Hue Value (OpenCV 0-180)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribusi Hue pada Area Buah')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].hist(B[mask_bool].flatten(), bins=50, color='blue', alpha=0.7)
        axes[1].axvline(mean_bstar, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_bstar:.1f}')
        axes[1].set_xlabel('b* Value (OpenCV 0-255)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribusi b* pada Area Buah')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

    # ==================== KLASIFIKASI KEMATANGAN ====================
    st.markdown("### Tahap Klasifikasi Kematangan Menurut Input Anda")

    hue_standar = mean_hue * 2 
    bstar_standar = (mean_bstar / 255) * 100 

    st.write(f"Mean Hue (skala standar 0-360°): **{hue_standar:.2f}°**")
    st.write(f"Mean b* (skala standar 0-100): **{bstar_standar:.2f}**")

    maturity_status = 'Tidak Diketahui'
    color_desc = 'Tidak Diketahui'

    # --- LOGIKA KLASIFIKASI BARU (Prioritas Hue) ---
    if 74 <= hue_standar <= 120:
        maturity_status = 'Mentah (R2-R3)'
        color_desc = 'Hijau tua (berdasarkan Hue)'

    elif 64 <= hue_standar <= 73.9:
        maturity_status = 'Setengah Matang (R4-R5)'
        color_desc = 'Kuning kehijauan (berdasarkan Hue)'

    elif 30 <= hue_standar <= 63.9:
        maturity_status = 'Matang Penuh (R6-R7)'
        color_desc = 'Kuning cerah (berdasarkan Hue)'

    else:
        # Fallback Logic
        diff_mentah = abs(hue_standar - 110)
        diff_setengah = abs(hue_standar - 82)
        diff_matang = abs(hue_standar - 65)

        min_diff = min(diff_mentah, diff_setengah, diff_matang)
        if min_diff == diff_mentah:
            maturity_status = 'Mentah (R2-R3)'
            color_desc = 'Hijau tua (terdekat berdasarkan Hue)'
        elif min_diff == diff_setengah:
            maturity_status = 'Setengah Matang (R4-R5)'
            color_desc = 'Kuning kehijauan (terdekat berdasarkan Hue)'
        else:
            maturity_status = 'Matang Penuh (R6-R7)'
            color_desc = 'Kuning cerah (terdekat berdasarkan Hue)'

    # --- VALIDASI b* (Catatan) ---
    consistency_msg = ""
    if maturity_status == 'Mentah (R2-R3)' and not (28 <= bstar_standar <= 69.9):
        consistency_msg = "nilai b* tidak konsisten dengan kategori Mentah"
    elif maturity_status == 'Setengah Matang (R4-R5)' and not (70 <= bstar_standar <= 84.9):
        consistency_msg = "nilai b* tidak konsisten dengan kategori Setengah Matang"
    elif maturity_status == 'Matang Penuh (R6-R7)' and not (85 <= bstar_standar <= 120):
        consistency_msg = "nilai b* tidak konsisten dengan kategori Matang Penuh"
    else:
        consistency_msg = "nilai b* konsisten dengan klasifikasi berdasarkan Hue"

    st.info(f"Catatan: b* = {bstar_standar:.2f} — {consistency_msg}")

    st.success(f"### Hasil: {maturity_status}")
    st.write(f"Deskripsi: {color_desc}")

    # ==================== VISUALISASI AKHIR ====================
    st.markdown("### Visualisasi Hasil Akhir + Contoh Warna Pisang Sesuai Input Anda")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Gambar Asli
    axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Gambar Asli')
    axes[0].axis('off')

    # Plot Segmentasi
    axes[1].imshow(mask_closed, cmap='gray')
    axes[1].set_title('Hasil Segmentasi')
    axes[1].axis('off')

    # Plot Klasifikasi
    axes[2].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Hasil: {maturity_status}\n{color_desc}')
    axes[2].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

    # ==================== GENERATE GAMBAR REFERENSI ====================
    st.markdown("### Contoh Warna Pisang Sesuai Input Anda")

    colors_ref = {
        'Mentah (R2-R3)': {'hue': 110.0, 'bstar': 32.0},
        'Setengah Matang (R4-R5)': {'hue': 82.0, 'bstar': 38.0},
        'Matang Penuh (R6-R7)': {'hue': 65.0, 'bstar': 78.0}
    }

    fig_ref, axes_ref = plt.subplots(1, 3, figsize=(15, 5))
    for i, (label, params) in enumerate(colors_ref.items()):
        sample_img = create_color_sample(params['hue'], params['bstar'])
        axes_ref[i].imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
        axes_ref[i].set_title(f'{label}\nHue={params["hue"]:.1f}°\nb*={params["bstar"]:.1f}')
        axes_ref[i].axis('off')

    plt.suptitle("Warna Pisang Sesuai Input Anda\n(Mentah: Hue=110, b*=32 | Setengah: Hue=82, b*=38 | Matang: Hue=65, b*=78)", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig_ref)
