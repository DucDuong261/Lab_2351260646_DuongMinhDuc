import numpy as np
import librosa
from scipy.signal import lfilter
from pathlib import Path

# ==========================================
# 1. CÁC HÀM TIỀN XỬ LÝ & ĐẶC TRƯNG
# ==========================================
FS = 16000
FRAME_MS, HOP_MS = 25, 10
WIN = int(FS * FRAME_MS / 1000)
HOP = int(FS * HOP_MS / 1000)

def load_audio(path):
    y, sr = librosa.load(path, sr=FS, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y

def trim_endpoint(y, top_db=25, margin_ms=50):
    frames = librosa.util.frame(y, frame_length=WIN, hop_length=HOP).T
    energy_db = 10 * np.log10(np.sum(frames**2, axis=1) + 1e-9)
    threshold = np.max(energy_db) - top_db
    speech = np.where(energy_db > threshold)[0]
    
    if len(speech) == 0: return y
    m = int(FS * margin_ms / 1000)
    s = max(0, speech[0] * HOP - m)
    e = min(len(y), speech[-1] * HOP + WIN + m)
    return y[s:e]

def mfcc_feature(y):
    y_filt = lfilter([1.0, -0.97], [1.0], y)
    M = librosa.feature.mfcc(
        y=y_filt, sr=FS, n_mfcc=13, n_mels=24,
        n_fft=512, win_length=WIN, hop_length=HOP, window='hamming', center=False
    )
    M = M - np.mean(M, axis=1, keepdims=True)
    return M.T

def dtw_distance(X, Y):
    N, M = len(X), len(Y)
    D = np.full((N+1, M+1), np.inf)
    D[0, 0] = 0.0
    back = np.zeros((N+1, M+1, 2), dtype=int)
    
    for i in range(1, N+1):
        for j in range(1, M+1):
            local = np.linalg.norm(X[i-1] - Y[j-1])
            best, pi, pj = min([(D[i-1,j], i-1, j), (D[i,j-1], i, j-1), (D[i-1,j-1], i-1, j-1)], key=lambda z: z[0])
            D[i, j] = local + best
            back[i, j] = (pi, pj)
            
    path_len = 0
    i, j = N, M
    while i > 0 or j > 0:
        path_len += 1
        i, j = back[i, j]
    return D[N, M] / max(path_len, 1)

# ==========================================
# 2. THỰC THI QUÉT THƯ MỤC VÀ ĐỐI SÁNH
# ==========================================
if __name__ == '__main__':
    # Đường dẫn đến thư mục chứa 15 file
    TEST_DIR = Path('dataset/test')
    
    # Lấy danh sách tất cả các file .wav, sắp xếp theo thứ tự (mau_01 -> mau_15)
    wav_files = sorted(TEST_DIR.glob('*.wav'))
    
    if len(wav_files) < 2:
        print("Không tìm thấy đủ số lượng file .wav trong dataset/test để so sánh!")
    else:
        # Lấy file đầu tiên làm Template
        template_file = wav_files[0]
        print(f"[+] Đang nạp Template gốc từ file: {template_file.name}")
        y_ref = load_audio(template_file)
        mfcc_ref = mfcc_feature(trim_endpoint(y_ref))
        
        print("\n[+] Bắt đầu so sánh các đoạn còn lại với Template gốc:")
        
        # Danh sách lưu kết quả
        results = []
        
        # Duyệt qua các file còn lại (từ mau_02 trở đi)
        for test_file in wav_files[1:]:
            y_test = load_audio(test_file)
            mfcc_test = mfcc_feature(trim_endpoint(y_test))
            
            # Tính khoảng cách DTW
            score = dtw_distance(mfcc_ref, mfcc_test)
            results.append((test_file.name, score))
            
            print(f" -> {test_file.name} : Khoảng cách DTW = {score:.4f}")
            
        # Tìm ra đoạn có âm thanh giống đoạn 1 nhất (điểm DTW nhỏ nhất)
        best_match = min(results, key=lambda x: x[1])
        print(f"\n====> KẾT LUẬN: Đoạn âm thanh giống với {template_file.name} nhất là {best_match[0]} (Điểm: {best_match[1]:.4f})")