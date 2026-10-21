
import numpy as np
import librosa
import matplotlib.pyplot as plt

from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)


# =========================================================
# 0. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

BASE_DIR = Path(r"D:\Bob\TaiLieuHoc\SpokenLanguage\Lab03")

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"

LABELS = ["mo", "dong", "len", "xuong", "dung"]

SAMPLE_RATE = 16000
N_MFCC = 13
N_STATES = 5


# Kiểm tra thư mục dữ liệu
print("=" * 60)
print("THÔNG TIN DỮ LIỆU")
print("=" * 60)
print(f"Thư mục gốc : {BASE_DIR}")
print(f"Train       : {TRAIN_DIR}")
print(f"Test        : {TEST_DIR}")

if not BASE_DIR.exists():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục dự án:\n{BASE_DIR}\n"
        "Hãy sửa lại BASE_DIR cho đúng với máy tính của bạn."
    )

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục train:\n{TRAIN_DIR}"
    )


# =========================================================
# 1. TRÍCH XUẤT ĐẶC TRƯNG MFCC 39 CHIỀU
#    13 MFCC + 13 Delta + 13 Delta-Delta
# =========================================================

def extract_feat(path, sr=SAMPLE_RATE):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file âm thanh: {path}")

    # Đọc âm thanh mono, chuẩn hóa tần số lấy mẫu về 16 kHz
    y, _ = librosa.load(path, sr=sr, mono=True)

    if len(y) == 0:
        raise ValueError(f"File âm thanh rỗng: {path}")

    # Loại bỏ khoảng lặng ở đầu và cuối
    y, _ = librosa.effects.trim(y, top_db=30)

    if len(y) == 0:
        raise ValueError(
            f"Không còn tín hiệu sau khi loại bỏ khoảng lặng: {path}"
        )

    # Pre-emphasis
    y = librosa.effects.preemphasis(y, coef=0.97)

    # MFCC: 13 hệ số
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=512,
        win_length=int(0.025 * sr),
        hop_length=int(0.010 * sr),
        window="hamming",
        n_mels=26,
        fmin=20,
        fmax=sr / 2,
    )

    # Delta và Delta-Delta
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    # Chuyển thành ma trận (số frame, 39 đặc trưng)
    F = np.vstack([mfcc, d1, d2]).T

    # Chuẩn hóa CMVN theo từng file âm thanh
    mean = F.mean(axis=0, keepdims=True)
    std = F.std(axis=0, keepdims=True) + 1e-8
    F = (F - mean) / std

    return F


# =========================================================
# 2. KHỞI TẠO GAUSSIAN HMM TRÁI-SANG-PHẢI
# =========================================================

def make_lr_hmm(n_states=N_STATES, n_iter=80):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        tol=1e-3,
        min_covar=1e-3,

        # Chỉ khởi tạo/cập nhật tham số Gaussian.
        # Giữ nguyên startprob và transmat đã thiết lập.
        init_params="mc",
        params="mc",

        random_state=0,
    )

    # Xác suất bắt đầu: luôn bắt đầu ở trạng thái đầu tiên
    start = np.zeros(n_states)
    start[0] = 1.0

    # Ma trận chuyển trạng thái trái-sang-phải
    A = np.zeros((n_states, n_states))

    for i in range(n_states - 1):
        A[i, i] = 0.7
        A[i, i + 1] = 0.3

    # Trạng thái cuối tự chuyển về chính nó
    A[-1, -1] = 1.0

    model.startprob_ = start
    model.transmat_ = A

    return model


# =========================================================
# 3. HUẤN LUYỆN MỘT HMM CHO MỖI TỪ
# =========================================================

models = {}

print("\n" + "=" * 60)
print("HUẤN LUYỆN MÔ HÌNH")
print("=" * 60)

for label in LABELS:
    train_label_dir = TRAIN_DIR / label

    if not train_label_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu của nhãn '{label}':\n"
            f"{train_label_dir}"
        )

    wav_files = sorted(train_label_dir.glob("*.wav"))

    print(f"\nNhãn: {label}")
    print(f"Thư mục: {train_label_dir}")
    print(f"Số file WAV: {len(wav_files)}")

    if not wav_files:
        raise FileNotFoundError(
            f"Không tìm thấy file WAV trong:\n{train_label_dir}\n"
            "Hãy kiểm tra lại cấu trúc dataset/train."
        )

    features = []
    lengths = []

    for wav in wav_files:
        F = extract_feat(wav)

        if len(F) < 3:
            print(f"Bỏ qua file quá ngắn: {wav.name}")
            continue

        features.append(F)
        lengths.append(len(F))

    if not features:
        raise ValueError(
            f"Không trích xuất được đặc trưng cho nhãn '{label}'."
        )

    # Ghép đặc trưng của nhiều câu nói.
    # lengths giúp HMM phân biệt ranh giới giữa các file.
    X = np.vstack(features)

    model = make_lr_hmm()
    model.fit(X, lengths)

    models[label] = model

    print(f"Tổng số frame: {len(X)}")
    print(f"HMM hội tụ: {model.monitor_.converged}")
    print(f"Số vòng lặp: {model.monitor_.iter}")

print("\nĐã huấn luyện xong tất cả các mô hình.")


# =========================================================
# 4. NHẬN DẠNG BẰNG MAXIMUM LIKELIHOOD
# =========================================================

def recognize(path, models):
    F = extract_feat(path)

    # Điểm số log-likelihood của từng mô hình
    scores = {
        label: model.score(F)
        for label, model in models.items()
    }

    # Chọn nhãn có log-likelihood lớn nhất
    predicted_label = max(scores, key=scores.get)

    return predicted_label, scores


# Thử nhận dạng một file riêng lẻ:
# test_file = TEST_DIR / "mo" / "mo_01.wav"
# prediction, scores = recognize(test_file, models)
# print("Kết quả nhận dạng:", prediction)
# print("Điểm số:", scores)


# =========================================================
# 5. ĐÁNH GIÁ VÀ VẼ CONFUSION MATRIX
# =========================================================

def collect_wav_files(data_dir):
    """Thu thập file WAV theo cấu trúc data_dir/label/*.wav."""
    data = []

    for label in LABELS:
        label_dir = data_dir / label

        if not label_dir.exists():
            continue

        for wav in sorted(label_dir.glob("*.wav")):
            data.append((wav, label))

    return data


test_data = collect_wav_files(TEST_DIR)

if test_data:
    evaluation_data = test_data
    print("\nĐánh giá trên TẬP TEST.")
else:
    # Dự phòng nếu chưa tạo dataset/test.
    # Kết quả này là độ chính xác trên train, không phải test.
    evaluation_data = collect_wav_files(TRAIN_DIR)
    print(
        "\nCẢNH BÁO: Không tìm thấy WAV trong dataset/test.\n"
        "Tạm thời đánh giá trên TRAIN; kết quả có thể cao hơn "
        "khả năng nhận dạng trên dữ liệu mới."
    )

if not evaluation_data:
    raise FileNotFoundError(
        "Không có file WAV để đánh giá trong train hoặc test."
    )

true_y = []
pred_y = []

print("\n" + "=" * 60)
print("KẾT QUẢ NHẬN DẠNG")
print("=" * 60)

for wav, true_label in evaluation_data:
    predicted_label, scores = recognize(wav, models)

    true_y.append(true_label)
    pred_y.append(predicted_label)

    print(
        f"{wav.name:20s} | "
        f"Thực tế: {true_label:6s} | "
        f"Dự đoán: {predicted_label}"
    )

accuracy = accuracy_score(true_y, pred_y)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

cm = confusion_matrix(
    true_y,
    pred_y,
    labels=LABELS,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS,
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Gaussian HMM")
plt.tight_layout()
plt.show()


# =========================================================
# 6. VITERBI ALIGNMENT: QUAN SÁT CHUỖI TRẠNG THÁI
# =========================================================

sample_candidates = [
    (wav, label)
    for wav, label in evaluation_data
    if label == "mo"
]

if not sample_candidates:
    train_candidates = collect_wav_files(TRAIN_DIR)
    sample_candidates = [
        (wav, label)
        for wav, label in train_candidates
        if label == "mo"
    ]

if sample_candidates:
    sample_file, sample_label = sample_candidates[0]

    F_sample = extract_feat(sample_file)

    logp, states = models[sample_label].decode(
        F_sample,
        algorithm="viterbi",
    )

    print("\n" + "=" * 60)
    print("VITERBI ALIGNMENT")
    print("=" * 60)
    print(f"File: {sample_file}")
    print(f"Nhãn mô hình: {sample_label}")
    print(f"Log-likelihood: {logp:.4f}")

    plt.figure(figsize=(10, 4))
    plt.step(
        np.arange(len(states)),
        states,
        where="post",
    )
    plt.xlabel("Frame")
    plt.ylabel("HMM State")
    plt.yticks(range(N_STATES))
    plt.title(f"Viterbi Alignment: {sample_file.name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Không có file phù hợp để hiển thị Viterbi alignment.")
