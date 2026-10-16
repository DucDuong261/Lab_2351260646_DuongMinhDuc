import re

# ĐẦU VÀO: Đoạn văn bản đã được trích xuất sẵn từ ảnh
raw_text = """
trong tập hợp. Nếu như S biểu hiện cho tập hợp của 3 số 2,3,4 chúng ta có thể liệt kê các yếu tố như sau:
S = {2,3,4}.
Nếu như chúng ta giả định N* là tập hợp toàn bộ số nguyên dương, việc liệt kê tập N* sẽ gặp khó khăn và chúng ta có thể giải thích một cách đơn giản các thành phần trong tập hợp và viết như sau:
N* = {x | x là một số nguyên dương}.
Tập hợp trên có thể được đọc như sau: N* là tập hợp gồm các số nguyên dương.
Hoặc tập hợp có thể được mô tả bằng các ký hiệu toán học như:
J = {x ∈ ℝ | 2 < x < 5}.
Tập hợp này gồm toàn bộ số thực lớn hơn 2 nhưng nhỏ hơn 5.
Trong cách mô tả một tập hợp, dấu ngoặc nhọn ({}) được sử dụng để bao hàm tập hợp và thanh đứng (|) được sử dụng để tách rời ký tự chung của các thành phần từ việc mô tả các thành phần trong tập hợp đó.
Các phần tử trong một tập hợp được xác định bằng ký hiệu ∈ được đọc là thuộc. Với hai tập hợp được đề cập bên trên như S và I chúng ta có thể viết:
2 ∈ S; 3 ∈ S; 8 ∈ I; 9 ∈ I.
Và rõ ràng 8 ∉ S và đọc là 8 không thuộc S.
"""

def document_structure_detection(text):
    """Phát hiện ranh giới câu bằng biểu thức chính quy (dấu chấm, dấu chấm phẩy)"""
    sentences = re.split(r'[.;\n]+', text)
    return [sent.strip() for sent in sentences if sent.strip()]

def text_normalization(text):
    """Chuẩn hóa văn bản (Text Normalization) sang dạng chuẩn SNOR"""
    replacements = {
        "∈": " thuộc ",
        "∉": " không thuộc ",
        "ℝ": " rờ ",
        "N*": " nờ sao ",
        "{": " mở ngoặc nhọn ",
        "}": " đóng ngoặc nhọn ",
        "|": " sao cho ",
        "<": " nhỏ hơn ",
        "=": " bằng ",
        "{}": " mở ngoặc nhọn đóng ngoặc nhọn ",
        "2,3,4": "hai, ba, bốn" # Xử lý cụm số dính liền
    }
    for symbol, word in replacements.items():
        text = text.replace(symbol, word)

    num_dict = {"1": "một", "2": "hai", "3": "ba", "4": "bốn", "5": "năm", "8": "tám", "9": "chín"}
    for digit, word in num_dict.items():
        text = re.sub(rf'\b{digit}\b', word, text)

    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s,]', '', text)
    
    return text

def homograph_disambiguation_mock(sentence):
    """
    Mô phỏng Phân tích từ loại (POS Tagging) để giải quyết từ đồng âm.
    Do không dùng thư viện ngoài nên ở đây ta gắn nhãn mặc định là [WORD]
    """
    words = sentence.split()
    return [(word, "WORD") for word in words if word != ","]

def grapheme_to_phoneme_vietnamese(sentence):
    """Chuyển đổi Chữ sang Âm (G2P/LTS) bằng cách tách thành các âm tiết"""
    clean_text = sentence.replace(",", "")
    return clean_text.split()



print("=== THỰC THI BƯỚC 1: XỬ LÝ VĂN BẢN TTS ===")

sentences = document_structure_detection(raw_text)

all_normalized_sentences = []

for i, sent in enumerate(sentences):
    print(f"\n[Câu {i+1} Gốc]: {sent}")
    normalized_sent = text_normalization(sent)
    print(f"  -> 2. Chuẩn hóa (TN)  : {normalized_sent}")
    
    all_normalized_sentences.append(normalized_sent)

    pos_tagged = homograph_disambiguation_mock(normalized_sent)
    print(f"  -> 3. Gán nhãn (POS)  : {pos_tagged[:5]} ... (hiển thị 5 từ đầu)")
    
    phonemes = grapheme_to_phoneme_vietnamese(normalized_sent)
    print(f"  -> 4. Âm vị (G2P)     : {phonemes}")

print("\n" + "="*50)
print("TOÀN BỘ ĐOẠN VĂN BẢN SAU KHI CHUẨN HÓA:")
print("="*50)
final_text = " ".join(all_normalized_sentences)
print(final_text)