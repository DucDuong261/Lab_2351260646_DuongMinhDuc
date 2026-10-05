import librosa
import soundfile as sf
import os

def split_audio(input_file, output_dir, chunk_duration=2.0, target_sr=16000):
    """
    Cắt một file âm thanh dài thành các đoạn nhỏ bằng nhau.
    """
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Đang tải và chuẩn hóa file: {input_file}...")
    # Tự động chuyển về mono và 16kHz bất kể định dạng gốc
    y, sr = librosa.load(input_file, sr=target_sr, mono=True)
    
    # Tính số lượng mẫu (samples) cho mỗi đoạn cắt
    samples_per_chunk = int(chunk_duration * sr)
    total_chunks = len(y) // samples_per_chunk
    
    print(f"Bắt đầu băm thành {total_chunks} file, mỗi file {chunk_duration} giây...")
    
    for i in range(total_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        
        # Trích xuất đoạn âm thanh
        chunk = y[start:end]
        
        # Đặt tên file tự động theo chuẩn ASCII để không bị lỗi đường dẫn
        out_name = os.path.join(output_dir, f"mau_{i+1:02d}.wav")
        
        # Ghi ra định dạng WAV
        sf.write(out_name, chunk, sr)
        
    print(f"✅ Đã băm xong! Toàn bộ file lưu tại thư mục: '{output_dir}'")

# Sử dụng:
# Thay 'bai_hat_cua_ban.wav' bằng tên file bạn vừa tải lên
split_audio("D:\\Bob\\TaiLieuHoc\\SpokenLanguage\\LyricsSearch\\dataset\\snippets\\_8vekzCF04Q.wav", output_dir="dataset/test", chunk_duration=2.0)