# Mã nguồn tham khảo: CT144E@NguyenThanhTung

from scipy.signal import windows, firwin, lfilter, freqz, butter, bilinear
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import math
import os
from moviepy import VideoFileClip
import time

# Hàm trích xuất âm thanh từ video MP4
def extract_audio_from_mp4(path_input, audio_format="wav"):             
    if not os.path.exists(path_input):                        # Kiểm tra nếu file video tồn tại
        return      
    clip = VideoFileClip(path_input)                          # Mở file video 
    filename, ext = os.path.splitext(path_input)              # Tách tên file và phần mở rộng
    output_filename = f"{filename}.{audio_format}"            # Tạo tên file đầu ra
    # Chỉ trích xuất nếu file chưa tồn tại      
    if not os.path.exists(output_filename):                   # Kiểm tra nếu file chưa tồn tại
        clip.audio.write_audiofile(output_filename)           # Ghi âm thanh ra file
        print("Âm thanh đã được trích xuất thành công!")

# Hàm tính thông số cửa sổ Kaiser (FIR)
def kaiser_parameters(alpha_pass, alpha_stop, delta_omega):
    delta1 = (10**(alpha_pass / 20) - 1) / (10**(alpha_pass / 20) + 1)      # Biên độ dao động trong dải thông 
    delta2 = 10**(-alpha_stop / 20)                                         # Biên độ dao động trong dải chặn
    alpha = -20 * np.log10(min(delta1, delta2))                             # Tính alpha

    if alpha <= 21:                           
        D = 0.9222                              # Công thức tính D theo alpha
    else:
        D = (alpha - 7.95) / 14.36              # Công thức tính D theo alpha
    
    N = 1 + (D * 2*np.pi) / delta_omega         # Tính bậc N
    N = math.ceil(N)                            # Làm tròn lên bậc N

    if N % 2 == 0:
        N += 1                                  # Đảm bảo N lẻ

    if alpha < 21:
        beta = 0                                # Công thức tính beta theo alpha
    elif alpha <= 50:
        beta = 0.5842 * (alpha - 21)**0.4 + 0.07886 * (alpha - 21)      # Công thức tính beta theo alpha
    else: 
        beta = 0.1102 * (alpha - 8.7)                                   # Công thức tính beta theo alpha
    return N, beta

# Hàm thiết kế bộ lọc tương tự Butterworth (IIR)
def design_iir_manual(f_pass, f_stop, delta1, delta2, Fs, btype='low'):
    # Đổi tần số Hz sang rad/s  
    Omega_pass = np.pi * f_pass / Fs                
    Ohm_pass = 2 * Fs * math.tan(Omega_pass)            # Công thức biến đổi bilinear
    
    Omega_stop = np.pi * f_stop / Fs                   
    Ohm_stop = 2 * Fs * math.tan(Omega_stop)            # Công thức biến đổi bilinear

    # Tính bậc N
    term1 = (1 / delta2**2) - 1
    term2 = (1 / delta1**2) - 1
    
    # Tỷ lệ giữa Lowpass và Highpass
    if btype == 'low':
        ratio = Ohm_stop / Ohm_pass         # Tỷ lệ tần số Lowpass
    else:
        ratio = Ohm_pass / Ohm_stop         # Tỷ lệ tần số Highpass  
        
    N = math.ceil(np.log10(term1 / term2) / (2 * np.log10(ratio)))

    # Tính tần số cắt Analog Ohm_c
    if btype == 'low':
        Ohm_c = Ohm_pass / (term2**(1 / (2 * N)))       # Lowpass
    else:
        Ohm_c = Ohm_pass * (term2**(1 / (2 * N)))       # Highpass

    # Thiết kế bộ lọc Butterworth
    b_analog, a_analog = butter(N, Ohm_c, btype=btype, analog=True)
    b_z, a_z = bilinear(b_analog, a_analog, fs=Fs)
    return b_z, a_z

# Hàm vẽ FFT
def plot_fft(signal, Fs, title, ax):
    ax.magnitude_spectrum(signal, Fs=Fs, scale='linear', color='blue')
    ax.set_title(title)
    ax.set_ylim(0, 25)
    ax.set_ylabel("Biên độ (năng lượng)")
    ax.set_xlabel("Tần số (Hz)")
    ax.grid(True)

# Hàm vẽ đáp ứng biên độ FIR
def plot_fir_response(fir_low, fir_high, Fs):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Đáp ứng biên độ của bộ lọc FIR - Kaiser', fontsize=16)
    
    # Lowpass
    w_low, h_low = freqz(fir_low, worN=8000)
    axs[0].plot((w_low/np.pi)*Fs/2, 20*np.log10(abs(h_low)), color='blue')
    axs[0].set_title("FIR Lowpass")
    axs[0].set_xlabel("Tần số (Hz)")
    axs[0].set_ylabel("Biên độ (dB)")
    axs[0].grid()
    
    # Highpass
    w_high, h_high = freqz(fir_high, worN=8000)
    axs[1].plot((w_high/np.pi)*Fs/2, 20*np.log10(abs(h_high)), color='blue')
    axs[1].set_title("FIR Highpass")
    axs[1].set_xlabel("Tần số (Hz)")
    axs[1].set_ylabel("Biên độ (dB)")
    axs[1].grid()
    
    plt.tight_layout()
    plt.show()

# Hàm vẽ đáp ứng biên độ IIR
def plot_iir_response(b_low_iir, a_low_iir, b_high_iir, a_high_iir, Fs):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Đáp ứng biên độ của bộ lọc IIR - Butterworth', fontsize=16)
    
    # Lowpass
    w_low, h_low = freqz(b_low_iir, a_low_iir, worN=8000)
    axs[0].plot((w_low/np.pi)*Fs/2, abs(h_low), 'b')
    axs[0].set_title("IIR Lowpass")
    axs[0].set_xlabel("Tần số (Hz)")
    axs[0].set_ylabel("Biên độ")
    axs[0].set_xlim([0, Fs/2])
    axs[0].grid(True)
    
    # Highpass
    w_high, h_high = freqz(b_high_iir, a_high_iir, worN=8000)
    axs[1].plot((w_high/np.pi)*Fs/2, abs(h_high), 'b')
    axs[1].set_title("IIR Highpass")
    axs[1].set_xlabel("Tần số (Hz)")
    axs[1].set_ylabel("Biên độ")
    axs[1].set_xlim([0, Fs/2])
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# CHƯƠNG TRÌNH CHÍNH
# ==============================================================================

# Trích âm thanh
path_video = r"D:\Python\DoAnNhom\videogoc.mp4"
path_audio = r"D:\Python\DoAnNhom\videogoc.wav"

extract_audio_from_mp4(path_video)
time.sleep(1)

# Đọc audio
Fs, audio_data = wavfile.read(path_audio)           # Fs: tần số lấy mẫu, audio_data: dữ liệu âm thanh

L_audio = audio_data[:,0].astype(np.float64)        # Kênh trái
R_audio = audio_data[:,1].astype(np.float64)        # Kênh phải

# Lấy dải 500Hz - 2000Hz
fc_high_cut = 2000      # Cắt bỏ tần số > 2000 (Lowpass)
fc_low_cut = 500        # Cắt bỏ tần số < 500 (Highpass)

# ------------------------------------------------------------------------------
# BỘ LỌC FIR
# ------------------------------------------------------------------------------
print("\n--- ĐANG XỬ LÝ FIR ---")
alpha_pass = 1                          # Độ dốc dải thông (dB)
alpha_stop = 80                         # Độ dốc dải chặn (dB)
delta_omega = 2 * np.pi * 500 / Fs      # Độ rộng chuyển tiếp (rad/s)

# Tính tham số
N_fir, beta_fir = kaiser_parameters(alpha_pass, alpha_stop, delta_omega)        # Tính bậc N và beta

# Tạo bộ lọc FIR
fir_low = firwin(N_fir, fc_high_cut, window=('kaiser', beta_fir), fs=Fs)                    # Thông thấp 2000
fir_high = firwin(N_fir, fc_low_cut, window=('kaiser', beta_fir), fs=Fs, pass_zero=False)   # Thông cao 500

# Lọc tín hiệu FIR
# Kênh L
L_FIR_temp = lfilter(fir_low, [1.0], L_audio)           # Qua Lowpass
L_FIR = lfilter(fir_high, [1.0], L_FIR_temp)            # Qua Highpass
# Kênh R
R_FIR_temp = lfilter(fir_low, [1.0], R_audio)           # Qua Lowpass
R_FIR = lfilter(fir_high, [1.0], R_FIR_temp)            # Qua Highpass

# Vẽ đáp ứng biên độ FIR
print("\nĐang vẽ đồ thị Đáp ứng biên độ IIR...")
plot_fir_response(fir_low, fir_high, Fs)   

# ------------------------------------------------------------------------------
# BỘ LỌC IIR
# ------------------------------------------------------------------------------
print("\n--- ĐANG XỬ LÝ IIR ---")
delta1_iir = 0.89125        # Sai số dải thông
delta2_iir = 0.17783        # Sai số dải chặn

# Tạo bộ lọc IIR Thông thấp (Lowpass) để cắt tần số > 2000Hz
# Pass: 1700, Stop: 2000
b_low_iir, a_low_iir = design_iir_manual(fc_high_cut - 300, fc_high_cut, delta1_iir, delta2_iir, Fs, 'low')

# Tạo bộ lọc IIR Thông cao (Highpass) để cắt tần số < 500Hz
# Pass: 500, Stop:300 
b_high_iir, a_high_iir = design_iir_manual(fc_low_cut, fc_low_cut - 200, delta1_iir, delta2_iir, Fs, 'high')

# Lọc tín hiệu IIR
# Kênh L
L_IIR_temp = lfilter(b_low_iir, a_low_iir, L_audio)       # Qua Lowpass
L_IIR = lfilter(b_high_iir, a_high_iir, L_IIR_temp)       # Qua Highpass
# Kênh R
R_IIR_temp = lfilter(b_low_iir, a_low_iir, R_audio)       # Qua Lowpass
R_IIR = lfilter(b_high_iir, a_high_iir, R_IIR_temp)       # Qua Highpass

# Vẽ đáp ứng biên đọo IIR
print("\nĐang vẽ đồ thị Đáp ứng biên độ IIR...")
plot_iir_response(b_low_iir, a_low_iir, b_high_iir, a_high_iir, Fs)

# ------------------------------------------------------------------------------
# VẼ ĐỒ THỊ FFT
# ------------------------------------------------------------------------------
print("\nĐang vẽ đồ thị FFT...")
fig, axs = plt.subplots(2, 3, figsize=(16, 8))

# Trái
plot_fft(L_audio, Fs, "Kênh L - GỐC", axs[0, 0])
plot_fft(L_FIR, Fs, "Kênh L - FIR (Kaiser)", axs[0, 1])
plot_fft(L_IIR, Fs, "Kênh L - IIR (Butterworth)", axs[0, 2])

# Phải
plot_fft(R_audio, Fs, "Kênh R - GỐC", axs[1, 0])
plot_fft(R_FIR, Fs, "Kênh R - FIR (Kaiser)", axs[1, 1])
plot_fft(R_IIR, Fs, "Kênh R - IIR (Butterworth)", axs[1, 2])

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# LƯU KẾT QUẢ
# ------------------------------------------------------------------------------

# Lưu file FIR
output_fir = np.column_stack((L_FIR, R_FIR)).astype(np.int16)
wavfile.write(r"D:\Python\DoAnNhom\amthanh_daloc_FIR.wav", Fs, output_fir)
print("Đã lưu file FIR: amthanh_daloc_FIR.wav")

# Lưu file IIR
output_iir = np.column_stack((L_IIR, R_IIR)).astype(np.int16)
wavfile.write(r"D:\Python\DoAnNhom\amthanh_daloc_IIR.wav", Fs, output_iir)
print("Đã lưu file IIR: amthanh_daloc_IIR.wav")

print("\nCHƯƠNG TRÌNH HOÀN THÀNH!")