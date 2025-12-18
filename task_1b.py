import os
from utils.util import *

init_figure_setup()

T = 0.01          
over = 100        
Ts = T / over  

N_inst = 1000  # no of different instances of the process
N_sym = 100      
N_bits = 102     

# pulse, normalized, A = 1/sqrt(T)
g_len = over
g = ones(g_len) / sqrt(T)

# axis
L_impulse = N_sym * over
L_conv = L_impulse + g_len - 1
T_sim = L_conv * Ts
f_axis = fftxaxis(L_conv, Ts)

# THEORETICAL PSD : S(f) = sinc^2(fT) * sin^2(2*pi*fT)
S_theory = (sinc(f_axis*T)**2) * (sin(2*pi*f_axis*T)**2)

# EXPERIMENTAL PSD
P_sum = zeros(L_conv)
periodogram_scaler = (Ts**2) / T_sim

for i in range(N_inst):
    # bits
    b = (rand(N_bits) > 0.5) * 2 - 1
    
    # symbols as we did in class, a difference of bits with lag 2
    s_symbols = 0.5 * (b[2:] - b[:-2]) 
    
    # upsample, convolve with pulse, FFT and periodogram
    s_impulse = upsample(s_symbols, over)
    s_t = conv(s_impulse, g)
    S_f = fftshift(fft(s_t))

    P_inst = periodogram_scaler * (abs(S_f)**2)
    P_sum = P_sum + P_inst

S_exp = P_sum / N_inst

# plots
figure()
plot(f_axis, S_exp, label='experimental PSD')
plot(f_axis, S_theory, 'r--', label='theoretical PSD')
title('(1b) : Power Spectral Density')
xlabel('freq (Hz)')
ylabel('PSD')
xlim(-500, 500)
legend()

if not os.path.exists('plots'):
    os.makedirs('plots')
savefig('plots/task_1b.png')