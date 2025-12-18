import os
from utils.util import *
from utils.gen_srrc import gen_srrc
from scipy.stats import norm


init_figure_setup()

# 800KHz carrier, 80KHz passband BW [-W,W] = [-40,40]KHz
fc = 800**3       
bw_pass = 80**3      
W = bw_pass / 2     

half_dur = 4         
rolloff = 0.35

# deriving symb period T from the BW, as mentioned in report
T = (1 + rolloff) / bw_pass

over = 200          
Ts = T / over
Fs = 1 / Ts         

### 4-QAM : a=1, 10^5 bits, 2 bits per symbol
a=1.0            
Es= 2*(a**2) # E[|X|^2]
N_bits = 10**5     
N_sym =N_bits // 2 

snr_db = arange(0, 14, 2)

## CHANNEL SCENARIOS, have to be changed manually for each of the three groups
# (a0, a1, t1), where t1 is only the multiplication factor, the T hasn't been added yet
scenarios = [
    (0.5, 0.3, 0.2222), #b
    (0.5, 0.3, 0.4444),   #r 
    (0.5, 0.3, 0.963)     #g 
]
colors = ['b', 'r', 'g']

# pulse
phi,t_phi = gen_srrc(T, over, half_dur, rolloff)
phi = phi/sqrt(sum(abs(phi)**2)) #normalize it, make sure Sum(|phi|^2) = 1.

## plots 
fig_spec = figure()
ax_mag = subplot(2, 1, 1)
ax_phs = subplot(2, 1, 2)

fig_ber = figure()
ax_ber = subplot(1, 1, 1)

#for each of the 3 (a0,a1,t1) scenario in the group chosen above:
for s_idx, (a0, a1, tau_T) in enumerate(scenarios):
    color = colors[s_idx]
    
    t1 = tau_T * T # now this is the delay as a decimal multiple of T
    
    ###### H(f) ######
    # asked [-10W, 10W] in assignment
    f_plot = arange(-10*W, 10*W, W/50)
    
    H_f = a0 + a1 * exp(-1j*2*pi*fc*t1) * exp(-1j*2*pi*f_plot*t1)
    
    # magnitude plot
    figure(fig_spec.number)
    subplot(2, 1, 1)

    # highlight the [-W,W] part
    if s_idx == 0: 
        axvspan(-W, W, color='yellow', rolloff=0.3, label='BW')
    
    plot(f_plot, abs(H_f), color=color, label=f'({a0}, {a1}, {tau_T}T)')
    
    # phase plot
    subplot(2, 1, 2)
    if s_idx == 0:
        axvspan(-W, W, color='yellow', rolloff=0.3)
    
    # normalizing phase by pi, to make it in [-1,1]
    plot(f_plot, angle(H_f)/pi, color=color)

    ###### BER #######
    
    # h_hat=H(0)
    h_hat = a0 + a1 * exp(-1j*2*pi*fc*t1)
    h_abs_sq = abs(h_hat)**2
    
    BER_exp = []
    BER_theo = []
    
    #random bit/symbol generation
    # real part +/- a, imag part +/- a
    bits_I = 2 * (rand(N_sym) > 0.5) - 1 # -1 or 1
    bits_Q = 2 * (rand(N_sym) > 0.5) - 1
    syms = a * (bits_I + 1j * bits_Q)
    
    # create baseband signal
    s_impulse = upsample(syms, over)
    s_baseband = conv(s_impulse, phi)
    
    for SNR_val in snr_db:
        # Calculate N0 = Es / SNR
        SNR_lin = 10**(SNR_val / 10)
        N0 = Es / SNR_lin
        
        # 1. APPLY CHANNEL (Baseband Equivalent)
        # y(t) = a0 * s(t) + a1 * s(t - t1) * exp(-j * 2*pi * fc * t1)
        
        # delay in samples
        delay_samples = int(round(t1 / Ts))
        phase_shift = exp(-1j * 2 * pi * fc * t1)
        
        # make delayed copy
        s_delayed = zeros(len(s_baseband), dtype=complex)
        
        if delay_samples == 0:            # zero delay: just copy the signal
            s_delayed = s_baseband
        elif delay_samples < len(s_baseband): #otherwise, normal delay logic
            s_delayed[delay_samples:] = s_baseband[:-delay_samples]
            
        # combine rays
        r_clean = a0 * s_baseband + a1 * s_delayed * phase_shift
        
        ## ADD NOISE
        # noise variance per sample for simulation = N0 / Ts
        # complex noise: Real var = sigma^2/2, Imag var = sigma^2/2
        sigma_noise = sqrt(N0)
        noise = (sigma_noise / sqrt(2)) * (randn(len(r_clean)) + 1j * randn(len(r_clean)))
        
        r_received = r_clean + noise
        
        ### RECEIVER ###
        # matched filter : since phi is real adn symmetric, g*(-t)=g(t)
        y_filtered = conv(r_received, phi)
        
        start_idx = 2 * int(half_dur * over)        
        y_sampled = y_filtered[start_idx : start_idx + N_sym * over : over]
        
        ## DETECTION : decision rule Z = (h_hat*) *  Y
        # So it's Re(Z)>0 =>a & Re(Z)<0 => -a on real branch
        # im(Z)>0 =>a & Im(Z)<0 => -a on imaginary

        Z = y_sampled * conj(h_hat)
        dec_I = 2*(real(Z)>0) -1
        dec_Q = 2*(imag(Z)>0) - 1
        
        #### EXPERIMENTAL BER : count (bit errors)/(total bits),
        # where total bits = 2*N_symbols, and errors = (err in Re) + (err in Im)
        err_I = sum(dec_I != bits_I)
        err_Q = sum(dec_Q != bits_Q)
        ber = (err_I + err_Q) / (2 * N_sym)
        BER_exp.append(ber)
        
        ### THEORETICAL BER = Q(sqrt(|h|^2 * SNR))
        arg = sqrt(h_abs_sq * SNR_lin)
        ber_th_exact = norm.sf(arg)
        BER_theo.append(ber_th_exact)

    # plotting BER curve for the scenario
    figure(fig_ber.number)
    semilogy(snr_db, BER_exp, color + '-', label=f'Exp ({a0},{a1},{tau_T})')
    semilogy(snr_db, BER_theo, color + '--', label=f'Th ({a0},{a1},{tau_T})')

#### PLOTS ####

# Spectrum Figure
figure(fig_spec.number)
subplot(2, 1, 1)
title('$|H(f)|$')
ylabel('$|H(f)|$')
ylim(0, 1)
legend(loc='upper right', fontsize='small')

subplot(2, 1, 2)
title('$arg \; H(f)$ (norm. by $\pi$)')
ylabel('$arg \; H(f)$ / $\pi$')
xlabel('freq (Hz)')
ylim(-1, 1)

if not os.path.exists('plots'):
    os.makedirs('plots')
savefig('plots/task_4g_spectrum1.png')

# BER Figure
figure(fig_ber.number)
title('BER vs SNR$_{T_X}$, 4-QAM, 2-ray channel')
xlabel('SNR$_{T_X}$ (dB)')
ylabel('bit error rate (BER)')
grid(True, which="both", ls="-")
legend()
savefig('plots/task_4g_ber1.png')