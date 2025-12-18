import os
from utils.util import *
from utils.gen_srrc import gen_srrc 

init_figure_setup()

## params : carrier freq (Hz), symbol period, oversampling factor
fc = 6.0       
T = 1.0          
over = 100      
Ts = T / over    
Fs = 1 / Ts      

#5-FSK
M = 5            
symbols_indices = arange(M) #0,1,2,3,4

# mapping as we suggested in the report : sm = m - 2 : {-2, -1, 0, 1, 2}
s_mapped = symbols_indices - 2

# two cases for ΔF
DeltaF_values = [1/T, 1/(2*T)]
DeltaF_labels = ['1/T', '1/2T']

#### pulse phi(t) constant in [0,T), amplitude normalized for unit energy : 1/sqrt(T)
phi_amp = 1/sqrt(T)
phi = ones(over) * phi_amp 
t_phi = arange(0, T, Ts)

# 1st for loop, for each ΔF
for d_idx, DeltaF in enumerate(DeltaF_values):
    label_DF = DeltaF_labels[d_idx]
    
    # 2nd loop, for each symbol index
    for m in range(M):
        sm = s_mapped[m] #freq multiplier for each symbol
        
        # pulse g_m(t)
        g_m_t = phi*exp(1j*2*pi*sm*DeltaF*t_phi)
        
        # baseband s(t)
        s_baseband = g_m_t
        t_s = t_phi
        
        #upconv, sBP(t)
        s_bp = 2*real(s_baseband*exp(1j*2*pi*fc*t_s))
        
        r_bp = s_bp #channel ideal h=1
        
       
        # unfiltered downconv r(t)
        r_unfiltered = r_bp * exp(-1j*2*pi*fc*t_s)
        
        # matched filters bank
        filter_outputs = []
        filter_time_axes = []
        
        for m_prime in range(M):
            sm_prime = s_mapped[m_prime]
            
            # Generate the pulse for frequency m'
            g_m_prime_t = phi * exp(1j*2*pi*sm_prime*DeltaF*t_phi)
            
            # matched filter g*(-t)
            h_filter = conj(flip(g_m_prime_t))
            
            # y = r * h, scale by Ts for continuous approx
            y_out = conv(r_unfiltered, h_filter) * Ts
            filter_outputs.append(y_out)
            
            t_h = arange(-T + Ts, Ts, Ts) 
            t_y = convxaxis(t_s, t_h)
            filter_time_axes.append(t_y)                                                                    

        
        ### PLOTTING ###
        figure()
        
        suptitle(f'$\Delta F = {label_DF}$, symbol: $m={m}$ ($s_m={sm}$)')
        
        # baseband s(t), sBP(t), r(t)
        subplot(4, 1, 1)
        plot(t_s, real(s_baseband), 'b', label='Real')
        plot(t_s, imag(s_baseband), 'r', label='Imag')
        title('baseband s(t)')
        ylabel('amplitude')
        
        subplot(4, 1, 2)
        plot(t_s, s_bp, 'b')
        title('upconverted $s_{BP}(t)$')
        ylabel('amplitude')
        
        subplot(4, 1, 3)
        plot(t_s, real(r_unfiltered), 'b')
        plot(t_s, imag(r_unfiltered), 'r')
        title('downconverted unfiltered r(t)')
        ylabel('amplitude')
        
        # filters bank outputs
        max_val = 0
        for out in filter_outputs:
            current_max = max(abs(out))
            if current_max > max_val:
                max_val = current_max
        
        for i in range(5):
            subplot(4, 5, 16 + i)
            plot(filter_time_axes[i], real(filter_outputs[i]), 'b')
            plot(filter_time_axes[i], imag(filter_outputs[i]), 'r')
            plot()
            title(f'filter {i} (magn)\n($s_{{m\'}}={s_mapped[i]}$)')
            xlabel('time (s)')
            
            if max_val > 0:
                ylim(0, max_val * 1.1)

        filename = f'plots/task_3g_DF{d_idx+1}_Sym{m}.png'
        if not os.path.exists('plots'):
            os.makedirs('plots')
        savefig(filename)