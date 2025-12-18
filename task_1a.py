import os
from utils.util import *
init_figure_setup()

N0 = 2 
W = 20 #BW [-W,W]
Th = 0.25 #duration of the impulse response
Fs = 1000 
Ts = 1/Fs 

# x(t) is wgn with zero mean and Rxx(t)=N0*δ(t), in [0, 1]
t_x = iarange(0, 1, Ts)
sigma_x = sqrt(N0 / Ts) # !! discrete variance = N0 / Ts !! 
x = sigma_x * randn(len(t_x)) # randn() returns a random signal with mean 0 and variance 1

# h(t)
# if H(f)=Π(f/2W), then h(t)=2Wsinc(2Wt)
# duration [-Th/2, Th/2]
t_h = iarange(-Th/2, Th/2, Ts)
h = 2*W*sinc(2*W*t_h)

# y(t), approximating continuous convolution so scaled by Ts
y = conv(x, h) * Ts
t_y = convxaxis(t_x, t_h)

# ffts, scaled by Ts for continuous approx
X_f = fftshift(fft(x)) * Ts 
f_x = fftxaxis(len(x), Ts)

H_f = fftshift(fft(h)) * Ts
f_h = fftxaxis(len(h), Ts)

Y_f = fftshift(fft(y)) * Ts
f_y = fftxaxis(len(y), Ts)

figure()

# 1st row - time domain
subplot(2, 3, 1)
plot(t_x, x)
title('input x(t)')
xlabel('time (s)')
ylabel('amplitude')

subplot(2, 3, 2)
plot(t_h, h)
title('impulse response h(t)')
xlabel('time (s)')

subplot(2, 3, 3)
plot(t_y, y)
title('output y(t)')
xlabel('time (s)')

# 2nd row - freq domain (magnitude)
subplot(2, 3, 4)
plot(f_x, abs(X_f))
title('|X(f)|')
xlabel('freq (Hz)')

subplot(2, 3, 5)
plot(f_h, abs(H_f))
title('|H(f)|')
xlabel('freq (Hz)')
xlim(-3*W, 3*W)

subplot(2, 3, 6)
plot(f_y, abs(Y_f))
title('|Y(f)|')
xlabel('freq (Hz)')
xlim(-3*W, 3*W)


if not os.path.exists('plots'):
    os.makedirs('plots')

savefig('plots/task_1a.png')
