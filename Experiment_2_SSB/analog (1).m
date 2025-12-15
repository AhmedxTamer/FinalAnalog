clc; clear; close all;

%% =========================================================
% PART 1: Read Audio Signal & Spectrum
% =========================================================

[m, Fs0] = audioread('eric.wav');
m = m(:,1);            % Mono
m = m(:);              % Column vector

t0 = (0:length(m)-1).' / Fs0;

M = fftshift(fft(m));
f0 = linspace(-Fs0/2, Fs0/2, length(M)).';

figure;
plot(f0, abs(M));
title('Original Message Spectrum');
xlabel('Frequency (Hz)'); ylabel('|M(f)|');

%% ================================================
% PART 2: Ideal Low Pass Filter (BW = 4 kHz)
% =========================================================

BW = 4e3;

H = abs(f0) <= BW;                 % Ideal LPF
M_f = M .* H;
m_f = real(ifft(ifftshift(M_f)));
m_f = m_f(:);

sound(m_f, Fs0);

%% =========================================================
% PART 3: DSB-SC Modulation
% =========================================================

Fc = 100e3;
Fs = 5*Fc;

m_r = resample(m_f, Fs, Fs0);
m_r = m_r(:);

t = (0:length(m_r)-1).' / Fs;
carrier = cos(2*pi*Fc*t);

dsb = m_r .* carrier;

DSB = fftshift(fft(dsb));
f = linspace(-Fs/2, Fs/2, length(DSB)).';

figure;
plot(f, abs(DSB));
title('DSB-SC Spectrum');
xlabel('Frequency (Hz)'); ylabel('|DSB(f)|');

%% =========================================================
% PART 4: SSB (LSB) Using Ideal Filter
% =========================================================

Hssb = (f > (Fc-BW)) & (f < Fc);
SSB = DSB .* Hssb;

ssb = real(ifft(ifftshift(SSB)));
ssb = ssb(:);

figure;
plot(f, abs(SSB));
title('SSB-LSB Spectrum');
xlabel('Frequency (Hz)'); ylabel('|SSB(f)|');

%% =========================================================
% PART 5: Coherent Detection (No Noise)
% =========================================================

r = ssb .* carrier;

R = fftshift(fft(r));
Hbb = abs(f) <= BW;

Rbb = R .* Hbb;
m_rec = real(ifft(ifftshift(Rbb)));
m_rec = m_rec(:);

sound(m_rec, Fs);

%% =========================================================
% PART 6: Practical 4th Order Butterworth Filter
% =========================================================

[b,a] = butter(4, BW/(Fs/2));

ssb_b = filter(b,a, ssb);
r_b = ssb_b .* carrier;
m_rec_b = filter(b,a, r_b);

sound(m_rec_b, Fs);

%% =========================================================
% PART 7: Effect of Noise on SSB (No Toolbox)
% =========================================================

SNR = [0 10 30];

signal_power = mean(ssb.^2);

for k = 1:length(SNR)

    SNR_linear = 10^(SNR(k)/10);
    noise_power = signal_power / SNR_linear;

    noise = sqrt(noise_power) * randn(size(ssb));
    ssb_n = ssb + noise;

    r_n = ssb_n .* carrier;

    Rn = fftshift(fft(r_n));
    Rn_bb = Rn .* Hbb;

    m_n = real(ifft(ifftshift(Rn_bb)));
    sound(m_n, Fs);
end

%% =========================================================
% PART 8: SSB-TC and Envelope Detection
% =========================================================

A = 2 * max(abs(m_r));

ssb_tc = (A + m_r) .* carrier;

env = abs(hilbert(ssb_tc)) - A;
env = env(:);

sound(env, Fs);

%%%% all sounds //////

disp('Filtered message (baseband)');
sound(m_f / max(abs(m_f)), Fs0);
pause(length(m_f)/Fs0 + 1);

disp('Recovered SSB - Ideal filter (no noise)');
sound(m_rec / max(abs(m_rec)), Fs);
pause(length(m_rec)/Fs + 1);

disp('Recovered SSB - Butterworth filter');
sound(m_rec_b / max(abs(m_rec_b)), Fs);
pause(length(m_rec_b)/Fs + 1);

disp('Recovered SSB with noise');
sound(m_n / max(abs(m_n)), Fs);
pause(length(m_n)/Fs + 1);

disp('Envelope detected SSB-TC');
sound(env / max(abs(env)), Fs);
