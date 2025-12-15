clear all; close all; clc;
fprintf('EXPERIMENT 3: NARROWBAND FM\n');

%% Helper Functions
function filtered_signal = ideal_lowpass_filter(signal, Fs, cutoff_freq)
    % IDEAL LOWPASS FILTER (ROBUST VERSION)
    % Works for row OR column input signals

    signal = signal(:);        % FORCE column vector
    N = length(signal);

    % Frequency axis (column)
    freq_axis = linspace(-Fs/2, Fs/2, N).';

    % FFT (column)
    signal_freq = fftshift(fft(signal));

    % Ideal LPF mask (column)
    filter_mask = abs(freq_axis) <= cutoff_freq;

    % Apply filter (column .* column)
    filtered_freq = signal_freq .* filter_mask;

    % Back to time domain
    filtered_signal = real(ifft(ifftshift(filtered_freq)));
end

%% ppaths
project_dir = "/MATLAB Drive/FinalAnalog";
addpath(project_dir);     % add helper_functions.m
exp3_dir = fullfile(project_dir,'Experiment_3_NBFM'); % correct PATH for ANY OS

figures_dir = fullfile(exp3_dir,'Figures');
if ~exist(figures_dir,'dir'); mkdir(figures_dir); end

%% step 1: read audio
fprintf('\n step 1: reading audio file \n');

[audio, Fs_original] = audioread(fullfile(project_dir,'eric.unknown'));
% audio: signal sampples
audio = audio(:,1); % 1st col only (convert to mono if stereo)
                    % because FM assumes single signal

fprintf('Audio loaded\n');
fprintf('Original Fs = %d Hz\n',Fs_original);

%% step 2: spectrum of original signal
fprintf('\n step 2: original signal spectrum \n');

N_audio = length(audio); % no. of samples
f_audio = linspace(-Fs_original/2,Fs_original/2,N_audio); % freq axis
AUDIO_F = fftshift(fft(audio)); 

figure;
plot(f_audio/1000,abs(AUDIO_F),'LineWidth',1.2);
xlabel('freq (kHz)'); ylabel('magnitude'); title('original audio spectrum'); grid on;

saveas(gcf,fullfile(figures_dir,'step 1: original_spectrum.png'));

%% step 3: lowpass filter to 4 kHz
fprintf('\n step 3: filter message to 4 kHz \n');

cutoff_freq = 4000;   % 4 kHz message bandwidth
message = ideal_lowpass_filter(audio, Fs_original, cutoff_freq);

fprintf('message filtered to 4 kHz\n');

%% step 4: play filtered message
fprintf('\n step 4: play filtered message \n');

sound(message, Fs_original);
pause(3.5);

%% step 5: resample for FM
fprintf('\n step 5: resampling for high-frequency carrier \n');

Fc = 100e3;          % Carrier frequency = 100 kHz
Fs = 5 * Fc;         % Sampling frequency = 500 kHz

[P,Q] = rat(Fs/Fs_original);
message_resampled = resample(message,P,Q);

fprintf('resampled to Fs = %d Hz\n',Fs);

%% step 6: NBFM generation
fprintf('\n step 6: generating NBFM signal \n');

% time vector
t = (0:length(message_resampled)-1)/Fs;

% normalize message
m_norm = message_resampled / max(abs(message_resampled));

% small frequency sensitivity for NBFM
kf = 2*pi*50;   % rad/s per unit amplitude

% integrate message
int_m = (cumsum(m_norm)/Fs).';

% NBFM signal
s_nbfm = cos(2*pi*Fc*t + kf*int_m);

fprintf('NBFM signal generated\n');

%% step 7: NBFM spectrum
fprintf('\n step 7: NBFM spectrum \n');

N = length(s_nbfm);
f = linspace(-Fs/2,Fs/2,N);
S_fm = fftshift(fft(s_nbfm));

figure;
plot(f/1000,abs(S_fm),'LineWidth',1.2);
xlabel('freq (kHz)'); ylabel('magnitude'); title('NBFM spectrum'); grid on;
xlim([90 110]);

saveas(gcf,fullfile(figures_dir,'step 7: nbfm_spectrum.png'));

fprintf('OBSERVATION: spectrum is narrow and centered around Fc\n');

%% step 8: NBFM condition
fprintf('\n step 8: NBFM condition \n');

beta = max(abs(kf*int_m));
fprintf('modulation index beta = %.4f\n',beta);
fprintf('condition for NBFM: beta << 1\n');

%% step 9: demodulation
fprintf(' step 9: NBFM demodulation \n');

% differentiator
diff_signal = diff(s_nbfm);
diff_signal = [diff_signal diff_signal(end)];

% envelope detector
envelope = abs(hilbert(diff_signal));

% remove DC offset
demodulated = envelope - mean(envelope);

fprintf('demodulation completed\n');

%% step 10: lowpass filter
fprintf('\n step 10: lowpass filtering recovered signal \n');

demod_filtered = ideal_lowpass_filter(demodulated, Fs, cutoff_freq);

%% step 11: downsample and play
fprintf('\n step 11: audio playback \n');

received_audio = resample(demod_filtered,Q,P);
received_audio = received_audio / max(abs(received_audio));

play_samples = min(3*Fs_original,length(received_audio));

fprintf('playing recovered NBFM audio...\n');
sound(received_audio(1:play_samples), Fs_original);
pause(3.5);

%% step 12: time domain comparison
fprintf('\n step 12: time domain comparison \n');

min_len = min(length(message),length(received_audio));
t_audio = (0:min_len-1)/Fs_original;

figure;
plot(t_audio,message(1:min_len),'b','LineWidth',1.3); hold on;
plot(t_audio,received_audio(1:min_len),'r--','LineWidth',1.3);
hold off;
legend('original message','recovered message');
xlabel('time (s)'); ylabel('amplitude'); grid on;

saveas(gcf,fullfile(figures_dir,'step 12: time_comparison.png'));