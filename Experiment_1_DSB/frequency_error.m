clear all; close all; clc;

%% SETUP PATHS
fprintf('========================================\n');
fprintf('EXPERIMENT 1 - FREQUENCY ERROR ANALYSIS\n');
fprintf('Member 3 - Part A\n');
fprintf('========================================\n\n');

% Navigate to project root
project_dir = fileparts(fileparts(pwd));
addpath(fullfile(project_dir, 'Code'));

% Define directories
results_dir = fullfile(project_dir, 'Results');
figures_dir = fullfile(project_dir, 'Figures', 'Experiment_1_DSB');
audio_dir = fullfile(project_dir, 'Audio_Samples');

%% STEP 1: LOAD DSB-SC RESULTS
fprintf('=== STEP 1: Loading DSB-SC Data from Member 2 ===\n');

dsbsc_path = fullfile(results_dir, 'dsb_sc_results.mat');
filtered_path = fullfile(results_dir, 'filtered_audio.mat');

if ~exist(dsbsc_path, 'file')
    error(['ERROR: dsb_sc_results.mat not found!\n' ...
           'Member 2 must run exp1_member2_dsb_sc.m first!']);
end

load(dsbsc_path);      % dsb_sc, message, carrier, t, Fs, Fc, P, Q
load(filtered_path);   % audio_filtered, Fs_original, cutoff_freq

fprintf('✓ Loaded DSB-SC data\n');
fprintf('  Carrier Frequency: %.3f kHz\n', Fc/1000);
fprintf('  Sampling Frequency: %.0f kHz\n', Fs/1000);
fprintf('  DSB-SC signal length: %d samples\n', length(dsb_sc));

%% STEP 2: DEFINE FREQUENCY ERROR
fprintf('\n=== STEP 2: Defining Frequency Error ===\n');

Fc_correct = Fc;        % 100.000 kHz
Fc_error = 100.1e3;     % 100.100 kHz (0.1 kHz error)

freq_error_Hz = Fc_error - Fc_correct;
freq_error_percent = (freq_error_Hz / Fc_correct) * 100;

fprintf('Frequency Synchronization:\n');
fprintf('  Correct Frequency: %.3f kHz\n', Fc_correct/1000);
fprintf('  Receiver Frequency: %.3f kHz\n', Fc_error/1000);
fprintf('  Frequency Error: %.3f Hz (%.4f%%)\n', freq_error_Hz, freq_error_percent);
fprintf('\nThis simulates imperfect carrier recovery at receiver\n');

%% STEP 3: GENERATE CARRIER WITH FREQUENCY ERROR
fprintf('\n=== STEP 3: Generating Erroneous Carrier ===\n');

% Receiver uses wrong frequency
carrier_error = cos(2*pi*Fc_error*t)';

fprintf('✓ Generated carrier with frequency error\n');
fprintf('  Formula: cos(2π × 100.1kHz × t)\n');

%% STEP 4: THEORETICAL ANALYSIS
fprintf('\n=== STEP 4: Theoretical Analysis ===\n');

fprintf('Mathematical Analysis:\n');
fprintf('When multiplying by carrier with frequency error:\n');
fprintf('  m(t)×cos(2πf₁t) × cos(2πf₂t)\n');
fprintf('  = 0.5×m(t)×[cos(2π(f₁-f₂)t) + cos(2π(f₁+f₂)t)]\n');
fprintf('\n');
fprintf('Where:\n');
fprintf('  f₁ = transmitted carrier = %.3f kHz\n', Fc_correct/1000);
fprintf('  f₂ = receiver carrier = %.3f kHz\n', Fc_error/1000);
fprintf('  f₁-f₂ = %.1f Hz (BEATING frequency)\n', abs(Fc_correct-Fc_error));
fprintf('  f₁+f₂ = %.1f kHz (high freq, filtered out)\n', (Fc_correct+Fc_error)/1000);
fprintf('\n');

beating_freq = abs(Fc_correct - Fc_error);
beating_period = 1 / beating_freq;

fprintf('Beating Characteristics:\n');
fprintf('  Beating Frequency: %.1f Hz\n', beating_freq);
fprintf('  Beating Period: %.4f seconds\n', beating_period);
fprintf('  This creates amplitude modulation at %.1f Hz\n', beating_freq);

%% STEP 5: COHERENT DETECTION WITH FREQUENCY ERROR
fprintf('\n=== STEP 5: Performing Coherent Detection ===\n');

% Use clean signal (no noise) to clearly show beating
fprintf('Using clean DSB-SC signal (no noise) to clearly show effect\n');

% Multiply by erroneous carrier
demodulated_freq_error = dsb_sc .* carrier_error;

fprintf('✓ Multiplication complete\n');

% Lowpass filter
fprintf('Applying lowpass filter (cutoff = %d Hz)...\n', cutoff_freq);

N_demod = length(demodulated_freq_error);
f_demod = linspace(-Fs/2, Fs/2, N_demod);
Demod_freq = fftshift(fft(demodulated_freq_error));

% Create ideal LPF
LPF = zeros(size(Demod_freq));
LPF(abs(f_demod) <= cutoff_freq) = 1;

% Apply filter
Demod_filtered_freq = Demod_freq .* LPF;
received_freq_error = real(ifft(ifftshift(Demod_filtered_freq)));

% Scale by 2
received_freq_error = 2 * received_freq_error;

fprintf('✓ Coherent detection complete\n');

%% STEP 6: PLOT BEATING EFFECT (TIME DOMAIN)
fprintf('\n=== STEP 6: Visualizing Beating Effect ===\n');

% Plot duration to show several beats
duration_to_plot = 5 * beating_period; % Show 5 complete beats
samples_to_plot = round(duration_to_plot * Fs);
samples_to_plot = min(samples_to_plot, length(t));

figure('Position', [100, 100, 1200, 600]);
t_plot = t(1:samples_to_plot);
plot(t_plot, received_freq_error(1:samples_to_plot), 'r', 'LineWidth', 1.5);
hold on;
plot(t_plot, message(1:samples_to_plot), 'b--', 'LineWidth', 1);
hold off;
title('Coherent Detection with Frequency Error: BEATING EFFECT', ...
      'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
legend('Received (with beating)', 'Original Message', 'Location', 'best');
grid on;

% Add annotation
annotation('textbox', [0.15, 0.75, 0.3, 0.15], ...
    'String', {sprintf('Beating Frequency: %.1f Hz', beating_freq), ...
               sprintf('Period: %.3f s', beating_period), ...
               'Amplitude varies periodically'}, ...
    'BackgroundColor', 'yellow', 'FontSize', 10, 'FontWeight', 'bold');

saveas(gcf, fullfile(figures_dir, 'exp1_15_frequency_error_beating.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_15_frequency_error_beating.fig'));
fprintf('✓ Saved: exp1_15_frequency_error_beating.png\n');

%% STEP 7: ZOOMED VIEW OF ONE BEAT CYCLE
fprintf('\n=== STEP 7: Zoomed View of Beating ===\n');

% Show 1.5 beat cycles
samples_one_beat = round(1.5 * beating_period * Fs);
samples_one_beat = min(samples_one_beat, length(t));

figure('Position', [100, 100, 1200, 600]);
t_zoom = t(1:samples_one_beat);
plot(t_zoom, received_freq_error(1:samples_one_beat), 'r', 'LineWidth', 2);
hold on;
plot(t_zoom, message(1:samples_one_beat), 'b--', 'LineWidth', 1.5);

% Calculate and plot envelope of beating
envelope_beating = abs(hilbert(received_freq_error(1:samples_one_beat)));
plot(t_zoom, envelope_beating, 'g:', 'LineWidth', 2.5);
plot(t_zoom, -envelope_beating, 'g:', 'LineWidth', 2.5);
hold off;

title('Beating Effect (Zoomed - One Complete Cycle)', ...
      'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
legend('Received (with error)', 'Original', 'Beat Envelope', 'Location', 'best');
grid on;

saveas(gcf, fullfile(figures_dir, 'exp1_16_frequency_error_zoomed.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_16_frequency_error_zoomed.fig'));
fprintf('✓ Saved: exp1_16_frequency_error_zoomed.png\n');

%% STEP 8: FREQUENCY DOMAIN ANALYSIS
fprintf('\n=== STEP 8: Frequency Domain Analysis ===\n');

[f_rec, Rec_freq] = get_spectrum(received_freq_error, Fs);

figure('Position', [100, 100, 1000, 500]);
plot(f_rec/1000, abs(Rec_freq), 'r', 'LineWidth', 1.5);
hold on;
% Mark the beating frequency
[~, beat_idx] = min(abs(f_rec - beating_freq));
beat_mag = abs(Rec_freq(beat_idx));
plot([beating_freq/1000, beating_freq/1000], [0, beat_mag*1.2], 'b--', 'LineWidth', 2);
text(beating_freq/1000+0.02, beat_mag*1.1, ...
     sprintf('← Beating\n   %.1f Hz', beating_freq), 'FontSize', 10);
hold off;

title('Received Signal Spectrum (with Frequency Error)', ...
      'FontSize', 14, 'FontWeight', 'bold');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude', 'FontSize', 12);
grid on;
xlim([0, 5]);

saveas(gcf, fullfile(figures_dir, 'exp1_17_frequency_error_spectrum.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_17_frequency_error_spectrum.fig'));
fprintf('✓ Saved: exp1_17_frequency_error_spectrum.png\n');

fprintf('\nSpectrum shows:\n');
fprintf('  - Message spectrum centered at baseband\n');
fprintf('  - Additional component at %.1f Hz (beating)\n', beating_freq);
fprintf('  - This beating modulates the message amplitude\n');

%% STEP 9: DOWNSAMPLE AND PLAY AUDIO
fprintf('\n=== STEP 9: Audio Demonstration ===\n');

% Downsample
received_freq_error_audio = resample(received_freq_error, Q, P);
received_freq_error_audio = received_freq_error_audio / max(abs(received_freq_error_audio));

fprintf('✓ Downsampled to %d Hz\n', Fs_original);

% Play audio
play_samples = min(5*Fs_original, length(received_freq_error_audio)); % 5 seconds
fprintf('\nPlaying audio with frequency error (%.1f Hz beating)...\n', beating_freq);
fprintf('Listen for:\n');
fprintf('  - Periodic "wobbling" or "tremolo" effect\n');
fprintf('  - Volume oscillating at %.1f Hz\n', beating_freq);
fprintf('  - Message still intelligible but distorted\n\n');

sound(received_freq_error_audio(1:play_samples), Fs_original);
pause(5.5);

% Save audio
audiowrite(fullfile(audio_dir, 'dsb_sc_freq_error_beating.wav'), ...
           received_freq_error_audio, Fs_original);
fprintf('✓ Saved: dsb_sc_freq_error_beating.wav\n');

%% STEP 10: ERROR CALCULATION
fprintf('\n=== STEP 10: Error Analysis ===\n');

min_len = min(length(audio_filtered), length(received_freq_error_audio));
error_signal = audio_filtered(1:min_len) - received_freq_error_audio(1:min_len);

mse_freq = mean(error_signal.^2);
signal_power = mean(audio_filtered(1:min_len).^2);
normalized_error = mse_freq / signal_power;
SER_dB = 10*log10(signal_power / mse_freq);

fprintf('Error Metrics:\n');
fprintf('  Mean Squared Error: %.6f\n', mse_freq);

%% HELPER FUNCTION
function [freq_axis, spectrum] = get_spectrum(signal, Fs)
    N = length(signal);
    freq_axis = linspace(-Fs/2, Fs/2, N);
    spectrum = fftshift(fft(signal));
end