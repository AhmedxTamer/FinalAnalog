
clear all; close all; clc;

% Navigate to project root
project_dir = "/MATLAB Drive/FinalAnalog";
addpath(fullfile(project_dir, 'Code'));

% Define directories
results_dir = fullfile(project_dir, 'Results');
figures_dir = fullfile(project_dir, 'Figures', 'Experiment_1_DSB');
audio_dir = fullfile(project_dir, 'Audio_Samples');

%% STEP 1: LOAD FILTERED AUDIO FROM PART A
fprintf('=== STEP 1: Loading Filtered Audio ===\n');

filtered_data_path = fullfile(results_dir, 'filtered_audio.mat');

if ~exist(filtered_data_path, 'file')
    error(['ERROR: filtered_audio.mat not found!\n' ...
           'Please run exp1_member1_audio_preparation.m first!']);
end

load(filtered_data_path);
fprintf('✓ Loaded: filtered_audio.mat\n');
fprintf('  Fs_original: %d Hz\n', Fs_original);
fprintf('  Signal length: %d samples\n', length(audio_filtered));
fprintf('  Cutoff frequency: %d Hz\n', cutoff_freq);

%% STEP 2: MODULATION PARAMETERS
fprintf('\n=== STEP 2: Setting Modulation Parameters ===\n');

Fc = 100e3;  % Carrier frequency = 100 kHz
Fs = 5 * Fc;  % Sampling frequency = 500 kHz (Nyquist requirement)

fprintf('Carrier Frequency (Fc): %.0f kHz\n', Fc/1000);
fprintf('New Sampling Frequency (Fs): %.0f kHz\n', Fs/1000);
fprintf('Nyquist requirement: Fs > 2*Fc → %.0f > %.0f ✓\n', Fs/1000, 2*Fc/1000);

%% STEP 3: RESAMPLE AUDIO TO HIGH SAMPLING FREQUENCY
fprintf('\n=== STEP 3: Resampling Signal ===\n');

fprintf('Old Fs: %d Hz (%.1f kHz)\n', Fs_original, Fs_original/1000);
fprintf('New Fs: %d Hz (%.1f kHz)\n', Fs, Fs/1000);

% Calculate resampling ratio
[P, Q] = rat(Fs / Fs_original);
fprintf('Resampling ratio: P/Q = %d/%d = %.4f\n', P, Q, P/Q);

% Perform resampling
fprintf('Resampling in progress...\n');
message = resample(audio_filtered, P, Q);

fprintf('✓ Resampling complete\n');
fprintf('  Old length: %d samples\n', length(audio_filtered));
fprintf('  New length: %d samples\n', length(message));
fprintf('  New duration: %.4f seconds\n', length(message)/Fs);

%% STEP 4: GENERATE CARRIER SIGNAL
fprintf('\n=== STEP 4: Generating Carrier Signal ===\n');

% Time vector
t = (0:length(message)-1) / Fs;
fprintf('Time vector: 0 to %.4f seconds\n', t(end));

% Carrier: cos(2*pi*Fc*t)
carrier = cos(2*pi*Fc*t)';
fprintf('✓ Carrier generated: cos(2π × %.0f kHz × t)\n', Fc/1000);

%% STEP 5: CALCULATE DC BIAS FOR DSB-TC
fprintf('\n=== STEP 5: Calculating DC Bias ===\n');

% For modulation index μ = 0.5, we need Ac = 2 * max(|m(t)|)
A_c = 2 * max(abs(message));

fprintf('Message statistics:\n');
fprintf('  Max amplitude: %.6f\n', max(abs(message)));
fprintf('  Min amplitude: %.6f\n', min(message));
fprintf('  Mean: %.6f\n', mean(message));
fprintf('\nDSB-TC Parameters:\n');
fprintf('  DC Bias (Ac): %.6f\n', A_c);
fprintf('  Modulation Index (μ): %.3f\n', max(abs(message))/A_c);
fprintf('  Note: μ = 0.5 means 50%% modulation depth\n');

%% STEP 6: GENERATE DSB-TC SIGNAL
fprintf('\n=== STEP 6: Generating DSB-TC Signal ===\n');

% DSB-TC formula: s(t) = [Ac + m(t)] × cos(2πfct)
dsb_tc = (A_c + message) .* carrier;

fprintf('✓ DSB-TC signal generated\n');
fprintf('  Formula: s(t) = [%.4f + m(t)] × cos(2π×100kHz×t)\n', A_c);
fprintf('  Signal statistics:\n');
fprintf('    Max: %.4f\n', max(dsb_tc));
fprintf('    Min: %.4f\n', min(dsb_tc));
fprintf('    Mean: %.6f\n', mean(dsb_tc));

% Verify modulation is correct (envelope should be positive)
envelope_check = A_c + message;
if min(envelope_check) >= 0
    fprintf('  ✓ Envelope is always positive (correct for DSB-TC)\n');
else
    warning('  ⚠ Envelope has negative values! Check modulation index.');
end

%% STEP 7: PLOT DSB-TC SPECTRUM
fprintf('\n=== STEP 7: Plotting DSB-TC Spectrum ===\n');

% Compute spectrum
N_dsb = length(dsb_tc);
f_dsb = linspace(-Fs/2, Fs/2, N_dsb);
DSB_TC_freq = fftshift(fft(dsb_tc));

% Full spectrum view
figure('Position', [100, 100, 1000, 500]);
plot(f_dsb/1000, abs(DSB_TC_freq), 'LineWidth', 1.5);
title('DSB-TC Modulated Signal Spectrum', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude', 'FontSize', 12);
grid on;
xlim([80, 120]);

saveas(gcf, fullfile(figures_dir, 'exp1_04_dsb_tc_spectrum.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_04_dsb_tc_spectrum.fig'));
fprintf('✓ Saved: exp1_04_dsb_tc_spectrum.png\n');

% Zoomed spectrum around carrier
figure('Position', [100, 100, 1000, 500]);
plot(f_dsb/1000, abs(DSB_TC_freq), 'b', 'LineWidth', 1.5);
hold on;
% Mark carrier frequency
[~, carrier_idx] = min(abs(f_dsb - Fc));
carrier_mag = abs(DSB_TC_freq(carrier_idx));
plot([Fc/1000, Fc/1000], [0, carrier_mag*1.1], 'r--', 'LineWidth', 2);
plot(Fc/1000, carrier_mag, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
title('DSB-TC Spectrum (Zoomed Around Carrier)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude', 'FontSize', 12);
grid on;
xlim([95, 105]);
legend('DSB-TC Spectrum', 'Carrier @ 100 kHz', 'Location', 'best');
text(100.5, carrier_mag*1.05, sprintf('Carrier\n100 kHz'), 'FontSize', 10);

saveas(gcf, fullfile(figures_dir, 'exp1_05_dsb_tc_spectrum_zoomed.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_05_dsb_tc_spectrum_zoomed.fig'));
fprintf('✓ Saved: exp1_05_dsb_tc_spectrum_zoomed.png\n');

fprintf('\nSpectrum Observations:\n');
fprintf('  - Carrier visible at 100 kHz\n');
fprintf('  - Upper Sideband (USB): 100 to 104 kHz\n');
fprintf('  - Lower Sideband (LSB): 96 to 100 kHz\n');
fprintf('  - Total bandwidth: 8 kHz (2 × message BW)\n');

%% STEP 8: PLOT DSB-TC TIME DOMAIN
fprintf('\n=== STEP 8: Plotting Time Domain Signal ===\n');

figure('Position', [100, 100, 1000, 500]);
samples_plot = min(2000, length(t));
plot(t(1:samples_plot)*1000, dsb_tc(1:samples_plot), 'b', 'LineWidth', 1);
title('DSB-TC Modulated Signal (Time Domain)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (ms)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
grid on;

saveas(gcf, fullfile(figures_dir, 'exp1_06_dsb_tc_time.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_06_dsb_tc_time.fig'));
fprintf('✓ Saved: exp1_06_dsb_tc_time.png\n');

%% STEP 9: ENVELOPE DETECTION
fprintf('\n=== STEP 9: Envelope Detection ===\n');

fprintf('Extracting envelope using Hilbert transform...\n');
% Hilbert transform method: envelope = |analytic signal|
envelope_tc = abs(hilbert(dsb_tc));

fprintf('✓ Envelope extracted\n');
fprintf('  Max envelope: %.6f\n', max(envelope_tc));
fprintf('  Min envelope: %.6f\n', min(envelope_tc));
fprintf('  Expected max: %.6f (Ac + max(m))\n', A_c + max(abs(message)));

% Plot modulated signal with envelope
figure('Position', [100, 100, 1000, 600]);
plot(t(1:samples_plot)*1000, dsb_tc(1:samples_plot), 'b', 'LineWidth', 0.8);
hold on;
plot(t(1:samples_plot)*1000, envelope_tc(1:samples_plot), 'r', 'LineWidth', 2.5);
plot(t(1:samples_plot)*1000, -envelope_tc(1:samples_plot), 'r', 'LineWidth', 2.5);
hold off;
title('DSB-TC: Modulated Signal with Envelope', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (ms)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
legend('Modulated Signal', 'Envelope', 'Location', 'best');
grid on;

saveas(gcf, fullfile(figures_dir, 'exp1_07_dsb_tc_envelope.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_07_dsb_tc_envelope.fig'));
fprintf('✓ Saved: exp1_07_dsb_tc_envelope.png\n');

%% STEP 10: RECOVER MESSAGE FROM ENVELOPE
fprintf('\n=== STEP 10: Recovering Original Message ===\n');

% Remove DC bias: m_received(t) = envelope(t) - Ac
received_tc = envelope_tc - A_c;

fprintf('✓ DC bias removed\n');
fprintf('  Received signal max: %.6f\n', max(abs(received_tc)));
fprintf('  Original message max: %.6f\n', max(abs(message)));
fprintf('  Difference: %.6f\n', abs(max(abs(received_tc)) - max(abs(message))));

% Plot comparison: Original vs Received
figure('Position', [100, 100, 1000, 600]);
plot_samp = min(10000, length(t));
plot(t(1:plot_samp)*1000, message(1:plot_samp), 'b', 'LineWidth', 1.5);
hold on;
plot(t(1:plot_samp)*1000, received_tc(1:plot_samp), 'r--', 'LineWidth', 1.5);
hold off;
title('DSB-TC Demodulation: Original vs Received Message', ...
      'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (ms)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
legend('Original Message', 'Received Message', 'Location', 'best');
grid on;

saveas(gcf, fullfile(figures_dir, 'exp1_08_dsb_tc_comparison.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_08_dsb_tc_comparison.fig'));
fprintf('✓ Saved: exp1_08_dsb_tc_comparison.png\n');

%% STEP 11: DOWNSAMPLE FOR AUDIO PLAYBACK
fprintf('\n=== STEP 11: Preparing Audio for Playback ===\n');

% Downsample back to original sampling frequency
fprintf('Downsampling from %d Hz to %d Hz...\n', Fs, Fs_original);
received_tc_audio = resample(received_tc, Q, P);

% Normalize
received_tc_audio = received_tc_audio / max(abs(received_tc_audio));

fprintf('✓ Downsampling complete\n');
fprintf('  Output samples: %d\n', length(received_tc_audio));
fprintf('  Duration: %.2f seconds\n', length(received_tc_audio)/Fs_original);

%% STEP 12: PLAY RECEIVED AUDIO
fprintf('\n=== STEP 12: Playing Received Audio ===\n');

play_duration = min(3, length(received_tc_audio)/Fs_original);
play_samples = round(play_duration * Fs_original);

fprintf('Playing received audio (first %.1f seconds)...\n', play_duration);
fprintf('Expected: Clear audio, almost identical to original\n\n');

sound(received_tc_audio(1:play_samples), Fs_original);
pause(play_duration + 1);

fprintf('✓ Playback complete\n');

% Save received audio
audiowrite(fullfile(audio_dir, 'dsb_tc_received.wav'), ...
           received_tc_audio, Fs_original);
fprintf('✓ Saved: dsb_tc_received.wav\n');

%% STEP 13: ERROR ANALYSIS
fprintf('\n=== STEP 13: Error Analysis ===\n');

% Ensure same length for comparison
min_len = min(length(audio_filtered), length(received_tc_audio));
error_signal = audio_filtered(1:min_len) - received_tc_audio(1:min_len);

% Calculate metrics
mse = mean(error_signal.^2);
signal_power = mean(audio_filtered(1:min_len).^2);
normalized_error = mse / signal_power;
SER_dB = 10*log10(signal_power / mse);

fprintf('Error Metrics:\n');
fprintf('  Mean Squared Error (MSE): %.8f\n', mse);
fprintf('  Signal Power: %.6f\n', signal_power);
fprintf('  Normalized Error: %.6f (%.4f%%)\n', normalized_error, normalized_error*100);
fprintf('  Signal-to-Error Ratio (SER): %.2f dB\n', SER_dB);

if normalized_error < 0.01
    fprintf('  ✓ Excellent recovery! Error < 1%%\n');
elseif normalized_error < 0.05
    fprintf('  ✓ Good recovery! Error < 5%%\n');
else
    fprintf('  ⚠ Moderate error. Check modulation parameters.\n');
end

%% STEP 14: SAVE ALL RESULTS
fprintf('\n=== STEP 14: Saving Results ===\n');

save(fullfile(results_dir, 'dsb_tc_results.mat'), ...
     'dsb_tc', 'envelope_tc', 'received_tc', 'received_tc_audio', ...
     'Fs', 'Fc', 'A_c', 't', 'message', 'carrier', 'P', 'Q', ...
     'mse', 'normalized_error', 'SER_dB', 'f_dsb', 'DSB_TC_freq');
fprintf('Done\n');