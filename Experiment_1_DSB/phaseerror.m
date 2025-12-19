clear all; close all; clc;

%% HELPER FUNCTION
function [freq_axis, spectrum] = get_spectrum(signal, Fs)
    N = length(signal);
    freq_axis = linspace(-Fs/2, Fs/2, N);
    spectrum = fftshift(fft(signal));
end

%% SETUP PATHS
project_dir = fileparts(fileparts(pwd));
addpath(fullfile(project_dir, 'Code'));

results_dir = fullfile(project_dir, 'Results');
figures_dir = fullfile(project_dir, 'Figures', 'Experiment_1_DSB');
audio_dir = fullfile(project_dir, 'Audio_Samples');

%% STEP 1: LOAD DSB-SC RESULTS
fprintf('=== STEP 1: Loading DSB-SC Data===\n');

dsbsc_path = fullfile(results_dir, 'dsb_sc_results.mat');
filtered_path = fullfile(results_dir, 'filtered_audio.mat');

if ~exist(dsbsc_path, 'file')
    error('ERROR: dsb_sc_results.mat not found!');
end
if ~exist(filtered_path, 'file')
    error('ERROR: filtered_audio.mat not found!');
end

load(dsbsc_path);      % dsb_sc, message, carrier, t, Fs, Fc, P, Q
load(filtered_path);   % audio_filtered, Fs_original, cutoff_freq

fprintf('✓ Loaded DSB-SC data\n');
fprintf('  Carrier Frequency: %.3f kHz\n', Fc/1000);
fprintf('  Sampling Frequency: %.0f kHz\n', Fs/1000);
fprintf('  DSB-SC signal length: %d samples\n', length(dsb_sc));

%% STEP 2: DEFINE PHASE ERROR
fprintf('\n=== STEP 2: Defining Phase Error ===\n');

phase_error_deg = 20;       % Phase error in degrees
phase_error_rad = deg2rad(phase_error_deg);  % Convert to radians

fprintf('Phase Synchronization:\n');
fprintf('  Receiver phase error: %.1f degrees (%.3f rad)\n', phase_error_deg, phase_error_rad);

%% STEP 3: GENERATE CARRIER WITH PHASE ERROR
fprintf('\n=== STEP 3: Generating Carrier with Phase Error ===\n');

% Receiver uses correct frequency but with phase error
carrier_phase_error = cos(2*pi*Fc*t + phase_error_rad)';

fprintf('✓ Generated carrier with phase error\n');
fprintf('  Formula: cos(2π × %.3f kHz × t + %.3f rad)\n', Fc/1000, phase_error_rad);

%% STEP 4: COHERENT DETECTION WITH PHASE ERROR
fprintf('\n=== STEP 4: Performing Coherent Detection ===\n');

% Multiply DSB-SC signal by erroneous carrier
demodulated_phase_error = dsb_sc .* carrier_phase_error;

fprintf('✓ Multiplication complete\n');

% Lowpass filter
fprintf('Applying lowpass filter (cutoff = %d Hz)...\n', cutoff_freq);

N_demod = length(demodulated_phase_error);
f_demod = linspace(-Fs/2, Fs/2, N_demod);
Demod_freq = fftshift(fft(demodulated_phase_error));

% Ideal LPF
LPF = zeros(size(Demod_freq));
LPF(abs(f_demod) <= cutoff_freq) = 1;

Demod_filtered_freq = Demod_freq .* LPF;
received_phase_error = real(ifft(ifftshift(Demod_filtered_freq)));

% Scale by 2 (cos² identity)
received_phase_error = 2 * received_phase_error;

fprintf('✓ Coherent detection complete\n');

%% STEP 5: PLOT EFFECT OF PHASE ERROR
fprintf('\n=== STEP 5: Visualizing Effect of Phase Error ===\n');

plot_samples = min(5000, length(t));  % adjust as needed

figure('Position', [100, 100, 1200, 600]);
plot(t(1:plot_samples), received_phase_error(1:plot_samples), 'r', 'LineWidth', 1.5);
hold on;
plot(t(1:plot_samples), message(1:plot_samples), 'b--', 'LineWidth', 1);
hold off;
title(sprintf('Coherent Detection with Phase Error (%.1f°)', phase_error_deg), ...
      'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
legend('Received (with phase error)', 'Original Message', 'Location', 'best');
grid on;

saveas(gcf, fullfile(figures_dir, 'exp1_phase_error_time.png'));
fprintf('✓ Saved: exp1_phase_error_time.png\n');

%% STEP 6: FREQUENCY DOMAIN ANALYSIS
fprintf('\n=== STEP 6: Frequency Domain Analysis ===\n');

[f_rec, Rec_freq] = get_spectrum(received_phase_error, Fs);

figure('Position', [100, 100, 1000, 500]);
plot(f_rec/1000, abs(Rec_freq), 'r', 'LineWidth', 1.5);
title('Received Signal Spectrum with Phase Error', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude', 'FontSize', 12);
grid on;

saveas(gcf, fullfile(figures_dir, 'exp1_phase_error_spectrum.png'));
fprintf('✓ Saved: exp1_phase_error_spectrum.png\n');

%% STEP 7: DOWNSAMPLE AND PLAY AUDIO
fprintf('\n=== STEP 7: Audio Demonstration ===\n');

received_phase_error_audio = resample(received_phase_error, Q, P);
received_phase_error_audio = received_phase_error_audio / max(abs(received_phase_error_audio));

play_samples = min(5*Fs_original, length(received_phase_error_audio));

fprintf('Playing audio with phase error (%.1f°)...\n', phase_error_deg);
sound(received_phase_error_audio(1:play_samples), Fs_original);
pause(5.5);

audiowrite(fullfile(audio_dir, 'dsb_sc_phase_error.wav'), ...
           received_phase_error_audio, Fs_original);
fprintf('✓ Saved: dsb_sc_phase_error.wav\n');

%% STEP 8: ERROR CALCULATION
fprintf('\n=== STEP 8: Error Analysis ===\n');

min_len = min(length(audio_filtered), length(received_phase_error_audio));
error_signal = audio_filtered(1:min_len) - received_phase_error_audio(1:min_len);

mse_phase = mean(error_signal.^2);
signal_power = mean(audio_filtered(1:min_len).^2);
normalized_error = mse_phase / signal_power;
SER_dB = 10*log10(signal_power / mse_phase);

fprintf('Error Metrics:\n');
fprintf('  Mean Squared Error: %.6f\n', mse_phase);
fprintf('  Normalized Error: %.4f (%.2f%%)\n', normalized_error, normalized_error*100);
fprintf('  Signal-to-Error Ratio: %.2f dB\n', SER_dB);

%% STEP 9: SAVE RESULTS
fprintf('\n=== STEP 9: Saving Results ===\n');

save(fullfile(results_dir, 'phase_error_results.mat'), ...
     'received_phase_error', 'received_phase_error_audio', ...
     'phase_error_deg', 'phase_error_rad', ...
     'mse_phase', 'normalized_error', 'SER_dB', ...
     'error_signal');

fprintf('✓ Saved: phase_error_results.mat\n');
