clear all; close all; clc;

% Navigate to project root
project_dir = "/MATLAB Drive/FinalAnalog";
addpath(fullfile(project_dir, 'Code'));

% Define directories
results_dir = fullfile(project_dir, 'Results');
figures_dir = fullfile(project_dir, 'Figures', 'Experiment_1_DSB');
audio_dir = fullfile(project_dir, 'Audio_Samples');
%% STEP 1: LOAD FILTERED AUDIO
fprintf('=== STEP 1: Loading Filtered Audio from Member 1 ===\n');
filtered_path = fullfile(results_dir, 'filtered_audio.mat');
if ~exist(filtered_path, 'file')
error(['ERROR: filtered_audio.mat not found!\n' ...
'Member 1 must run exp1_member1_audio_preparation.m first!']);
end
load(filtered_path);
fprintf('✓ Loaded: filtered_audio.mat\n');
fprintf('  Fs_original: %d Hz\n', Fs_original);
fprintf('  Signal length: %d samples\n', length(audio_filtered));
%% STEP 2: RESAMPLE TO HIGH SAMPLING FREQUENCY
fprintf('\n=== STEP 2: Resampling Signal ===\n');
Fc = 100e3;  % Carrier frequency = 100 kHz
Fs = 5 * Fc;  % Sampling frequency = 500 kHz
fprintf('Carrier Frequency: %.0f kHz\n', Fc/1000);
fprintf('Sampling Frequency: %.0f kHz\n', Fs/1000);
[P, Q] = rat(Fs / Fs_original);
message = resample(audio_filtered, P, Q);
fprintf('✓ Resampled to %d Hz\n', Fs);
fprintf('  New length: %d samples\n', length(message));
%% STEP 3: GENERATE CARRIER
fprintf('\n=== STEP 3: Generating Carrier ===\n');
t = (0:length(message)-1) / Fs;
carrier = cos(2*pi*Fc*t)';
fprintf('✓ Carrier generated: cos(2π × 100kHz × t)\n');
%% STEP 4: GENERATE DSB-SC SIGNAL
fprintf('\n=== STEP 4: Generating DSB-SC Signal ===\n');
% DSB-SC: NO DC bias added
% Formula: s(t) = m(t) × cos(2πfct)
dsb_sc = message .* carrier;
fprintf('✓ DSB-SC signal generated\n');
fprintf('  Formula: s(t) = m(t) × cos(2π×100kHz×t)\n');
fprintf('  Note: NO DC bias (Ac = 0)\n');
fprintf('  Signal statistics:\n');
fprintf('    Max: %.6f\n', max(abs(dsb_sc)));
fprintf('    Mean: %.6f (should be ~0)\n', mean(dsb_sc));
%% STEP 5: PLOT DSB-SC SPECTRUM
fprintf('\n=== STEP 5: Plotting DSB-SC Spectrum ===\n');
N_dsb = length(dsb_sc);
f_dsb = linspace(-Fs/2, Fs/2, N_dsb);
DSB_SC_freq = fftshift(fft(dsb_sc));
% Full spectrum
figure('Position', [100, 100, 1000, 500]);
plot(f_dsb/1000, abs(DSB_SC_freq), 'b', 'LineWidth', 1.5);
title('DSB-SC Spectrum (No Carrier)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude', 'FontSize', 12);
grid on;
xlim([80, 120]);
saveas(gcf, fullfile(figures_dir, 'exp1_09_dsb_sc_spectrum.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_09_dsb_sc_spectrum.fig'));
fprintf('✓ Saved: exp1_09_dsb_sc_spectrum.png\n');
% Zoomed spectrum
figure('Position', [100, 100, 1000, 500]);
plot(f_dsb/1000, abs(DSB_SC_freq), 'b', 'LineWidth', 1.5);
hold on;
% Mark where carrier WOULD be (but isn't transmitted)
plot([100, 100], [0, max(abs(DSB_SC_freq))*0.5], 'r--', 'LineWidth', 2);
text(100.5, max(abs(DSB_SC_freq))*0.45, ...
'Carrier NOT transmitted', 'FontSize', 10, 'Color', 'r');
hold off;
title('DSB-SC Spectrum (Zoomed - Note Missing Carrier)', ...
'FontSize', 14, 'FontWeight', 'bold');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude', 'FontSize', 12);
grid on;
xlim([95, 105]);
saveas(gcf, fullfile(figures_dir, 'exp1_10_dsb_sc_spectrum_zoomed.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_10_dsb_sc_spectrum_zoomed.fig'));
fprintf('✓ Saved: exp1_10_dsb_sc_spectrum_zoomed.png\n');
fprintf('\nSpectrum Observations:\n');
fprintf('  - NO carrier component at 100 kHz\n');
fprintf('  - Upper Sideband: 100-104 kHz\n');
fprintf('  - Lower Sideband: 96-100 kHz\n');
fprintf('  - Bandwidth: 8 kHz (same as DSB-TC)\n');
fprintf('  - Power saved by not transmitting carrier!\n');
%% STEP 6: PLOT TIME DOMAIN
fprintf('\n=== STEP 6: Plotting Time Domain ===\n');
figure('Position', [100, 100, 1000, 500]);
samples_plot = min(2000, length(t));
plot(t(1:samples_plot)*1000, dsb_sc(1:samples_plot), 'b', 'LineWidth', 1);
title('DSB-SC Modulated Signal (Time Domain)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time (ms)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
grid on;
saveas(gcf, fullfile(figures_dir, 'exp1_11_dsb_sc_time.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_11_dsb_sc_time.fig'));
fprintf('✓ Saved: exp1_11_dsb_sc_time.png\n');
%% STEP 7: TRY ENVELOPE DETECTION (WILL FAIL!)
fprintf('\n=== STEP 7: Attempting Envelope Detection ===\n');
fprintf('⚠️  WARNING: This WILL FAIL for DSB-SC!\n');
fprintf('   (Demonstrating why envelope detection requires carrier)\n\n');
% Extract envelope
envelope_sc = abs(hilbert(dsb_sc));
fprintf('Envelope extracted\n');
fprintf('  Max envelope: %.6f\n', max(envelope_sc));
fprintf('  Min envelope: %.6f\n', min(envelope_sc));
% Plot - showing the failure
figure('Position', [100, 100, 1000, 600]);
plot(t(1:samples_plot)*1000, dsb_sc(1:samples_plot), 'b', 'LineWidth', 0.8);
hold on;
plot(t(1:samples_plot)*1000, envelope_sc(1:samples_plot), 'r', 'LineWidth', 2.5);
plot(t(1:samples_plot)*1000, -envelope_sc(1:samples_plot), 'r', 'LineWidth', 2.5);
hold off;
title('DSB-SC: Envelope Detection FAILS!', ...
'FontSize', 14, 'FontWeight', 'bold', 'Color', 'r');
xlabel('Time (ms)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
legend('DSB-SC Signal', 'Envelope (WRONG!)', 'Location', 'best');
grid on;
% Add annotation explaining the failure
annotation('textbox', [0.15, 0.75, 0.3, 0.15], ...
'String', {'Envelope does NOT', 'match original message!', ...
'Envelope detection fails', 'without carrier'}, ...
'BackgroundColor', 'yellow', 'FontSize', 10, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_dir, 'exp1_12_dsb_sc_envelope_fail.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_12_dsb_sc_envelope_fail.fig'));
fprintf('✓ Saved: exp1_12_dsb_sc_envelope_fail.png\n');
%% STEP 8: COMPARE ENVELOPE WITH ORIGINAL MESSAGE
fprintf('\n=== STEP 8: Demonstrating the Failure ===\n');
% Downsample envelope to compare
envelope_audio = resample(envelope_sc, Q, P);
envelope_audio = envelope_audio / max(abs(envelope_audio));
% Plot comparison
figure('Position', [100, 100, 1000, 600]);
plot_samples = min(10000, length(t));
t_plot = t(1:plot_samples) * 1000;
subplot(3,1,1);
plot(t_plot, message(1:plot_samples), 'b', 'LineWidth', 1.5);
title('Original Message', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Amplitude');
grid on;
subplot(3,1,2);
plot(t_plot, envelope_sc(1:plot_samples), 'r', 'LineWidth', 1.5);
title('Extracted Envelope (WRONG!)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
ylabel('Amplitude');
grid on;
subplot(3,1,3);
t_audio = (0:length(audio_filtered)-1) / Fs_original * 1000;
max_len = min(length(t_audio), length(envelope_audio));
plot(t_audio(1:max_len), audio_filtered(1:max_len), 'b', 'LineWidth', 1);
hold on;
plot(t_audio(1:max_len), envelope_audio(1:max_len), 'r--', 'LineWidth', 1);
hold off;
title('Comparison (Downsampled)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time (ms)');
ylabel('Amplitude');
legend('Original', 'Envelope', 'Location', 'best');
grid on;
xlim([0, 100]);
saveas(gcf, fullfile(figures_dir, 'exp1_12b_envelope_comparison.png'));
fprintf('✓ Saved: exp1_12b_envelope_comparison.png\n');
%% STEP 9: PLAY BOTH AUDIOS FOR COMPARISON
fprintf('\n=== STEP 9: Audio Comparison ===\n');
play_samples = min(3*Fs_original, length(audio_filtered));
fprintf('Playing ORIGINAL audio...\n');
sound(audio_filtered(1:play_samples)/max(abs(audio_filtered)), Fs_original);
pause(3.5);
fprintf('Playing ENVELOPE-DETECTED audio (will sound WRONG)...\n');
sound(envelope_audio(1:play_samples), Fs_original);
pause(3.5);
fprintf('\n❌ Envelope detection FAILS for DSB-SC!\n');
fprintf('   The audio sounds distorted/wrong.\n');
fprintf('   This is why we need coherent detection!\n');
% Save the wrong audio for documentation
audiowrite(fullfile(audio_dir, 'dsb_sc_envelope_WRONG.wav'), ...
envelope_audio, Fs_original);
fprintf('✓ Saved: dsb_sc_envelope_WRONG.wav (for demonstration)\n');
%% STEP 10: SAVE RESULTS FOR MEMBER 2 PART B
fprintf('\n=== STEP 10: Saving Results ===\n');
save(fullfile(results_dir, 'dsb_sc_results.mat'), ...
'dsb_sc', 'message', 'carrier', 't', 'Fs', 'Fc', 'P', 'Q', ...
'f_dsb', 'DSB_SC_freq', 'envelope_sc');
fprintf('✓ Saved: dsb_sc_results.mat\n');
fprintf('\n📦 File Contents:\n');
fprintf('   - dsb_sc    : DSB-SC modulated signal\n');
fprintf('   - message   : Original message (resampled)\n');
fprintf('   - carrier   : Carrier signal\n');
fprintf('   - All necessary parameters for Part B\n');
%% FINAL SUMMARY
fprintf('\n\n========================================\n');
fprintf('✅ DSB-SC GENERATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Summary:\n');
fprintf('  ✓ DSB-SC signal generated (no carrier)\n');
fprintf('  ✓ Envelope detection attempted\n');
fprintf('  ❌ Envelope detection FAILED (as expected)\n');
fprintf('  ✓ Failure demonstrated and documented\n');
fprintf('\nKey Findings:\n');
fprintf('  - DSB-SC spectrum has NO carrier component\n');
fprintf('  - Envelope detection requires transmitted carrier\n');
fprintf('  - Without carrier, envelope ≠ original message\n');
fprintf('  - Solution: Use coherent detection (Member 2 Part B)\n');
fprintf('\nFiles Created:\n');
fprintf('  - 5 figure files\n');
fprintf('  - 1 result file (.mat)\n');
fprintf('  - 1 audio file (wrong demodulation)\n');
fprintf('\n📋 Next: Run exp1_member2_coherent_detection.m\n');
fprintf('========================================\n');