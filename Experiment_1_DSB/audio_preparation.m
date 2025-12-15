clear all; close all; clc;

% Navigate to project root (up 2 levels from Code/Experiment_1_DSB/)
project_dir = "/MATLAB Drive/FinalAnalog";
addpath(fullfile(project_dir, 'Code'));

% Define directories
results_dir = fullfile(project_dir, 'Results');
figures_dir = fullfile(project_dir, 'Figures', 'Experiment_1_DSB');
audio_dir = fullfile(project_dir, 'Audio_Samples');

fprintf('Project Directory: %s\n\n', project_dir);
%% STEP 1: LOAD AUDIO FILE
fprintf('=== STEP 1: Loading Audio File (eric) ===\n');

% Find audio file
audio_path = fullfile(project_dir, 'eric');
if ~exist(audio_path, 'file')
    error('ERROR: Audio file "eric" or "eric.wav" not found in %s', project_dir);
end

fprintf('Loading: %s\n', audio_path);
[audio, Fs_original] = audioread(audio_path);

% Convert stereo to mono
if size(audio, 2) == 2
    audio = audio(:, 1);
    fprintf('✓ Converted stereo to mono\n');
end

fprintf('✓ Audio loaded successfully!\n');
fprintf('  Sampling Rate: %d Hz (%.1f kHz)\n', Fs_original, Fs_original/1000);
fprintf('  Duration: %.2f seconds\n', length(audio)/Fs_original);
fprintf('  Total Samples: %d\n', length(audio));
fprintf('  Max Amplitude: %.4f\n', max(abs(audio)));

%% STEP 2: PLOT ORIGINAL AUDIO SPECTRUM
fprintf('\n=== STEP 2: Analyzing Original Spectrum ===\n');

% Compute FFT
N = length(audio);
f = linspace(-Fs_original/2, Fs_original/2, N);
Audio_freq = fftshift(fft(audio));

% Plot full spectrum
figure('Position', [100, 100, 1000, 500]);
plot(f/1000, abs(Audio_freq), 'b', 'LineWidth', 1.5);
title('Original Audio Spectrum', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude', 'FontSize', 12);
grid on;
xlim([-Fs_original/2000, Fs_original/2000]);

% Save figure
saveas(gcf, fullfile(figures_dir, 'exp1_01_original_spectrum.png'));
fprintf('✓ Saved: exp1_01_original_spectrum.png\n');

%% STEP 3: DESIGN IDEAL LOWPASS FILTER
fprintf('\n=== STEP 3: Designing Ideal Lowpass Filter ===\n');

cutoff_freq = 4000; % 4 kHz bandwidth
fprintf('Filter Type: Ideal Lowpass\n');
fprintf('Cutoff Frequency: %d Hz\n', cutoff_freq);
fprintf('Bandwidth: %d Hz\n', cutoff_freq);

% Create ideal rectangular filter in frequency domain
Filter = zeros(size(Audio_freq));
Filter(abs(f) <= cutoff_freq) = 1;

fprintf('✓ Filter designed\n');
fprintf('  Passband: DC to %d Hz\n', cutoff_freq);
fprintf('  Stopband: > %d Hz\n', cutoff_freq);

%% STEP 4: APPLY FILTER AND GET FILTERED SIGNAL
fprintf('\n=== STEP 4: Applying Filter ===\n');

% Apply filter in frequency domain
Audio_filtered_freq = Audio_freq .* Filter;

% Convert back to time domain
audio_filtered = real(ifft(ifftshift(Audio_filtered_freq)));

fprintf('✓ Filtering complete\n');
fprintf('  Input samples: %d\n', length(audio));
fprintf('  Output samples: %d\n', length(audio_filtered));
fprintf('  Signal preserved: %.2f%%\n', (sum(abs(Filter))/length(Filter))*100);

%% STEP 5: PLOT FILTERED SPECTRUM
fprintf('\n=== STEP 5: Plotting Filtered Spectrum ===\n');

figure('Position', [100, 100, 1000, 500]);
plot(f/1000, abs(Audio_filtered_freq), 'r', 'LineWidth', 1.5);
title('Filtered Audio Spectrum (Bandwidth = 4 kHz)', ...
      'FontSize', 14, 'FontWeight', 'bold');
xlabel('Frequency (kHz)', 'FontSize', 12);
ylabel('Magnitude', 'FontSize', 12);
grid on;
xlim([-10, 10]);

% Add vertical lines at cutoff
hold on;
ymax = max(abs(Audio_filtered_freq));
plot([-4, -4], [0, ymax], 'k--', 'LineWidth', 1.5);
plot([4, 4], [0, ymax], 'k--', 'LineWidth', 1.5);
text(-4, ymax*0.9, '← -4 kHz', 'FontSize', 10);
text(4, ymax*0.9, '4 kHz →', 'FontSize', 10, 'HorizontalAlignment', 'right');
hold off;

saveas(gcf, fullfile(figures_dir, 'exp1_02_filtered_spectrum.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_02_filtered_spectrum.fig'));
fprintf('✓ Saved: exp1_02_filtered_spectrum.png\n');

%% STEP 6: COMPARE ORIGINAL VS FILTERED (TIME DOMAIN)
fprintf('\n=== STEP 6: Time Domain Comparison ===\n');

t_original = (0:length(audio)-1) / Fs_original;

figure('Position', [100, 100, 1000, 700]);

% Original signal
subplot(2,1,1);
plot(t_original, audio, 'b', 'LineWidth', 1);
title('Original Audio Signal', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Time (s)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
xlim([0, min(0.1, t_original(end))]);
grid on;

% Filtered signal
subplot(2,1,2);
plot(t_original, audio_filtered, 'r', 'LineWidth', 1);
title('Filtered Audio Signal (BW = 4 kHz)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Time (s)', 'FontSize', 11);
ylabel('Amplitude', 'FontSize', 11);
xlim([0, min(0.1, t_original(end))]);
grid on;

saveas(gcf, fullfile(figures_dir, 'exp1_03_time_comparison.png'));
saveas(gcf, fullfile(figures_dir, 'exp1_03_time_comparison.fig'));
fprintf('✓ Saved: exp1_03_time_comparison.png\n');

%% STEP 7: PLAY AND VERIFY FILTERED AUDIO
fprintf('\n=== STEP 7: Audio Quality Check ===\n');

% Normalize for playback
audio_filtered_norm = audio_filtered / max(abs(audio_filtered));

% Play first 3 seconds
play_duration = min(3, length(audio_filtered)/Fs_original);
play_samples = round(play_duration * Fs_original);

fprintf('Playing filtered audio (first %.1f seconds)...\n', play_duration);
fprintf('Listen for:\n');
fprintf('  - Clear speech/music\n');
fprintf('  - Removed high frequencies\n');
fprintf('  - Minimal distortion\n\n');

sound(audio_filtered_norm(1:play_samples), Fs_original);
pause(play_duration + 1);

fprintf('✓ Audio playback complete\n');

%% STEP 8: SAVE FILTERED AUDIO FILE
fprintf('\n=== STEP 8: Saving Audio Files ===\n');

% Save original
audiowrite(fullfile(audio_dir, 'original_audio.wav'), ...
           audio/max(abs(audio)), Fs_original);
fprintf('✓ Saved: original_audio.wav\n');

% Save filtered
audiowrite(fullfile(audio_dir, 'filtered_audio.wav'), ...
           audio_filtered_norm, Fs_original);
fprintf('✓ Saved: filtered_audio.wav\n');

%% STEP 9: SAVE DATA 
fprintf('\n=== STEP 9: Saving Data ===\n');

% Save MAT file with all necessary data
save(fullfile(results_dir, 'filtered_audio.mat'), ...
     'audio_filtered', 'Fs_original', 'cutoff_freq', ...
     'audio', 'f', 'Audio_freq', 'Audio_filtered_freq');