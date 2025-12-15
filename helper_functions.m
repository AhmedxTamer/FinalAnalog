%% ========================================================================
%  HELPER FUNCTIONS FOR ANALOG COMMUNICATION PROJECT
%  Save this as: helper_functions.m
%  All team members should have this file
%% ========================================================================

function [freq_axis, spectrum] = get_spectrum(signal, Fs)
    % Returns frequency axis and spectrum of a signal
    % Usage: [f, S] = get_spectrum(signal, Fs)
    
    N = length(signal);
    freq_axis = linspace(-Fs/2, Fs/2, N);
    spectrum = fftshift(fft(signal));
end

function filtered_signal = ideal_lowpass_filter(signal, Fs, cutoff_freq)
    % Applies ideal lowpass filter
    % Usage: filtered = ideal_lowpass_filter(signal, Fs, 4000)
    
    N = length(signal);
    freq_axis = linspace(-Fs/2, Fs/2, N);
    
    % Get spectrum
    signal_freq = fftshift(fft(signal));
    
    % Create ideal filter
    filter_mask = abs(freq_axis) <= cutoff_freq;
    
    % Apply filter
    filtered_freq = signal_freq .* filter_mask';
    
    % Convert back to time domain
    filtered_signal = real(ifft(ifftshift(filtered_freq)));
end

function filtered_signal = ideal_bandpass_filter(signal, Fs, f_low, f_high)
    % Applies ideal bandpass filter
    % Usage: filtered = ideal_bandpass_filter(signal, Fs, 96000, 104000)
    
    N = length(signal);
    freq_axis = linspace(-Fs/2, Fs/2, N);
    
    % Get spectrum
    signal_freq = fftshift(fft(signal));
    
    % Create ideal bandpass filter
    filter_mask = (abs(freq_axis) >= f_low) & (abs(freq_axis) <= f_high);
    
    % Apply filter
    filtered_freq = signal_freq .* filter_mask';
    
    % Convert back to time domain
    filtered_signal = real(ifft(ifftshift(filtered_freq)));
end

function plot_spectrum(signal, Fs, title_text, xlim_range)
    % Quick spectrum plotting
    % Usage: plot_spectrum(signal, Fs, 'My Signal', [-10 10])
    
    [f, S] = get_spectrum(signal, Fs);
    
    figure;
    plot(f/1000, abs(S), 'LineWidth', 1.5);
    title(title_text, 'FontSize', 14);
    xlabel('Frequency (kHz)', 'FontSize', 12);
    ylabel('Magnitude', 'FontSize', 12);
    grid on;
    
    if nargin > 3
        xlim(xlim_range);
    end
end

function plot_time_signal(signal, Fs, title_text, time_range)
    % Quick time domain plotting
    % Usage: plot_time_signal(signal, Fs, 'My Signal', [0 0.01])
    
    t = (0:length(signal)-1) / Fs;
    
    figure;
    plot(t, signal, 'LineWidth', 1.5);
    title(title_text, 'FontSize', 14);
    xlabel('Time (s)', 'FontSize', 12);
    ylabel('Amplitude', 'FontSize', 12);
    grid on;
    
    if nargin > 3
        xlim(time_range);
    end
end

function play_audio_safe(signal, Fs)
    % Safely plays audio (normalizes and clips)
    % Usage: play_audio_safe(signal, Fs)
    
    % Normalize to [-1, 1]
    signal = signal / (max(abs(signal)) + eps);
    
    % Clip any values outside range (safety)
    signal(signal > 1) = 1;
    signal(signal < -1) = -1;
    
    % Play
    sound(signal, Fs);
end

function save_plot(filename)
    % Saves current figure as high-res PNG
    % Usage: save_plot('exp1_dsb_tc_spectrum')
    
    saveas(gcf, [filename '.png']);
    fprintf('Saved: %s.png\n', filename);
end

function SNR_actual = calculate_snr(signal, noisy_signal)
    % Calculates actual SNR in dB
    % Usage: SNR = calculate_snr(original, noisy)
    
    noise = noisy_signal - signal;
    signal_power = mean(signal.^2);
    noise_power = mean(noise.^2);
    SNR_actual = 10*log10(signal_power / noise_power);
    fprintf('Actual SNR: %.2f dB\n', SNR_actual);
end