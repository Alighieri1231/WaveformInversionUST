clear
clc

% Load Functions
addpath(genpath('Functions'));

% Sound Speed Map 
dxi = 1.0e-3; xmax = 120e-3;
xi = -xmax:dxi:xmax; yi = xi;
Nxi = numel(xi); Nyi = numel(yi);
[Xi, Zi] = meshgrid(xi, yi);
[C, c_bkgnd] = soundSpeedPhantom2D(Xi, Zi);

% Create Transducer Ring
circle_radius = 110e-3; numElements = 256;
circle_rad_pixels = floor(circle_radius/dxi);
theta = -pi:2*pi/numElements:pi-2*pi/numElements;
x_circ = circle_radius*cos(theta); 
y_circ = circle_radius*sin(theta); 
[x_idx, y_idx, ind] = sampled_circle(Nxi, Nyi, circle_rad_pixels, theta);

% Generate Sources
SRC = zeros(Nyi, Nxi, numel(theta));
for elmt_idx = 1:numel(theta)
    % Single Element Source
    SRC(y_idx(elmt_idx), x_idx(elmt_idx), elmt_idx) = 1; 
end

% Ultrasound Frequency
df = 0.005e6; % Increment in Frequency [Hz]
flow = 0.1e6; % Lowest Frequency [Hz]
fhigh = 0.5e6; % Highest Frequency [Hz]
f = flow:df:fhigh; % Frequency [Hz]
resp_freq = hanning(numel(f)); % Frequency Response

% Solve Helmholtz Equation at Each Frequency
a0 = 10; % PML Constant
L_PML = 9.0e-3; % Thickness of PML  
adjoint = false; % Forward Solve
elmt = 64; % Which Element to Transmit From
WVFIELD_F = zeros(Nyi, Nxi, numel(f));
for f_idx = 1:numel(f)
    WVFIELD_F(:,:,f_idx) = solveHelmholtz(xi, yi, C, ...
        SRC(:,:,elmt), f(f_idx), a0, L_PML, adjoint);
    disp(['Frequency ', num2str(f(f_idx)), ' Hz']);
end

% Time Axis and Inverse Fourier Transform
Nt = 501; % Number of Time Points
tend = 2*xmax/c_bkgnd;
time = linspace(0, tend, Nt);

% Inverse Discrete-Time Fourier Transform (DTFT) - Not an IFFT though!
IDTFT = exp(1i*2*pi*f.*time')*df;
WVFIELD_T = permute(pagemtimes(IDTFT, resp_freq .* ...
    permute(WVFIELD_F,[3,1,2])), [2,3,1]);

%% Plot Wavefield as a Function of Time

% Sound Speed Map and Elements
subplot(2,2,1); imagesc(xi, yi, C); axis image; colorbar;
xlabel('x [m]'); ylabel('y [m]'); title('Sound Speed [m/s]'); 
hold on; plot(x_circ, y_circ, 'r.', ...
    x_circ(elmt), y_circ(elmt), 'yo', 'LineWidth', 2);

% Recorded Channel Data
channelData = zeros(numel(time), numel(theta));

% Plot Wavefield and Recorded Data vs Time
ntskip = 5; % Number of Time Steps to Skip
for t_idx = 1:Nt
    % Assemble Channel Data
    for elmt_idx = 1:numel(theta)
        channelData(t_idx, elmt_idx) = ...
            WVFIELD_T(y_idx(elmt_idx), x_idx(elmt_idx), t_idx);
    end
    if mod(t_idx, ntskip) == 0
        % Plot Wavefield vs Time
        subplot(2,2,3); imagesc(xi, yi, real(WVFIELD_T(:,:,t_idx)))
        xlabel('x [m]'); ylabel('y [m]'); axis image; colorbar; 
        title(['Wavefield at Time t = ', num2str(time(t_idx)*(1e6)), ' \mus']); 
        wvfield_range = min(max(real(WVFIELD_T(:,:,t_idx)),[],'all'), ...
            -min(real(WVFIELD_T(:,:,t_idx)),[],'all'));
        clim([-1,1]*wvfield_range); hold on; plot(x_circ, y_circ, 'r.', ...
            x_circ(elmt), y_circ(elmt), 'yo', 'LineWidth', 2);
        % Plot Channel Data
        subplot(2,2,[2,4]); imagesc(1:numel(theta), time*(1e6), real(channelData))
        xlabel('Element'); ylabel('time [\mus]'); title('Channel Data'); 
        clim([-1,1]*wvfield_range); colorbar; drawnow;
    end
end