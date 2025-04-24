clear
clc

% Load Functions
addpath(genpath('Functions'));

% Sound Speed Map 
dx = 0.3e-3; xmax = 120e-3;
x = -xmax:dx:xmax; y = x;
Nx = numel(x); Ny = numel(y);
[X, Y] = meshgrid(x, y);
[C, c_bkgnd] = soundSpeedPhantom2D(X, Y);

% Create Transducer Ring
circle_radius = 110e-3; numElements = 256;
circle_rad_pixels = floor(circle_radius/dx);
theta = -pi:2*pi/numElements:pi-2*pi/numElements;
x_circ = circle_radius*cos(theta); 
y_circ = circle_radius*sin(theta); 
[x_idx, y_idx, ind] = sampled_circle(Nx, Ny, circle_rad_pixels, theta);

% Generate Sources
SRC = zeros(Ny, Nx, numel(theta));
for elmt_idx = 1:numel(theta)
    % Single Element Source (Random Phase and Amplitude)
    SRC(y_idx(elmt_idx), x_idx(elmt_idx), elmt_idx) = randn()+1i*randn(); 
end

% CHANGE FREQUENCY IN CLASS TO SHOW CYCLE SKIPPING
f = 0.35e6; % Frequency [Hz] -- Use 0.1-0.6 MHz

% Solve Helmholtz Equation
a0 = 10; % PML Constant
L_PML = 9.0e-3; % Thickness of PML  
adjoint = false; % Forward Solve
WVFIELD = solveHelmholtz(x, y, C, SRC, f, a0, L_PML, adjoint);

% Plot Wavefield Produced by Each Element
showAnimation = true;
if showAnimation
    subplot(1,2,1); imagesc(x, y, C); axis image; colorbar;
    xlabel('x [m]'); ylabel('y [m]'); title('Sound Speed [m/s]'); 
    hold on; plot(x_circ, y_circ, 'r.', 'LineWidth', 2);
    for elmt_idx = 1:numel(theta)
        wvfield_range = min(max(real(WVFIELD(:,:,elmt_idx)),[],"all"), ...
            -min(real(WVFIELD(:,:,elmt_idx)),[],"all"));
        subplot(1,2,2); imagesc(x, y, real(WVFIELD(:,:,elmt_idx)))
        xlabel('x [m]'); ylabel('y [m]'); axis image; colorbar; 
        title(['Wavefield from Element ', num2str(elmt_idx)]); 
        clim([-1,1]*wvfield_range); colormap gray; drawnow;
    end
end

% Assemble Channel Data 
REC_DATA = zeros(numel(theta), numel(theta));
for rx_elmt_idx = 1:numel(theta)
    REC_DATA(:,rx_elmt_idx) = ...
        WVFIELD(y_idx(rx_elmt_idx), x_idx(rx_elmt_idx), :);
end

% Save Problem Data
save('RecordedData.mat', '-v7.3', 'x', 'y', 'C', ...
    'x_circ', 'y_circ', 'f', 'REC_DATA');