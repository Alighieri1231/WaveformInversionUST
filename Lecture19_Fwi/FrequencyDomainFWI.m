clear
clc

% Load Functions
addpath(genpath('Functions'));

% Load Problem Data
load('RecordedData.mat', 'x', 'y', 'C', ...
    'x_circ', 'y_circ', 'f', 'REC_DATA');
numElements = numel(x_circ); % Number of Transducer Elements
assert(numElements == numel(y_circ));

% Which Subset of Transmits to Use
dwnsmp = 1; % can be 1, 2, or 4 (faster with more downsampling)
            % NOTE: dwnsmp = 1 to get the results in the paper
tx_include = 1:dwnsmp:numElements;
REC_DATA = REC_DATA(tx_include,:); 

% Extract Subset of Signals within Acceptance Angle
numElemLeftRightExcl = 31;
elemLeftRightExcl = -numElemLeftRightExcl:numElemLeftRightExcl;
elemInclude = true(numElements, numElements);
for tx_element = 1:numElements 
    elemLeftRightExclCurrent = elemLeftRightExcl + tx_element;
    elemLeftRightExclCurrent(elemLeftRightExclCurrent<1) = numElements + ...
         elemLeftRightExclCurrent(elemLeftRightExclCurrent<1);
    elemLeftRightExclCurrent(elemLeftRightExclCurrent>numElements) = ...
        elemLeftRightExclCurrent(elemLeftRightExclCurrent>numElements) - numElements;
    elemInclude(tx_element,elemLeftRightExclCurrent) = false;
end

%% Frequency-Domain Full Waveform Inversion (FWI)

% Parameters for Conjugate Gradient Reconstruction
Niter = 10; % Number of Iterations
momentumFormula = 4; % Momentum Formula for Conjugate Gradient
                     % 0 -- No Momentum (Gradient Descent)
                     % 1 -- Fletcher-Reeves (FR)
                     % 2 -- Polak-Ribiere (PR)
                     % 3 -- Combined FR + PR
                     % 4 -- Hestenes-Stiefel (HS)
stepSizeCalculation = 1; % Which Step Size Calculation:
                         % 1 -- Not Involving Gradient Nor Search Direction
                         % 2 -- Involving Gradient BUT NOT Search Direction
                         % 3 -- Involving Gradient AND Search Direction
c_init = 1480; % Initial Homogeneous Sound Speed [m/s] Guess

% Computational Grid (and Element Placement on Grid) for Reconstruction
dxi = 0.8e-3; xmax = 120e-3;
xi = -xmax:dxi:xmax; yi = xi;
Nxi = numel(xi); Nyi = numel(yi);
[Xi, Yi] = meshgrid(xi, yi);
x_idx = dsearchn(xi(:), x_circ(:));
y_idx = dsearchn(yi(:), y_circ(:));
ind = sub2ind([Nyi, Nxi], y_idx, x_idx);

% Solver Options for Helmholtz Equation
a0 = 10; % PML Constant
L_PML = 9.0e-3; % Thickness of PML  

% Generate Sources
SRC = zeros(Nyi, Nxi, numel(tx_include));
for tx_elmt_idx = 1:numel(tx_include)
    % Single Element Source
    x_idx_src = x_idx(tx_include(tx_elmt_idx)); 
    y_idx_src = y_idx(tx_include(tx_elmt_idx)); 
    SRC(y_idx_src, x_idx_src, tx_elmt_idx) = 1; 
end

% (Nonlinear) Conjugate Gradient
search_dir = zeros(Nyi,Nxi); % Conjugate Gradient Direction
gradient_img_prev = zeros(Nyi,Nxi); % Previous Gradient Image
VEL_ESTIM = c_init*ones(Nyi,Nxi); % Initial Sound Speed Image [m/s]
SLOW_ESTIM = 1./VEL_ESTIM; % Initial Slowness Image [s/m]
crange = [1400, 1600]; % For reconstruction display [m/s]
for iter = 1:Niter
    % Step 1: Calculate Gradient/Backprojection
    % (1A) Solve Forward Helmholtz Equation (H is Helmholtz matrix and u is the wavefield)
    tic; WVFIELD = solveHelmholtz(xi, yi, VEL_ESTIM, SRC, f, a0, L_PML, false);
    disp(num2str(norm(WVFIELD(:))))

    % (1B) Estimate Forward Sources and Adjust Simulated Fields Accordingly
    SRC_ESTIM = zeros(1,1,numel(tx_include));
    for tx_elmt_idx = 1:numel(tx_include)
        WVFIELD_elmt = WVFIELD(:,:,tx_elmt_idx);
        REC_SIM = WVFIELD_elmt(ind(elemInclude(tx_include(tx_elmt_idx),:))); 
        REC = REC_DATA(tx_elmt_idx, elemInclude(tx_include(tx_elmt_idx),:)); 
        SRC_ESTIM(tx_elmt_idx) = (REC_SIM(:)'*REC(:)) / ...
            (REC_SIM(:)'*REC_SIM(:)); % Source Estimate
    end
    WVFIELD = SRC_ESTIM.*WVFIELD;
    disp(num2str(norm(WVFIELD(:))))

    % (1C) Build Adjoint Sources - Based on Errors
    ADJ_SRC = zeros(Nyi, Nxi, numel(tx_include));
    REC_SIM = zeros(numel(tx_include), numElements);
    for tx_elmt_idx = 1:numel(tx_include)
        WVFIELD_elmt = WVFIELD(:,:,tx_elmt_idx);
        REC_SIM(tx_elmt_idx,elemInclude(tx_include(tx_elmt_idx),:)) = ...
            WVFIELD_elmt(ind(elemInclude(tx_include(tx_elmt_idx),:))); 
        ADJ_SRC_elmt = zeros(Nyi, Nxi);
        ADJ_SRC_elmt(ind(elemInclude(tx_include(tx_elmt_idx),:))) = ...
            REC_SIM(tx_elmt_idx, elemInclude(tx_include(tx_elmt_idx),:)) - ...
            REC_DATA(tx_elmt_idx, elemInclude(tx_include(tx_elmt_idx),:));
        ADJ_SRC(:,:,tx_elmt_idx) = ADJ_SRC_elmt;
    end
    % (1D) Calculate Virtual Source [dH/ds u] where s is slowness
    VIRT_SRC = ((2*(2*pi*f).^2).*SLOW_ESTIM).*WVFIELD;
    % (1E) Backproject Error (Gradient = Backprojection)
    ADJ_WVFIELD = solveHelmholtz(xi, yi, VEL_ESTIM, ADJ_SRC, f, a0, L_PML, true);
    disp(num2str(norm(ADJ_WVFIELD(:))))
    break
end

%     BACKPROJ = -real(conj(VIRT_SRC).*ADJ_WVFIELD);
%     gradient_img = sum(BACKPROJ,3);
%     % Step 2: Compute New Conjugate Gradient Search Direction from Gradient
%     % (2A) Conjugate Gradient Momentum Calculation
%     if (iter == 1) || (momentumFormula == 0)
%         beta = 0; % Gradient Descent (No Momentum)
%     else 
%         switch momentumFormula
%             case 1 % Fletcher-Reeves
%                 beta = (gradient_img(:)'*gradient_img(:)) / ...
%                     (gradient_img_prev(:)'*gradient_img_prev(:));
%             case 2 % Polak=Ribiere
%                 beta = (gradient_img(:)'*...
%                     (gradient_img(:)-gradient_img_prev(:))) / ...
%                     (gradient_img_prev(:)'*gradient_img_prev(:)); 
%             case 3 % Combined Fletcher-Reeves and Polak-Ribiere
%                 betaPR = (gradient_img(:)'*...
%                     (gradient_img(:)-gradient_img_prev(:))) / ...
%                     (gradient_img_prev(:)'*gradient_img_prev(:));
%                 betaFR = (gradient_img(:)'*gradient_img(:)) / ...
%                     (gradient_img_prev(:)'*gradient_img_prev(:));
%                 beta = min(max(betaPR,0),betaFR);
%             case 4 % Hestenes-Stiefel
%                 beta = (gradient_img(:)'*...
%                     (gradient_img(:)-gradient_img_prev(:))) / ...
%                     (search_dir(:)'*(gradient_img(:)-gradient_img_prev(:))); 
%         end
%     end
%     % (2B) Search Direction Based on Conjugate Gradient Momentum
%     search_dir = beta*search_dir-gradient_img;
%     gradient_img_prev = gradient_img;
%     % Step 3: Compute Forward Projection of Current Search Direction
%     PERTURBED_WVFIELD = solveHelmholtz(xi, yi, VEL_ESTIM, ...
%         -VIRT_SRC.*search_dir, f, a0, L_PML, false);
%     dREC_SIM = zeros(numel(tx_include), numElements);
%     for tx_elmt_idx = 1:numel(tx_include)
%         % Forward Projection of Search Direction Image
%         PERTURBED_WVFIELD_elmt = PERTURBED_WVFIELD(:,:,tx_elmt_idx);
%         dREC_SIM(tx_elmt_idx,elemInclude(tx_include(tx_elmt_idx),:)) = ...
%             PERTURBED_WVFIELD_elmt(ind(elemInclude(tx_include(tx_elmt_idx),:)));
%     end
%     % Step 4: Perform a Linear Approximation of Exact Line Search
%     switch stepSizeCalculation
%         case 1 % Not Involving Gradient Nor Search Direction
%             stepSize = real(dREC_SIM(:)'*(REC_DATA(:)-REC_SIM(:))) / ...
%                 (dREC_SIM(:)'*dREC_SIM(:)); 
%             % REVIEW STEP SIZE CALC TO EXPLAIN WHY REAL() IS USED HERE
%         case 2 % Involving Gradient BUT NOT Search Direction
%             stepSize = (gradient_img(:)'*gradient_img(:)) / ...
%                 (dREC_SIM(:)'*dREC_SIM(:));
%         case 3 % Involving Gradient AND Search Direction
%             stepSize = -(gradient_img(:)'*search_dir(:)) / ...
%                 (dREC_SIM(:)'*dREC_SIM(:));
% 
%     end
%     SLOW_ESTIM = SLOW_ESTIM + stepSize * search_dir;
%     VEL_ESTIM = 1./real(SLOW_ESTIM); % Wave Velocity Estimate [m/s]
%     % Visualize Reconstructed Solution
%     subplot(2,2,1); imagesc(x,y,C,crange);
%     title('True Sound Speed [m/s]'); axis image;
%     xlabel('Lateral [m]'); ylabel('Axial [m]'); colorbar; colormap gray;
%     subplot(2,2,2); imagesc(xi,yi,VEL_ESTIM,crange);
%     title(['Estimated Sound Speed ', num2str(iter)]); axis image;
%     xlabel('Lateral [m]'); ylabel('Axial [m]'); colorbar; colormap gray;
%     subplot(2,2,3); imagesc(xi,yi,search_dir)
%     xlabel('Lateral [m]'); ylabel('Axial [m]'); axis image;
%     title(['Search Direction Iteration ', num2str(iter)]); colorbar; colormap gray; 
%     subplot(2,2,4); imagesc(xi,yi,-gradient_img)
%     xlabel('Lateral [m]'); ylabel('Axial [m]'); axis image;
%     title(['Gradient Iteration ', num2str(iter)]); colorbar; colormap gray; 
%     drawnow; disp(['Iteration ', num2str(iter)]); toc;
% end