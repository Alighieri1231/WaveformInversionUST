clear
clc

%% Simulate Series of Masses and Springs

% Create series of masses and springs
N = 19; % Number of masses
x = (0:N+1)/(N+1); % Resting Locations of Masses (and Walls at the Ends)
masses = rand(N,1)+10; % Random unknown masses
springs = rand(N+1,1)+10; % springs between masses and walls

% Create Time Axis
dt = 0.25; % Time Step (should be less than sqrt(m/k))
Nt = 401; % Number of Time Points
t = (0:Nt-1)*dt; % Time Axis

% Random Force Input
force = zeros(N, Nt);
force(:,1) = [1; zeros(N-1,1)];

% Forward Simulation
u = simulate(t, force, masses, springs);
% for t_idx = 1:Nt
%     stem(x, [0; u(:,t_idx); 0]); ylim([-1,1]*0.05); drawnow;
% end

% Only Measure the Displacement of One Mass
meas_idx = 1:N;
umeas = u(meas_idx, :);


%% Inverse-Problem: What were the masses and springs in the simulation?

% Parameters for Conjugate Gradient Reconstruction
Niter = 300; % Number of Iterations
momentumFormula = 1; % Momentum Formula for Conjugate Gradient
                     % 0 -- No Momentum (Gradient Descent)
                     % 1 -- Fletcher-Reeves (FR)
                     % 2 -- Polak-Ribiere (PR)
                     % 3 -- Combined FR + PR
                     % 4 -- Hestenes-Stiefel (HS)
stepSizeCalculation = 1; % Which Step Size Calculation:
                         % 1 -- Not Involving Gradient Nor Search Direction
                         % 2 -- Involving Gradient BUT NOT Search Direction
                         % 3 -- Involving Gradient AND Search Direction

% Initial Guess for Masses and Springs
masses_estim = 10*ones(N,1);
springs_estim = 10*ones(N+1,1);

% (Nonlinear) Conjugate Gradient
flatten = @(v) v(:);
search_dir = zeros(2*N+1,1); % Conjugate Gradient Direction
grad_prev = zeros(2*N+1,1); % Previous Gradient Image
for iter = 1:Niter
    % Step 1: Calculate Gradient/Backprojection
    % (1A) Forward Simulation
    usim_full = simulate(t, force, masses_estim, springs_estim);
    usim = usim_full(meas_idx, :);
    % (1B) Virtual Sources
    virtSrcMasses = virtualSourceMasses(usim_full, t, masses_estim);
    virtSrcSprings = virtualSourceSprings(usim_full, t, springs_estim);
    % (1C) Build Adjoint Sources
    adj_force = zeros(N, Nt); adj_force(meas_idx,:) = usim - umeas;
    % (1D) Solve Adjoint Wave Equation
    adj_sim_full = adjoint_sim(t, adj_force, masses_estim, springs_estim);
    % (1E) Calculate Gradient/Backprojection
    grad_masses = -sum(conj(virtSrcMasses).*adj_sim_full,2);
    grad_springs = -flatten(sum(conj(virtSrcSprings).*adj_sim_full,[1,2]));
    grad = [grad_masses; grad_springs];
    % Step 2: Compute New Conjugate Gradient Search Direction from Gradient
    % (2A) Conjugate Gradient Momentum Calculation
    if (iter == 1) || (momentumFormula == 0)
        beta = 0; % Gradient Descent (No Momentum)
    else 
        switch momentumFormula
            case 1 % Fletcher-Reeves
                beta = (grad(:)'*grad(:)) / ...
                    (grad_prev(:)'*grad_prev(:));
            case 2 % Polak=Ribiere
                beta = (grad(:)'*(grad(:)-grad_prev(:))) / ...
                    (grad_prev(:)'*grad_prev(:)); 
            case 3 % Combined Fletcher-Reeves and Polak-Ribiere
                betaPR = (grad(:)'*(grad(:)-grad_prev(:))) / ...
                    (grad_prev(:)'*grad_prev(:));
                betaFR = (grad(:)'*grad(:)) / ...
                    (grad_prev(:)'*grad_prev(:));
                beta = min(max(betaPR,0),betaFR);
            case 4 % Hestenes-Stiefel
                beta = (grad(:)'*(grad(:)-grad_prev(:))) / ...
                    (search_dir(:)'*(grad(:)-grad_prev(:))); 
        end
    end
    % (2B) Search Direction Based on Conjugate Gradient Momentum
    search_dir = beta*search_dir-grad;
    search_dir_masses = search_dir(1:N);
    search_dir_springs = search_dir(N+1:end);
    grad_prev = grad;
    % Step 3: Compute Forward Projection of Current Search Direction
    pertSrcMasses = -virtSrcMasses.*search_dir_masses;
    pertSrcSprings = -sum(virtSrcSprings .* ...
        reshape(search_dir_springs,[1,1,N+1]),3);
    pertSrc = pertSrcMasses + pertSrcSprings;
    dusim_full = simulate(t, pertSrc, masses_estim, springs_estim);
    dusim = dusim_full(meas_idx, :);
    % Step 4: Perform a Linear Approximation of Exact Line Search
    switch stepSizeCalculation
        case 1 % Not Involving Gradient Nor Search Direction
            stepSize = (dusim(:)'*(umeas(:)-usim(:))) / ...
                (dusim(:)'*dusim(:)); 
            % REVIEW STEP SIZE CALC TO EXPLAIN WHY REAL() IS USED HERE
        case 2 % Involving Gradient BUT NOT Search Direction
            stepSize = (grad(:)'*grad(:)) / ...
                (dusim(:)'*dusim(:));
        case 3 % Involving Gradient AND Search Direction
            stepSize = -(grad(:)'*search_dir(:)) / ...
                (dusim(:)'*dusim(:));
    end
    masses_estim = masses_estim + stepSize * search_dir_masses;
    springs_estim = springs_estim + stepSize * search_dir_springs;
    % Visualize Reconstructed Solution
    subplot(2,2,1); plot(x(2:end-1), masses, 'k', ...
        x(2:end-1), masses_estim, 'r', 'LineWidth', 2);
    xlabel('Location'); ylabel('Masses'); 
    title(['Masses Iteration ', num2str(iter)]);
    subplot(2,2,2); plot((x(1:end-1)+x(2:end))/2, springs, 'k', ...
        (x(1:end-1)+x(2:end))/2, springs_estim, 'r', 'LineWidth', 2); 
    xlabel('Location'); ylabel('Spring Constants');
    title(['Spring Constants Iteration ', num2str(iter)]);
    subplot(2,2,3); imagesc(t, x(meas_idx), umeas); 
    title('Measured Signal'); xlabel('Time'); ylabel('Location'); 
    subplot(2,2,4); imagesc(t, x(meas_idx), usim); 
    title('Simulated Signal'); xlabel('Time'); ylabel('Location'); drawnow;
end

%% Differential Equations

% Forward Differential Equation: Forward-Time Propagation
%   F_i - k_i(u_i - u_{i-1}) - k_{i+1}(u_i - u_{i+1}) = m_i(d2u_i)
function u = simulate(t, F, mass, springs)
    % Inputs
    Nt = numel(t); % Number of Time Points
    dt = mean(diff(t)); % Time Step
    N = numel(springs)-1; % Number of Masses
    % Displacement of Each Mass (Walls Clamped to Zero)
    u = zeros(N+2, Nt); % Displacements vs Time
    % Time Advancement Loop
    for t_idx = 1:Nt-1
        if t_idx == 1
            u(2:end-1,t_idx+1) = 2*u(2:end-1,t_idx) + (dt^2) * ...
                (F(:,t_idx) - springs(1:end-1).*(u(2:end-1,t_idx)-u(1:end-2,t_idx)) - ...
                springs(2:end).*(u(2:end-1,t_idx)-u(3:end,t_idx))) ./ mass;
        else
            u(2:end-1,t_idx+1) = 2*u(2:end-1,t_idx) - u(2:end-1,t_idx-1) + (dt^2) * ...
                (F(:,t_idx) - springs(1:end-1).*(u(2:end-1,t_idx)-u(1:end-2,t_idx)) - ...
                springs(2:end).*(u(2:end-1,t_idx)-u(3:end,t_idx))) ./ mass;
        end
    end
    % Remove Walls
    u = u(2:end-1,:);
end

% Adjoint Differential Equation: Reverse-Time Propagation
%   F_i - k_i(u_i - u_{i-1}) - k_{i+1}(u_i - u_{i+1}) = m_i(d2u_i)
function u = adjoint_sim(t, F, mass, springs)
    % Inputs
    Nt = numel(t); % Number of Time Points
    dt = mean(diff(t)); % Time Step
    N = numel(springs)-1; % Number of Masses
    % Displacement of Each Mass (Walls Clamped to Zero)
    u = zeros(N+2, Nt); % Displacements vs Time
    % Time Advancement Loop
    for t_idx = Nt:-1:2
        if t_idx == Nt
            u(2:end-1,t_idx-1) = 2*u(2:end-1,t_idx) + (dt^2) * ...
                (F(:,t_idx) - springs(1:end-1).*(u(2:end-1,t_idx)-u(1:end-2,t_idx)) - ...
                springs(2:end).*(u(2:end-1,t_idx)-u(3:end,t_idx))) ./ mass;
        else
            u(2:end-1,t_idx-1) = 2*u(2:end-1,t_idx) - u(2:end-1,t_idx+1) + (dt^2) * ...
                (F(:,t_idx) - springs(1:end-1).*(u(2:end-1,t_idx)-u(1:end-2,t_idx)) - ...
                springs(2:end).*(u(2:end-1,t_idx)-u(3:end,t_idx))) ./ mass;
        end
    end
    % Remove Walls
    u = u(2:end-1,:);
end

%% Virtual Sources

% Virtual Source for Mass Estimation
function virtSrc = virtualSourceMasses(u, t, mass)
    % Inputs
    Nt = numel(t); % Number of Time Points
    dt = mean(diff(t)); % Time Step
    N = numel(mass); % Number of Masses
    % Displacement of Each Mass (Walls Clamped to Zero)
    virtSrc = zeros(N,Nt);
    virtSrc(:,1) = (-2*u(:,1)+u(:,2)) / (dt^2);
    virtSrc(:,2:Nt-1) = (u(:,1:Nt-2)-2*u(:,2:Nt-1)+u(:,3:Nt)) / (dt^2);
    virtSrc(:,Nt) = (u(:,Nt-1)-2*u(:,Nt)) / (dt^2);
end

% Virtual Source for Spring Estimation
function virtSrc = virtualSourceSprings(u, t, springs)
    % Inputs
    Nt = numel(t); % Number of Time Points
    dt = mean(diff(t)); % Time Step
    N = numel(springs)-1; % Number of Masses
    % Displacement of Each Mass (Walls Clamped to Zero)
    virtSrc = zeros(N,Nt,N+1);
    virtSrc(1,:,1) = u(1,:);
    for mass_idx = 1:N
        if mass_idx == 1
            virtSrc(mass_idx,:,mass_idx) = u(mass_idx,:);
        else
            virtSrc(mass_idx,:,mass_idx) = u(mass_idx,:)-u(mass_idx-1,:);
        end
        if mass_idx == N
            virtSrc(mass_idx,:,mass_idx+1) = u(mass_idx,:);
        else
            virtSrc(mass_idx,:,mass_idx+1) = u(mass_idx,:)-u(mass_idx+1,:);
        end
    end
end

%% Adjoint Test - Correctness of Forward/Adjoint Simulators

% Random Test Inputs
force_f = randn(N, Nt);
force_a = randn(N, Nt);

% Forward and Adjoint Simulations
u_f = simulate(t, force_f, masses, springs);
u_a = adjoint_sim(t, force_a, masses, springs);

% Adjoint Test
rel_err = (force_f(:)'*u_a(:) - force_a(:)'*u_f(:)) ./ ...
    (force_f(:)'*u_a(:) + force_a(:)'*u_f(:));
disp(['Adjoint Test Relative Error = ', num2str(rel_err)]);

%% Is Adjoint Simulation Just a Time-Reversed Forward Simulation?

% Random Test Inputs
force_f = randn(N, Nt);
force_a = randn(N, Nt);

% Forward and Adjoint Simulations
reverse_time = @(u) u(:,end:-1:1);
u_f = simulate(t, force_f, masses, springs);
u_a = reverse_time(simulate(t, reverse_time(force_a), masses, springs));

% Adjoint Test
rel_err = (force_f(:)'*u_a(:) - force_a(:)'*u_f(:)) ./ ...
    (force_f(:)'*u_a(:) + force_a(:)'*u_f(:));
disp(['Adjoint Test Relative Error = ', num2str(rel_err)]);