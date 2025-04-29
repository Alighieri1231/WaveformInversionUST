function wvfield = solveHelmholtz(x, y, vel, src, f, a0, L_PML, adjoint)
%SOLVEHELMHOLTZFAST Solve Helmholtz Equation with PML
%   wvfield = solveHelmholtz(x, y, vel, src, f, signConvention, a0, L_PML, adjoint)
%   INPUTS:
%       x, y = 1xN and 1xM arrays of x and y grid positions, respectively
%       vel = MxN array of wave velocities [length/time]
%       src = MxN array of sources
%       f = Wave Frequency [1/time]
%       a0 = PML strength parameter from Chen/Cheng/Feng/Wu 2013 Paper
%       L_PML = Length [length] of PML
%       adjoint = true if solving adjoint Helmholtz; 
%                 false if solving normal Helmholtz
%   OUTPUTS:
%       wvfield = MxN array of solved wavefield vs space

% Extract Grid Information
signConvention = -1; % -1 for exp(-ikr), +1 for exp(+ikr)
h = mean(diff(x)); % Grid Spacing in X [length]
gh = mean(diff(y)); g = gh/h; % Grid Spacing in Y [length]
xmin = min(x); xmax = max(x); % [length]
ymin = min(y); ymax = max(y); % [length]
Nx = numel(x); Ny = numel(y); % Grid Points in X and Y
    
% Calculate Complex Wavenumber
k = 2*pi*f./vel; % Wavenumber [1/m]

% Generate Functions for PML
xe = linspace(xmin, xmax, 2*(Nx-1)+1);
ye = linspace(ymin, ymax, 2*(Ny-1)+1);
[Xe, Ye] = meshgrid(xe,ye);
xctr = (xmin+xmax)/2; xspan = (xmax-xmin)/2;
yctr = (ymin+ymax)/2; yspan = (ymax-ymin)/2;
sx = 2*pi*a0*f*((max(abs(Xe-xctr)-xspan+L_PML,0)/L_PML).^2);
sy = 2*pi*a0*f*((max(abs(Ye-yctr)-yspan+L_PML,0)/L_PML).^2);
ex = 1+1i*sx*sign(signConvention)/(2*pi*f); 
ey = 1+1i*sy*sign(signConvention)/(2*pi*f);
A = ey./ex; A = A(1:2:end,2:2:end);
B = ex./ey; B = B(2:2:end,1:2:end);
C = ex.*ey; C = C(1:2:end,1:2:end);

% Linear Indexing Into Sparse Array
lin_idx = @(x_idx, y_idx) y_idx+Ny*(x_idx-1); val_idx = 1;

% Optimal Stencil Parameters
[b, d, e] = stencilOptParams(min(vel(:)),max(vel(:)),f,h,g);


% Structures to Form Sparse Matrices
rows = zeros(9*(Nx-2)*(Ny-2)+(Nx*Ny-(Nx-2)*(Ny-2)),1);
cols = zeros(9*(Nx-2)*(Ny-2)+(Nx*Ny-(Nx-2)*(Ny-2)),1);
vals = zeros(9*(Nx-2)*(Ny-2)+(Nx*Ny-(Nx-2)*(Ny-2)),1);

% Populate Sparse Matrix Structures
for x_idx = 1:Nx
    for y_idx = 1:Ny
        % Image of Stencil for Helmholtz Equation
        if ((x_idx == 1) || (x_idx == Nx) || (y_idx == 1) || (y_idx == Ny))
            % Dirichlet Boundary Condition
            rows(val_idx) = lin_idx(x_idx,y_idx); 
            cols(val_idx) = lin_idx(x_idx,y_idx);
            vals(val_idx) = 1; 
            val_idx = val_idx + 1;
        else
            % 9-Point Stencil 
            % Center of Stencil
            rows(val_idx) = lin_idx(x_idx,y_idx); 
            cols(val_idx) = lin_idx(x_idx,y_idx);
            vals(val_idx) = (1-d-e)*C(y_idx,x_idx)*(k(y_idx,x_idx)^2) ...
                - b*(A(y_idx,x_idx)+A(y_idx,x_idx-1) + ...
                B(y_idx,x_idx)/(g^2)+B(y_idx-1,x_idx)/(g^2))/(h^2);
            val_idx = val_idx + 1;
            % Left
            rows(val_idx) = lin_idx(x_idx,y_idx);  
            cols(val_idx) = lin_idx(x_idx-1,y_idx);
            vals(val_idx) = (b*A(y_idx,x_idx-1) - ...
                ((1-b)/2)*(B(y_idx,x_idx-1)/(g^2)+B(y_idx-1,x_idx-1)/(g^2)))/(h^2) + ...
                (d/4)*C(y_idx,x_idx-1)*(k(y_idx,x_idx-1)^2);
            val_idx = val_idx + 1;
            % Right
            rows(val_idx) = lin_idx(x_idx,y_idx); 
            cols(val_idx) = lin_idx(x_idx+1,y_idx);
            vals(val_idx) = (b*A(y_idx,x_idx) - ...
                ((1-b)/2)*(B(y_idx,x_idx+1)/(g^2)+B(y_idx-1,x_idx+1)/(g^2)))/(h^2) + ...
                (d/4)*C(y_idx,x_idx+1)*(k(y_idx,x_idx+1)^2);
            val_idx = val_idx + 1;
            % Down
            rows(val_idx) = lin_idx(x_idx,y_idx);  
            cols(val_idx) = lin_idx(x_idx,y_idx-1);
            vals(val_idx) = (b*B(y_idx-1,x_idx)/(g^2) - ...
                ((1-b)/2)*(A(y_idx-1,x_idx)+A(y_idx-1,x_idx-1)))/(h^2) + ...
                (d/4)*C(y_idx-1,x_idx)*(k(y_idx-1,x_idx)^2);
            val_idx = val_idx + 1;
            % Up
            rows(val_idx) = lin_idx(x_idx,y_idx); 
            cols(val_idx) = lin_idx(x_idx,y_idx+1);
            vals(val_idx) = (b*B(y_idx,x_idx)/(g^2) - ...
                ((1-b)/2)*(A(y_idx+1,x_idx)+A(y_idx+1,x_idx-1)))/(h^2) + ...
                (d/4)*C(y_idx+1,x_idx)*(k(y_idx+1,x_idx)^2);
            val_idx = val_idx + 1;
            % Bottom Left
            rows(val_idx) = lin_idx(x_idx,y_idx); 
            cols(val_idx) = lin_idx(x_idx-1,y_idx-1);
            vals(val_idx) = (((1-b)/2)*(A(y_idx-1,x_idx-1)+B(y_idx-1,x_idx-1)/(g^2)))/(h^2) + ...
                (e/4)*C(y_idx-1,x_idx-1)*(k(y_idx-1,x_idx-1)^2);
            val_idx = val_idx + 1;
            % Bottom Right
            rows(val_idx) = lin_idx(x_idx,y_idx); 
            cols(val_idx) = lin_idx(x_idx+1,y_idx-1);
            vals(val_idx) = (((1-b)/2)*(A(y_idx-1,x_idx)+B(y_idx-1,x_idx+1)/(g^2)))/(h^2) + ...
                (e/4)*C(y_idx-1,x_idx+1)*(k(y_idx-1,x_idx+1)^2);
            val_idx = val_idx + 1;
            % Top Left
            rows(val_idx) = lin_idx(x_idx,y_idx); 
            cols(val_idx) = lin_idx(x_idx-1,y_idx+1);
            vals(val_idx) = (((1-b)/2)*(A(y_idx+1,x_idx-1)+B(y_idx,x_idx-1)/(g^2)))/(h^2) + ...
                (e/4)*C(y_idx+1,x_idx-1)*(k(y_idx+1,x_idx-1)^2);
            val_idx = val_idx + 1;
            % Top Right
            rows(val_idx) = lin_idx(x_idx,y_idx); 
            cols(val_idx) = lin_idx(x_idx+1,y_idx+1);
            vals(val_idx) = (((1-b)/2)*(A(y_idx+1,x_idx)+B(y_idx,x_idx+1)/(g^2)))/(h^2) + ...
                (e/4)*C(y_idx+1,x_idx+1)*(k(y_idx+1,x_idx+1)^2);
            val_idx = val_idx + 1;
        end
    end
end

disp(size(rows))
disp(size(cols))
disp(size(vals))
disp(class(vals))
disp(Nx)
disp(Ny)
disp(size(src))
disp(numel(src)/(Nx*Ny))
% Generate Left-Hand Side of Sparse Array
HelmholtzEqn = sparse(rows, cols, vals, Nx*Ny, Nx*Ny);
disp(class(HelmholtzEqn))

% Solve the Helmholtz Equation - Brute-force CPU solution of linear system
if adjoint
    sol = (HelmholtzEqn')\reshape(src,[Nx*Ny, numel(src)/(Nx*Ny)]);
else
    sol = HelmholtzEqn\reshape(src,[Nx*Ny, numel(src)/(Nx*Ny)]);
end
wvfield = reshape(sol, size(src));
disp('1')
disp(class(wvfield))
end


function [b,d,e] = stencilOptParams(vmin,vmax,f,h,g)
%STENCILOPTPARAMS Optimal Params for 9-Point Stencil 
%   INPUTS:
%       vmin = minimum wave velocity [L/T]
%       vmax = maximum wave velocity [L/T]
%       f = frequency [1/T]
%       h = grid spacing in X [L]
%       g = (grid spacing in Y [L])/(grid spacing in X [L])
%   OUTPUTS:
%       b, d, e = optimal params according to Chen/Cheng/Feng/Wu 2013 Paper

l = 100; r = 10;
Gmin = vmin/(f*h); Gmax = vmax/(f*h);

m = 1:l; n = 1:r;
theta = (m-1)*pi/(4*(l-1));
G = 1./(1/Gmax + ((n-1)/(r-1))*(1/Gmin-1/Gmax));

[TH, GG] = meshgrid(theta, G);

P = cos(g*2*pi*cos(TH)./GG);
Q = cos(2*pi*sin(TH)./GG);

S1 = (1+1/(g^2))*(GG.^2).*(1-P-Q+P.*Q);
S2 = (pi^2)*(2-P-Q);
S3 = (2*pi^2)*(1-P.*Q);
S4 = 2*pi^2 + (GG.^2).*((1+1/(g^2))*P.*Q-P-Q/(g^2));

fixB = true;
if fixB
    b = 5/6; % Fix the Value to 5/6 based on Laplacian Derived by Robert E. Lynch
    A = [S2(:), S3(:)]; y = S4(:)-b*S1(:);
    params = (A'*A)\(A'*y);
    d = params(1); e = params(2);
else
    A = [S1(:), S2(:), S3(:)]; y = S4(:);
    params = (A'*A)\(A'*y);
    b = params(1); d = params(2); e = params(3);
end

end

