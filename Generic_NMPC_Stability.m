%% Nonlinear MPC with guaranteed stability
% NMPC controller by Zehua Jia, jiazehua@sjtu.edu.cn

%% System description (From Chen et al. IJACSP 2003 https://doi.org/10.1002/acs.731, and 
% Chen et al. Automatica 1998 https://doi.org/10.1016/S0005-1098(98)00073-9)

% x1' = x2 + u * (mu + (1 - mu) * x1)
% x2' = x1 + u * (mu + 4 * (1 - mu) * x2)
% mu = 0.9

% The designed local state-feeback control law is u = K * x.

%% Initiallization
clear all
close all
clc
%% Problem formulation
dt = 0.1; % The sampling period
N = 15; % The prediction horizon N * dt
T = 100; % The simulation time T * dt

Q = diag([0.5, 0.5]);
R = 1;
mu = 0.9;

%% Initial settings
x0 = [2; -3]; % x0
uM = 2; % The constraint of u for set M of the LDI
xM = 2; % The constraint of x for set M of the LDI
alphaM = -1;
% Define the variables storing the actual states and controls
xc = zeros(2, T);
uc = zeros(1, T);
xc(:, 1) = x0;
%% Get the enlarged terminal region and terminal penalty
[P, K, alpha] = NMPC_get_max_terminal (mu, Q, R, uM, xM, alphaM); % alphaM is the given upper bound of alpha. -1 means no given bound.
pause(1)
%% MPC Optimization problem using YALMIP
for i = 1 : T-N
    % Define decision variables
    x = sdpvar(2, N);
    u = sdpvar(1, N);
    x(:,1) = x0;
    % Define constraints
    const = [x <= 4, x >= -4, u <= 2, u>= -2];
    for k = 1 : N-1
    const = [const, x(:,k+1) == x(:, k) + dt * ( [x(2, k); x(1, k)] + u(k) * [mu + (1 - mu) * x(1, k); mu - 4 * (1 - mu) * x(2, k)] )];
    end
    const = [const, x(:, N)' * P * x(:, N) <= alpha];
    % Define objective
    obj = 0;
    for j = 1 : N-1
        obj = obj + x(:, j)' * Q * x(:, j) + u(j)' * R * u(j);
    end
    obj = obj + x(:, N)' * P * x(:, N);
    % Optimization
    optimize(const, obj, sdpsettings('solver','fmincon'));
    % Control and updates
    if i == 1
        xc_com = value(x); % Store the optimized states at the first time instant
        uc_com = value(u); % Store the optimized inputs at the first time instant
    end
    uc(i) = value(u(1));
    xc(:, i+1) = xc(:, i) + dt * ( [xc(2, i); xc(1, i)] + uc(i) * [mu + (1 - mu) * xc(1, i); mu - 4 * (1 - mu) * xc(2, i)] );
    x0 = xc(:, i+1);
    if i == T-N
        uc(i:i+N-1) = value(u);
        xc(:, i+1:i+N) = value(x);
    end
    i % Show the current step
end

%% Plot the trajectory
t = 0: dt: (T-1) * dt;
t_com = 0 : dt: (N-1) * dt
figure(1)
plot(t, xc(1,:), 'k', t, xc(2,:), 'b')
hold on
plot(t_com, xc_com(1, :),'r')
hold on
plot(t_com, xc_com(2, :),'g')


figure(2)
plot(t, uc(:))
hold on
plot(t_com, uc_com(:), 'r')

figure(3)
plot(xc(1, :),xc(2, :),'*')
hold on
draw_ellip(P, alpha, 'g')
hold on
plot(xc_com(1, :),xc_com(2, :),'r*')