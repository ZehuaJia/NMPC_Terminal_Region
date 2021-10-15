function [P, K, alpha] = NMPC_get_max_terminal (mu, Q, R, ucon, xcon, alpha)
%% On the terminal region of model predictive control for non-linear systems with input/state constraints (2003)
% Get the maximal terminal region By Zehua Jia, jiazehua@sjtu.edu.cn

% Simulation reproduction of Chen et. al 2003.

%% Input
% ucon: |u| <= ucon; 
% xcon: |x| <= xcon
% Here x(i) has the same upper and lower bounds, i = 1, 2.
% alpha: x' * P * x <= alpha
%% System description
% x1' = x2 + u * (mu + (1 - mu) * x1)
% x2' = x1 + u * (mu + 4 * (1 - mu) * x2)

% The designed local state-feeback control law is u = K * x.

%% Some Tips
% A. The constraints should be Linear, which means nonlinear terms are not
% permitted (such as x*y and x^2, where x,y are both sdpvars).

% B. The inverse of a sdpvar is not desired in constraints or objective.
% logdet(X^(-1)) = -logdet(X).

% C. logdet is concave, and then -logdet is convex.

% D. One should always try to reformulate the problem to obtain a convex 
% problem. 

% E. -logdet(a * X) (sdpvar a, X = sdpvar(n,n)) can be reformulated as 
% -logdet(Y) (sdpvar a, Y = sdpvar(n,n)). Then multiply LMI constraints by 
% a in both sides, and replace Y = a * X in constraints, 

%% Initiallization
umax = ucon;
umin = -ucon;
xmax = xcon;
xmin = -xcon;
%% LDI approximation within the selected set x <= 2
N=8; % The number of LDI 
A(:,:,1) = [(1 - mu) * umin, 1; 1, -4 * (1 - mu) * umin]; A(:,:,2) = [(1 - mu) * umin, 1; 1, -4 * (1 - mu) * umin]; % minimum u
A(:,:,3) = [(1 - mu) * umin, 1; 1, -4 * (1 - mu) * umin]; A(:,:,4) = [(1 - mu) * umin, 1; 1, -4 * (1 - mu) * umin]; % minimum u
A(:,:,5) = [(1 - mu) * umax, 1; 1, -4 * (1 - mu) * umax]; A(:,:,6) = [(1 - mu) * umax, 1; 1, -4 * (1 - mu) * umax]; % maximum u
A(:,:,7) = [(1 - mu) * umax, 1; 1, -4 * (1 - mu) * umax]; A(:,:,8) = [(1 - mu) * umax, 1; 1, -4 * (1 - mu) * umax]; % maximum u
% B: (x1, x2) in (min,min),(min,max),(max,min)(max,max) with minimal u
B(:,1) = [mu + (1 - mu) * xmin; mu - 4 * (1 - mu) * xmin]; B(:,2) = [mu + (1 - mu) * xmin; mu - 4 * (1 - mu) * xmax]; 
B(:,3) = [mu + (1 - mu) * xmax; mu - 4 * (1 - mu) * xmin]; B(:,4) = [mu + (1 - mu) * xmax; mu - 4 * (1 - mu) * xmax]; 
B(:,5) = [mu + (1 - mu) * xmin; mu - 4 * (1 - mu) * xmin]; B(:,6) = [mu + (1 - mu) * xmin; mu - 4 * (1 - mu) * xmax]; 
B(:,7) = [mu + (1 - mu) * xmax; mu - 4 * (1 - mu) * xmin]; B(:,8) = [mu + (1 - mu) * xmax; mu - 4 * (1 - mu) * xmax]; % (x1, x2) in (min,min),(min,max),(max,min)(max,max) with maximal u
for i = 1:N
    F(:,:,i) = [A(:,:,i), B(:,i)];
end

%% Solve the optimization problem using YALMIP
% Express the state/Input constraints in standard form
c(1, :) = [1/xmax, 0];% for x1 < 2
c(2, :) = [0, 1/xmax];% for x2 < 2
c(3, :) = -c(1, :);% for x1 > -2
c(4, :) = -c(2, :);% for x2 > -2
c(5, :) = zeros(1, 2);
c(6, :) = zeros(1, 2);
d(1) = 0;
d(2) = 0;
d(3) = 0;
d(4) = 0;
d(5) = 1/umax;
d(6) = - d(5);
[~, Nc] = size(d); % Number of constraints

% Define decision matrix variables
[n,~] = size(Q);
[~,m] = size(R);
alpha0 = sdpvar(1);
W1 = sdpvar(n,n); % W1 = alpha * W1 (The latter W1 is the W1 in the paper)
W2 = sdpvar(m,n,'full'); % W2 = alpha * W2 (The latter W2 is the W2 in the paper)
W = [W1, W2'];
MAT1 = sdpvar(2*n+m, 2*n+m, N);
MAT2 = sdpvar(1+n, 1+n, Nc);

% Define LMI constraints
for i = 1:N
    MAT1(:,:,i) = [-F(:,:,i) * W' - W * F(:,:,i)', [W1 * Q^(0.5), W2'];
        [W1 * Q^(0.5), W2']',alpha0 * [eye(2), zeros(2,1); zeros(1,2), R^(-1)]]; % Multiplying alpha in both sides in LMI (19)
end

for i = 1:Nc
    MAT2(:,:,i) = [1, c(i, :) * W1 + d(i) * W2; (c(i, :) * W1 + d(i) * W2)', W1]; % Multiplying alpha in both sides in LMI (20)
end

const = [];
for i = 1:N
    const = [const, MAT1(:,:,i)>=0];
end

for i = 1:Nc
    const = [const, MAT2(:,:,i)>=0];
end
% const = [const, alpha <= 10]; % Without this constraint, the solver will turn out error "lack of progress"
obj = -logdet(W1);
const1 = [const, alpha0 <= alpha]; % The LMI method with limited alpha
%% Solving
if alpha == -1
    optimize(const, obj, sdpsettings('solver','sdpt3')) % When input alpha == -1, it means no selected alpha
elseif alpha <= 0
    error('The alphaM must be larger than 0')
else
    optimize(const1, obj, sdpsettings('solver','sdpt3'))
end
%% Get results
W1 = double(W1) / double(alpha0);
W2 = double(W2) / double(alpha0);
P = W1^(-1);
K = W2 * W1^(-1);
alpha = double(alpha0);

%% Plot the obtained ellipsoid (terminal region)
% draw_ellip(P, alpha, 'k')
% hold on % Used for multiple eliipses comparison
end
