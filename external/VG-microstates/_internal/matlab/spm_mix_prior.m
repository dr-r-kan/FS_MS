function [mix] = spm_mix_prior(y, m, verbose, solution)
% Fit a multivariate Gaussian mixture model using VB, seeded by a solution.
% FORMAT [mix] = spm_mix_prior(y, m, verbose, solution)
%
% This is a local copy of SPM's spm_mix with the k-means initialiser replaced
% by a caller-supplied cluster solution. The VB update loop is intentionally
% kept the same as the SPM routine.
%
% Based on spm_mix by Will Penny, Copyright (C) 2007-2022 Wellcome Centre
% for Human Neuroimaging.
%
% solution.means  [m x d] initial component means in the same feature space
% solution.labels [N x 1] optional labels used to initialise priors/covariances
% solution.priors [1 x m] optional prior proportions
% solution.covs   [d x d x m] optional component covariances
% solution.mean_weight optional component-mean prior weight (default 1)

if nargin < 3 || isempty(verbose)
    verbose = 1;
end
if nargin < 4 || isempty(solution)
    error('spm_mix_prior requires a cluster solution.');
end

[N, d] = size(y);

% Put model info into data structure format
mix.nin = d;
mix.m = m;

% PRIORS
% Set mixing priors
lambda_0 = 1;
% Set mean priors
m_0 = mean(y);
beta_0 = 1;
% Set precision priors
% These settings set the prior covariance matrix to be
% equal to the 0.01*d*identity matrix - if original data is
% zmuv, this is reasonable
a_0 = d;
B_0 = 0.01 * d * eye(d);

% Also, store in data structure ready for output
mix.prior.lambda_0 = lambda_0;
mix.prior.m_0 = m_0;
mix.prior.beta_0 = beta_0;
mix.prior.a_0 = a_0;
mix.prior.B_0 = B_0;

sd = 0;
for i = 1:d
  sd = sd + psi((a_0 + 1 - i) / 2);
end
log_tilde_gamma_0 = sd - log(det(B_0)) + d * log(2); %#ok<NASGU>

if (m == 1)
    % For a single component we just have a Gaussian model

    % Set posteriors
    state(1).a = N + a_0;
    state(1).beta = N + beta_0;
    state(1).m = m_0;
    Cy = cov(y);
    state(1).B = N * Cy + B_0;

    fm = spm_lg_gamma(d, 0.5 * state(1).a);
    fm = fm - spm_lg_gamma(d, 0.5 * a_0);
    fm = fm - 0.5 * d * N * log(pi);
    fm = fm + 0.5 * d * (log(beta_0) - log(state(1).beta));
    fm = fm + 0.5 * a_0 * log(det(B_0));
    fm = fm - 0.5 * state(1).a * log(det(state(1).B));

    mix.m = m;
    mix.fm = fm;
    state(1).C = state(1).B / state(1).a;
    mix.state = state;

    mix.priors = 1;
    return;
end

% Initialise posterior from the supplied cluster solution instead of k-means.
[priors, means, covs] = spm_mix_prior_initial_solution(y, m, solution);
[m_0, beta_0] = spm_mix_prior_mean_prior(solution, means, m);
mix.prior.m_0 = m_0;
mix.prior.beta_0 = beta_0;

% Match SPM's small count offset for ML proportions.
priors = priors + (1 / N);
lambda = priors;
for s = 1:m
  state(s).m = means(s, :);
  state(s).beta = priors(s) * N + beta_0(s);
  state(s).a = priors(s) * N + a_0;
  Cov = covs(:, :, s);
  % If component cov is rank deficient set to prior
  if rank(Cov) < size(Cov, 1)
    state(s).B = B_0;
  else
    state(s).B = priors(s) * N * Cov;
  end
  N_bar(s) = priors(s) * N; %#ok<AGROW>
end

% Start algorithm
lik = [];
tol = 0.0001;
max_loops = 32;
for loops = 1:max_loops

    % E-step
    lambda_tot = sum(lambda);
    for s = 1:m
        state(s).bar_gamma = state(s).a * inv(state(s).B);
        log_tilde_pi(s) = psi(lambda(s)) - psi(lambda_tot); %#ok<AGROW>
        sd = 0;
        for i = 1:d
            sd = sd + psi((state(s).a + 1 - i) / 2);
        end
        log_tilde_gamma(s) = sd - log(det(state(s).B)) + d * log(2); %#ok<AGROW>

        tilde_pi(s) = exp(log_tilde_pi(s)); %#ok<AGROW>
        tilde_gamma(s) = exp(log_tilde_gamma(s)); %#ok<AGROW>
        for n = 1:N
            gamma(s, n) = tilde_pi(s) * tilde_gamma(s)^0.5; %#ok<AGROW>
            dy = (y(n, :) - state(s).m);
            gamma(s, n) = gamma(s, n) * (exp(-0.5 * dy * state(s).bar_gamma * dy') + eps) * exp(-d / (2 * state(s).beta));
        end
    end

    gamma_n = sum(gamma);
    for s = 1:m
        if mean(gamma_n) > eps
            % If component still exists
            gamma(s, :) = gamma(s, :) ./ gamma_n;
        end
    end

    % M-step

    % Part-I
    for s = 1:m
        pi_bar(s) = mean(gamma(s, :)) + eps; %#ok<AGROW>
        N_bar(s) = N * pi_bar(s) + eps;
        bar_mu(s, :) = (1 / N_bar(s)) * sum(gamma(s, :)' * ones(1, d) .* y); %#ok<AGROW>

    end

    % get weighted means and covariances
    for s = 1:m
        state(s).bar_sigma = zeros(d, d);
        for n = 1:N
            dy = y(n, :) - bar_mu(s, :);
            state(s).bar_sigma = state(s).bar_sigma + gamma(s, n) .* (dy' * dy);
        end
        state(s).bar_sigma = (state(s).bar_sigma) / N_bar(s);
    end

    % Now compute the free energy
    f1 = -spm_kl_dirichlet(lambda, lambda_0 * ones(1, m), log_tilde_pi);
    for s = 1:m
        f2(s) = -spm_kl_wishart(state(s).a, state(s).B, a_0, B_0); %#ok<AGROW>

        % KL-method for computing f3(s)
        Cs = state(s).B / (state(s).beta * state(s).a);
        beta_0_s = beta_0(s);
        m_0_s = m_0(s, :);
        C0 = state(s).B / (beta_0_s * state(s).a);
        f3(s) = -spm_kl_normal(state(s).m, Cs, m_0_s, C0); %#ok<AGROW>

        % Check f3(s)
        check_f3 = -0.5 * d * (log(state(s).beta) - log(beta_0_s) + (beta_0_s / state(s).beta) - 1);
        dm = state(s).m - m_0_s;
        check_f3 = check_f3 - 0.5 * dm * beta_0_s * state(s).a * inv(state(s).B) * dm';
        f3(s) = check_f3;

        f4(s) = N_bar(s) * log_tilde_pi(s) - sum(gamma(s, :) .* log(gamma(s, :) + eps)); %#ok<AGROW>
        LaB = 0;
        for i = 1:d
            LaB = LaB + psi((state(s).a + 1 - i) / 2);
        end
        LaB = LaB + d * log(2) - log(det(state(s).B));
        f5(s) = 0.5 * N_bar(s) * (-d * log(2 * pi) + LaB - trace(state(s).bar_gamma * state(s).bar_sigma) - (d / state(s).beta)); %#ok<AGROW>
        fkl(s) = f2(s) + f3(s) + f4(s) + f5(s); %#ok<NASGU,AGROW>
        fkl_adj(s) = f2(s) + f3(s) + (f4(s) + f5(s)) / N_bar(s); %#ok<NASGU,AGROW>
    end
    fm = f1 + sum(f2) + sum(f3) + sum(f4) + sum(f5);

    acc = sum(f4) + sum(f5);
    kl_proportions = -f1;
    kl_covs = -sum(f2);
    kl_centres = -sum(f3);

    if verbose
        disp(sprintf('Iter=%d, F1=%1.2f, F2=%1.2f, F3=%1.2f, F4=%1.2f, F5=%1.2f, Fm=%1.2f', loops, f1, sum(f2), sum(f3), sum(f4), sum(f5), fm));
    end

    % Convergence criterion
    oldlik = lik;
    lik = fm;
    if (loops > 1)
        if abs((lik - oldlik) / lik) < tol
            break;
        end
    end

    % Part-II

    for s = 1:m
        % Posterior mixing coefficients
        lambda(s) = N_bar(s) + lambda_0;
        % Posterior means
        beta_0_s = beta_0(s);
        m_0_s = m_0(s, :);
        state(s).m = (N_bar(s) * bar_mu(s, :) + beta_0_s * m_0_s) / (N_bar(s) + beta_0_s);
        state(s).beta = N_bar(s) + beta_0_s;
        % Posterior precisions
        state(s).a = N_bar(s) + a_0;
        dy = bar_mu(s, :) - m_0_s;
        state(s).B = N_bar(s) * state(s).bar_sigma + (N_bar(s) * beta_0_s * dy' * dy) / (N_bar(s) + beta_0_s) + B_0;
    end
end

% Put variables into data structure
mix.m = m;
mix.fm = fm;
mix.acc = acc;
mix.kl_proportions = kl_proportions;
mix.kl_covs = kl_covs;
mix.kl_centres = kl_centres;
mix.state = state;
mix.lambda = lambda;
mix.gamma = gamma;

% Put info into data structure
for j = 1:m
    mix.state(j).prior = pi_bar(j);
    mix.state(j).C = mix.state(j).B / mix.state(j).a;
end
end

function [priors, means, covs] = spm_mix_prior_initial_solution(y, k, solution)
    [N, d] = size(y);
    if isstruct(solution)
        if ~isfield(solution, 'means') || isempty(solution.means)
            error('Cluster solution must include means.');
        end
        means = double(solution.means);
        labels = field_or_empty(solution, 'labels');
        priors = field_or_empty(solution, 'priors');
        covs = field_or_empty(solution, 'covs');
    else
        means = double(solution);
        labels = [];
        priors = [];
        covs = [];
    end

    if size(means, 1) ~= k || size(means, 2) ~= d
        error('Cluster solution means must be %d x %d.', k, d);
    end

    if isempty(labels)
        labels = nearest_mean_labels(y, means);
    else
        labels = round(double(labels(:)));
    end
    if numel(labels) ~= N || any(labels < 1 | labels > k | ~isfinite(labels))
        error('Cluster solution labels must be finite integers in 1..%d.', k);
    end

    if isempty(priors)
        priors = zeros(1, k);
        for j = 1:k
            priors(j) = mean(labels == j);
        end
    else
        priors = double(priors(:)');
        if numel(priors) ~= k
            error('Cluster solution priors must have %d entries.', k);
        end
        priors(~isfinite(priors) | priors < 0) = 0;
        priors = priors ./ max(sum(priors), eps);
    end

    if isempty(covs)
        covs = zeros(d, d, k);
        for j = 1:k
            x = y(labels == j, :);
            if size(x, 1) > 1
                covs(:, :, j) = cov(x - ones(size(x, 1), 1) * means(j, :));
            else
                covs(:, :, j) = 0.0001 * eye(d);
            end
        end
    else
        covs = double(covs);
        if size(covs, 1) ~= d || size(covs, 2) ~= d || size(covs, 3) ~= k
            error('Cluster solution covs must be %d x %d x %d.', d, d, k);
        end
    end
end

function [mean_prior, beta] = spm_mix_prior_mean_prior(solution, means, k)
    mean_prior = means;
    weight = 1;
    if isstruct(solution)
        value = field_or_empty(solution, 'mean_weight');
        if isempty(value)
            value = field_or_empty(solution, 'prior_weight');
        end
        if ~isempty(value)
            weight = double(value);
        end
    end
    if isscalar(weight)
        beta = repmat(weight, 1, k);
    else
        beta = double(weight(:)');
        if numel(beta) ~= k
            error('Cluster solution mean_weight must be scalar or have %d entries.', k);
        end
    end
    beta(~isfinite(beta) | beta < 0) = 0;
    beta = max(beta, 1e-12);
end

function labels = nearest_mean_labels(y, means)
    d2 = (ones(size(means, 1), 1) * sum((y .^ 2)', 1))' + ...
        ones(size(y, 1), 1) * sum((means .^ 2)', 1) - ...
        2 .* (y * means');
    [~, labels] = min(d2', [], 1);
    labels = labels(:);
end

function value = field_or_empty(S, name)
    if isfield(S, name)
        value = S.(name);
    else
        value = [];
    end
end
