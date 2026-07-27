function test_fs_spm_prior_hierarchy()
%TEST_FS_SPM_PRIOR_HIERARCHY Synthetic checks for the FS hierarchical prior path.

    setup_microstate_paths();
    rng(19, 'twister');

    % Polarity reversal and state permutation leave the projective parent set unchanged.
    parent = normalise_rows(randn(6, 8));
    child_maps = zeros(180, 8);
    for k = 1:6
        child_maps((k-1)*30 + (1:30), :) = normalise_rows(parent(k, :) + .12 * randn(30, 8));
    end
    features = projective(child_maps);
    prior_features = projective(parent);
    perm = [4 1 6 2 5 3];
    mix_a = spm_mix_prior(features, 6, 0, struct('means', prior_features, 'mean_weight', 1));
    mix_b = spm_mix_prior(features, 6, 0, struct('means', projective(-parent(perm, :)), 'mean_weight', 1));
    means_a = vertcat(mix_a.state.m);
    means_b = vertcat(mix_b.state.m);
    similarity = abs(corr(means_a', means_b'));
    assert(min(max(similarity, [], 2)) > .95, 'Parent permutation/polarity changed recovered state identity.');

    % The mixture contains exactly the real observations and beta_0 is one.
    recovered_N = sum([mix_a.state.beta]) - sum(mix_a.prior.beta_0);
    assert(abs(recovered_N - size(features, 1)) < 1e-6, 'Parent information was counted as observations.');
    assert(all(abs(mix_a.prior.beta_0 - 1) < eps), 'beta_0 is not one for every state.');

    % Strong child evidence moves means away from the parent.
    y = [randn(120, 2) * .10 + [1.5 0]; randn(120, 2) * .10 + [5.5 0]];
    strong = spm_mix_prior(y, 2, 0, struct('means', [1 0; 5 0], 'mean_weight', 1));
    strong_means = sortrows(vertcat(strong.state.m), 1);
    assert(all(abs(strong_means(:, 1) - [1.5; 5.5]) < .25), 'Strong child evidence did not dominate the mean prior.');

    % A weak child is stabilised, but the posterior does not reproduce the parent exactly.
    weak_y = [randn(5, 2) * .15 + [1 0]; randn(5, 2) * .15 + [4 0]];
    weak_parent = [.7 0; 4.3 0];
    weak = spm_mix_prior(weak_y, 2, 0, struct('means', weak_parent, 'mean_weight', 1));
    weak_means = sortrows(vertcat(weak.state.m), 1);
    assert(all(abs(weak_means(:, 1) - weak_parent(:, 1)) > 1e-4), 'Weak fit copied the parent exactly.');
    assert(all(abs(weak_means(:, 1) - [1; 4]) < .5), 'Weak fit was not stabilised near the generating states.');

    % A generating-map parent improves small-sample recovery relative to a bad parent.
    small_y = [randn(4, 2) * .20 + [1 0]; randn(4, 2) * .20 + [4 0]];
    good = spm_mix_prior(small_y, 2, 0, struct('means', [1 0; 4 0], 'mean_weight', 1));
    bad = spm_mix_prior(small_y, 2, 0, struct('means', [-1 1; 6 -1], 'mean_weight', 1));
    good_means = vertcat(good.state.m);
    bad_means = vertcat(bad.state.m);
    good_error = mean(abs(sort(good_means(:, 1)) - [1; 4]));
    bad_error = mean(abs(sort(bad_means(:, 1)) - [1; 4]));
    assert(good_error < bad_error, 'A generating-map parent did not improve small-sample recovery.');

    malformed_failed = false;
    try
        spm_mix_prior(y, 2, 0, struct('means', [0 0], 'mean_weight', 1));
    catch
        malformed_failed = true;
    end
    assert(malformed_failed, 'A malformed non-K parent did not fail explicitly.');
end

function out = normalise_rows(x)
    x = x - mean(x, 2);
    out = x ./ max(sqrt(sum(x.^2, 2)), eps);
end

function out = projective(x)
    x = normalise_rows(x);
    d = size(x, 2);
    keep = triu(true(d));
    out = zeros(size(x, 1), nnz(keep));
    for i = 1:size(x, 1)
        outer = x(i, :)' * x(i, :);
        out(i, :) = outer(keep);
    end
end
