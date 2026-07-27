function test_spm_mix_prior()
%TEST_SPM_MIX_PRIOR Smoke test for the seeded local SPM mixture copy.

    setup_microstate_paths();
    util = microstate_utilities();
    cfg = util.load_config();
    [ok, info] = util.ensure_spm_mix('', cfg.paths.spm_mixture_paths, false);
    assert(ok, 'SPM spm_mix not found. Checked: %s', strjoin(info.attempted, ', '));
    assert(exist('spm_mix_prior', 'file') == 2, 'spm_mix_prior is not on the MATLAB path.');

    rng(7, 'twister');
    y = [randn(50, 2) * 0.08 + repmat([1 0], 50, 1); ...
         randn(50, 2) * 0.08 + repmat([-1 0], 50, 1)];
    mix = spm_mix_prior(y, 2, 0, struct('means', [1 0; -1 0], 'mean_weight', 2));

    assert(isstruct(mix) && isfinite(double(real(mix.fm))), 'Seeded mixture did not produce finite free energy.');
    assert(all(abs(mix.prior.beta_0 - 2) < eps), 'Mean-prior weight was not retained.');
    means = vertcat(mix.state.m);
    x_means = sort(means(:, 1));
    assert(x_means(1) < -0.3 && x_means(2) > 0.3, 'Seeded mixture collapsed separated clusters.');

    failed = false;
    try
        spm_mix_prior(y, 2, 0, struct('means', [1 0]));
    catch
        failed = true;
    end
    assert(failed, 'Bad seed shape should fail.');
end
