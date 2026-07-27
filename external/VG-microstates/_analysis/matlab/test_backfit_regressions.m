function test_backfit_regressions
% Regression check for hierarchical VB mixtures and full-record spatial filtering.
    rng(7);
    C = 6; K = 3; n = 300;
    centers = randn(K, C);
    centers = centers - mean(centers, 2);
    centers = centers ./ vecnorm(centers, 2, 2);
    labels = repelem((1:K)', 20);
    maps = centers(labels, :) + 0.05 * randn(numel(labels), C);
    maps = maps - mean(maps, 2);
    maps = maps ./ vecnorm(maps, 2, 2);
    Results = struct('centers', centers, 'method', 'spm_vb_hierarchical', ...
        'maps_nc', maps, 'labels', labels);

    angle = (0:C-1)' * 2*pi/C;
    Sim = struct('X_noisy', randn(C, n), 'sfreq', 100, ...
        'pos', [cos(angle), sin(angle), 0.5 * ones(C, 1)], ...
        'preprocessing', struct('apply_average_reference', false, 'filter_band', [], ...
        'spatial_filter', 'smoothing', 'spatial_filter_neighbours', 2, 'spatial_filter_strength', 0.75));
    smooth = backfit_microstate_timecourse(Sim, Results);
    assert(smooth.ok && smooth.mixture.available);

    util = microstate_utilities();
    [X_smooth, ~] = util.apply_spatial_filter(Sim.X_noisy, Sim);
    explicit = Sim;
    explicit.X_noisy = X_smooth;
    explicit.preprocessing.spatial_filter = 'none';
    expected = backfit_microstate_timecourse(explicit, Results);
    assert(isequal(smooth.hard.assignments, expected.hard.assignments));
    assert(max(abs(smooth.hard.confidence - expected.hard.confidence)) < 1e-12);
    fprintf('Backfit regression check passed.\n');
end
