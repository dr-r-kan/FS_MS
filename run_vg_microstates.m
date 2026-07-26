function outputs = run_vg_microstates(manifest_csv, output_dir, varargin)
%RUN_VG_MICROSTATES Fit VG hierarchy and TANOVA for the FS microstates manifest.

    root_dir = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(manifest_csv)
        data_dir = 'G:\FS_microstates_study\vg_microstates';
        manifest_csv = existing_manifest(data_dir);
        if isempty(manifest_csv)
            python_exe = fullfile(root_dir, '.venv', 'Scripts', 'python.exe');
            pipeline = fullfile(root_dir, 'vg_microstates_pipeline.py');
            command = sprintf('cd /d "%s" && "%s" "%s"', root_dir, python_exe, pipeline);
            fprintf('VG manifest missing; building it with the hardwired Python pipeline...\n');
            status = system(command);
            if status ~= 0
                error('vg_microstates_pipeline.py failed with exit status %d.', status);
            end
            manifest_csv = existing_manifest(data_dir);
        end
    end
    if nargin < 2 || isempty(output_dir)
        output_dir = 'G:\FS_microstates_study\vg_microstates\hierarchical';
    end

    if isempty(manifest_csv)
        error(['No VG manifest exists in G:\FS_microstates_study\vg_microstates. ' ...
            'Run vg_microstates_pipeline.py first.']);
    end

    p = inputParser;
    addRequired(p, 'manifest_csv', @(x) ischar(x) || isstring(x));
    addRequired(p, 'output_dir', @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_candidates', 6, @isnumeric);
    addParameter(p, 'global_manual_labels', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'interactive_global_labelling', true, @islogical);
    addParameter(p, 'relabel_global_states', false, @islogical);
    addParameter(p, 'spm_prior_weight', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'global_only', false, @islogical);
    addParameter(p, 'ecg_qc_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'run_backfit', true, @islogical);
    addParameter(p, 'run_participant_condition_fits', true, @islogical);
    addParameter(p, 'run_global_backfit', true, @islogical);
    addParameter(p, 'run_cardiac_phase', true, @islogical);
    addParameter(p, 'run_legacy_analyses', false, @islogical);
    addParameter(p, 'preictal_window_refit_only', false, @islogical);
    addParameter(p, 'window_refit_length_s', 60, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window_refit_step_s', 30, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window_refit_time_window_s', 1200, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window_refit_min_peaks', 30, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    addParameter(p, 'window_refit_min_state_peaks', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'window_refit_start_record', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'window_refit_max_records', Inf, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'window_refit_csv', 'preictal_window_refit_maps.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'compact_results_mat', false, @islogical);
    addParameter(p, 'redo', false, @islogical);
    addParameter(p, 'require_template_labels', true, @islogical);
    addParameter(p, 'run_topoplots', true, @islogical);
    addParameter(p, 'run_tanova', true, @islogical);
    addParameter(p, 'tanova_permutations', 5000, @isnumeric);
    addParameter(p, 'backfit_time_bin_s', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'backfit_time_window_s', 1200, @(x) isnumeric(x) && isscalar(x) && x > 0);
    parse(p, manifest_csv, output_dir, varargin{:});
    cfg = p.Results;
    if ~isequal(double(cfg.K_candidates(:)'), 6)
        error('The paper analysis requires the locked K=6 hierarchy.');
    end
    if cfg.redo
        python_exe = fullfile(root_dir, '.venv', 'Scripts', 'python.exe');
        pipeline = fullfile(root_dir, 'vg_microstates_pipeline.py');
        command = sprintf('cd /d "%s" && "%s" "%s" --overwrite', root_dir, python_exe, pipeline);
        fprintf('Redo requested; rebuilding the VG manifest, including ictal segments...\n');
        status = system(command);
        if status ~= 0
            error('vg_microstates_pipeline.py failed with exit status %d.', status);
        end
        cfg.manifest_csv = existing_manifest('G:\FS_microstates_study\vg_microstates');
        if isempty(cfg.manifest_csv)
            error('Manifest rebuild completed without producing a VG manifest.');
        end
    end

    vg_root = fullfile(root_dir, 'external', 'VG-microstates');
    addpath(vg_root);
    setup_microstate_paths(vg_root);

    if ~exist(char(cfg.output_dir), 'dir')
        mkdir(char(cfg.output_dir));
    end

    [HResults, results_mat] = fit_microstate_hierarchical_csv(char(cfg.manifest_csv), ...
        'output_dir', char(cfg.output_dir), ...
        'K_candidates', double(cfg.K_candidates), ...
        'global_manual_labels', cfg.global_manual_labels, ...
        'interactive_global_labelling', logical(cfg.interactive_global_labelling), ...
        'relabel_global_states', logical(cfg.relabel_global_states), ...
        'spm_prior_weight', double(cfg.spm_prior_weight), ...
        'global_only', logical(cfg.global_only), ...
        'ecg_qc_dir', char(cfg.ecg_qc_dir), ...
        'preictal_window_refit_only', logical(cfg.preictal_window_refit_only), ...
        'window_refit_length_s', double(cfg.window_refit_length_s), ...
        'window_refit_step_s', double(cfg.window_refit_step_s), ...
        'window_refit_time_window_s', double(cfg.window_refit_time_window_s), ...
        'window_refit_min_peaks', double(cfg.window_refit_min_peaks), ...
        'window_refit_min_state_peaks', double(cfg.window_refit_min_state_peaks), ...
        'window_refit_start_record', double(cfg.window_refit_start_record), ...
        'window_refit_max_records', double(cfg.window_refit_max_records), ...
        'window_refit_csv', char(cfg.window_refit_csv), ...
        'compact_results_mat', logical(cfg.compact_results_mat), ...
        'run_backfit', logical(cfg.run_backfit), ...
        'run_participant_condition_fits', logical(cfg.run_participant_condition_fits), ...
        'run_global_backfit', logical(cfg.run_global_backfit), ...
        'run_cardiac_phase', logical(cfg.run_cardiac_phase), ...
        'backfit_time_bin_s', double(cfg.backfit_time_bin_s), ...
        'backfit_time_window_s', double(cfg.backfit_time_window_s), ...
        'resume', ~logical(cfg.redo), ...
        'require_template_labels', logical(cfg.require_template_labels), ...
        'infer_localizer_dir', false);

    omnibus = struct();
    if cfg.run_topoplots || cfg.run_tanova
        omnibus = analyse_hierarchical_outputs(results_mat, ...
            'output_dir', fullfile(char(cfg.output_dir), 'omnibus_analysis'), ...
            'run_topoplots', logical(cfg.run_topoplots), ...
            'run_tanova', logical(cfg.run_tanova), ...
            'semiology_manifest_csv', char(cfg.manifest_csv), ...
            'tanova_permutations', double(cfg.tanova_permutations));
    end

    if cfg.run_backfit && cfg.run_legacy_analyses && ~cfg.preictal_window_refit_only
        python_exe = fullfile(root_dir, '.venv', 'Scripts', 'python.exe');
        analysis_script = fullfile(root_dir, 'analyse_ce_group_condition.py');
        command = sprintf('"%s" "%s" --results-dir "%s"', python_exe, analysis_script, char(cfg.output_dir));
        status = system(command);
        if status ~= 0
            error('C/E group-condition-semiology analysis failed with exit status %d.', status);
        end
        analysis_script = fullfile(root_dir, 'analyse_transition_matrices.py');
        command = sprintf('"%s" "%s" --results-dir "%s"', python_exe, analysis_script, char(cfg.output_dir));
        status = system(command);
        if status ~= 0
            error('Participant C/E transition analysis failed with exit status %d.', status);
        end
        analysis_script = fullfile(root_dir, 'analyse_participant_ce_cardiac_phase.py');
        command = sprintf('"%s" "%s" --results-dir "%s"', python_exe, analysis_script, char(cfg.output_dir));
        status = system(command);
        if status ~= 0
            error('Participant-wise C/E cardiac-phase prediction analysis failed with exit status %d.', status);
        end
        if cfg.backfit_time_bin_s <= 20
            analysis_script = fullfile(root_dir, 'analyse_global_c_last_minute.py');
            command = sprintf('"%s" "%s" --results-dir "%s"', python_exe, analysis_script, char(cfg.output_dir));
            status = system(command);
            if status ~= 0
                error('Global-C final-minute analysis failed with exit status %d.', status);
            end
        end
    end

    outputs = struct('HResults', HResults, 'results_mat', results_mat, 'omnibus', omnibus);
end

function manifest_csv = existing_manifest(data_dir)
    candidates = fullfile(data_dir, { ...
        'vg_microstate_manifest_mat.csv', ...
        'vg_microstate_manifest_set.csv', ...
        'vg_microstate_manifest_fif.csv'});
    hit = find(cellfun(@isfile, candidates), 1);
    if isempty(hit)
        manifest_csv = '';
    else
        manifest_csv = candidates{hit};
    end
end
