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
    addParameter(p, 'K_candidates', [4 5 6 7], @isnumeric);
    addParameter(p, 'run_backfit', true, @islogical);
    addParameter(p, 'redo', false, @islogical);
    addParameter(p, 'require_template_labels', true, @islogical);
    addParameter(p, 'run_topoplots', true, @islogical);
    addParameter(p, 'run_tanova', true, @islogical);
    addParameter(p, 'tanova_permutations', 5000, @isnumeric);
    parse(p, manifest_csv, output_dir, varargin{:});
    cfg = p.Results;

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
        'run_backfit', logical(cfg.run_backfit), ...
        'resume', ~logical(cfg.redo), ...
        'require_template_labels', logical(cfg.require_template_labels), ...
        'infer_localizer_dir', false);

    omnibus = struct();
    if cfg.run_topoplots || cfg.run_tanova
        omnibus = analyse_hierarchical_outputs(results_mat, ...
            'output_dir', fullfile(char(cfg.output_dir), 'omnibus_analysis'), ...
            'run_topoplots', logical(cfg.run_topoplots), ...
            'run_tanova', logical(cfg.run_tanova), ...
            'tanova_permutations', double(cfg.tanova_permutations));
    end

    if cfg.run_backfit
        python_exe = fullfile(root_dir, '.venv', 'Scripts', 'python.exe');
        analysis_script = fullfile(root_dir, 'analyse_ce_group_condition.py');
        command = sprintf('"%s" "%s" --results-dir "%s"', python_exe, analysis_script, char(cfg.output_dir));
        status = system(command);
        if status ~= 0
            error('C/E group-condition-semiology analysis failed with exit status %d.', status);
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
