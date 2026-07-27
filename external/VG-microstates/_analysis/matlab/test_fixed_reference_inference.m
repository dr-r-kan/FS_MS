function test_fixed_reference_inference
% Small end-to-end regression check for participant-level fixed backfitting.
    rng(2); K=2; C=4;
    centers=[1 -1 0 0;0 0 1 -1]; centers=centers./vecnorm(centers,2,2);
    participant=strings(36,1); group=participant; condition=participant; semiology=participant;
    seizure_type=participant; file_path=participant;
    peak_records=cell(36,1); conditions=["II","preict","postict"]; q=0;
    for p=1:12
        for c=1:numel(conditions)
            q=q+1; participant(q)=sprintf('P%02d',p);
            if p<=6, group(q)="FS"; seizure_type(q)="FS"; else, group(q)="ES"; seizure_type(q)="ES"; end
            if mod(p,2), semiology(q)="hypo"; else, semiology(q)="hyper"; end
            condition(q)=conditions(c);
            file_path(q)=sprintf('P%02d_%s.mat',p,conditions(c));
            X=[centers(1,:)+.08*randn(30,C);centers(2,:)+.08*randn(30,C)];
            X=X-mean(X,2); X=X./vecnorm(X,2,2);
            peak_records{q}=struct('maps_norm',single(X));
        end
    end
    manifest=table(participant,condition,group,file_path);
    HResults=struct('manifest',manifest,'peak_records',{peak_records}, ...
        'global',struct('centers',centers,'canonical_state_labels',{{'A','B'}}), ...
        'selected_K',K,'common_labels',{{'F3','F4','P3','P4'}});
    mat_file=[tempname '.mat']; out_dir=tempname; save(mat_file,'HResults');
    manifest_csv=[tempname '.csv']; writetable(table(participant,condition,file_path,seizure_type,semiology, ...
        'VariableNames',{'participant','condition','file_path','seizure_type','hypo_hyper'}),manifest_csv);
    cleanup=onCleanup(@()clean(mat_file,manifest_csv,out_dir)); %#ok<NASGU>
    [B,~]=run_microstate_bhattacharyya_tanova(mat_file,'output_dir',out_dir, ...
        'semiology_manifest_csv',manifest_csv,'n_permutations',20,'n_posterior',20,'save_plots',false);
    [T,~]=run_microstate_hierarchical_tanova(mat_file,'output_dir',fullfile(out_dir,'standard'), ...
        'semiology_manifest_csv',manifest_csv,'n_permutations',20,'n_posterior',20,'save_plots',false,'verbose',false);
    assert(height(B.data_manifest)==36 && ~isempty(B.tests));
    assert(numel(unique(string(B.tests.effect)))<=10);
    assert(any(string(B.tests.effect)=='People: ES only vs FS only'));
    assert(numel(unique(string(T.results.effect)))==10);
    assert(any(string(T.results.effect)=='Functional events (FS or DD): Preict Hypomotor vs Preict Hypermotor'));
    fprintf('Fixed-reference inference check passed.\n');
end

function clean(mat_file,manifest_csv,out_dir)
    if isfile(mat_file), delete(mat_file); end
    if isfile(manifest_csv), delete(manifest_csv); end
    if isfolder(out_dir), rmdir(out_dir,'s'); end
end
