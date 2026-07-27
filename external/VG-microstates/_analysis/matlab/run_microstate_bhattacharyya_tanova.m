function [Results, results_csv] = run_microstate_bhattacharyya_tanova(results_mat, varargin)
%RUN_MICROSTATE_BHATTACHARYYA_TANOVA Channel-wise distributional TANOVA.
% Participant-condition means/widths come from GFP maps assigned to one
% group-blind global reference; no participant-condition clustering is used.
% Effects use univariate Gaussian Bhattacharyya distance at each channel.

    p = inputParser;
    addRequired(p, 'results_mat', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'semiology_manifest_csv', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'n_permutations', 5000, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'n_posterior', NaN, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'bayesian_fdr', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
    addParameter(p, 'random_seed', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'save_plots', true, @islogical);
    addParameter(p, 'reuse_results', false, @(x) islogical(x) && isscalar(x));
    parse(p, results_mat, varargin{:});
    cfg = p.Results;
    cfg.results_mat = char(cfg.results_mat);
    cfg.n_permutations = round(cfg.n_permutations);
    if isnan(cfg.n_posterior), cfg.n_posterior = cfg.n_permutations; end
    cfg.n_posterior = round(cfg.n_posterior);
    if isempty(cfg.output_dir)
        cfg.output_dir = fullfile(fileparts(cfg.results_mat), 'omnibus_analysis', 'bhattacharyya_tanova');
    else
        cfg.output_dir = char(cfg.output_dir);
    end
    if ~exist(cfg.output_dir, 'dir'), mkdir(cfg.output_dir); end
    plot_dir = fullfile(cfg.output_dir, 'difference_topoplots');
    if cfg.save_plots && ~exist(plot_dir, 'dir'), mkdir(plot_dir); end
    rng(cfg.random_seed, 'twister');

    results_csv = fullfile(cfg.output_dir, 'bhattacharyya_tanova_results.csv');
    summary_mat = fullfile(cfg.output_dir, 'bhattacharyya_tanova_summary.mat');
    if cfg.reuse_results
        if ~isfile(summary_mat), error('Saved Bhattacharyya summary not found: %s', summary_mat); end
        S = load(summary_mat, 'Results');
        Results = S.Results;
        [H, D] = load_plot_geometry(cfg.results_mat, Results.tests);
        plot_specs = plot_specs_from_results(Results.tests, D);
        if cfg.save_plots && exist('topoplot', 'file') == 2
            for i = 1:numel(plot_specs), plot_difference_maps(plot_specs{i}, H, D, plot_dir); end
        end
        fprintf('Bhattacharyya TANOVA plots reused saved results: %s\n', results_csv);
        return;
    end

    S = load(cfg.results_mat, 'HResults');
    if ~isfield(S, 'HResults'), error('The MAT file does not contain HResults.'); end
    H = S.HResults;
    D = extract_channel_distributions(H, cfg.semiology_manifest_csv, "participant_condition");
    D_event = extract_channel_distributions(H, cfg.semiology_manifest_csv, "event_type");
    D_semiology = extract_channel_distributions(H, cfg.semiology_manifest_csv, "semiology");
    [all_rows, plot_specs] = run_limited_planned_tests(D, D_event, D_semiology, cfg);

    Results = struct('config', cfg, 'data_manifest', D.manifest, 'tests', table(), ...
        'method', 'Per-channel Gaussian Bhattacharyya distance with permutation nulls and Bayesian-bootstrap FDR.');
    if ~isempty(all_rows), Results.tests = vertcat(all_rows{:}); end
    writetable(Results.tests, results_csv);
    writetable(D.manifest, fullfile(cfg.output_dir, 'bhattacharyya_solution_manifest.csv'));
    writetable(D_event.manifest, fullfile(cfg.output_dir, 'bhattacharyya_event_type_manifest.csv'));
    writetable(D_semiology.manifest, fullfile(cfg.output_dir, 'bhattacharyya_semiology_manifest.csv'));
    save(summary_mat, 'Results', '-v7.3');
    write_method_notes(fullfile(cfg.output_dir, 'bhattacharyya_tanova_methods.txt'), cfg);
    if cfg.save_plots && exist('topoplot', 'file') == 2
        for i = 1:numel(plot_specs)
            plot_difference_maps(plot_specs{i}, H, D, plot_dir);
        end
    end
    fprintf('Bhattacharyya TANOVA: %s\n', results_csv);
end

function D = extract_channel_distributions(H, semiology_manifest_csv, grouping)
    D = extract_fixed_reference_peak_distributions(H, semiology_manifest_csv, grouping);
    D.node_name = D.participant + "_" + D.condition;
    D.manifest = table(D.participant,D.group,D.condition,D.event_type,D.semiology,D.node_name,sum(D.n_maps,2), ...
        repmat(D.reference,numel(D.participant),1), ...
        'VariableNames',{'participant','group','condition','event_type','semiology','node_name','n_assigned_maps','reference'});
end

function [rows, plot_specs] = run_limited_planned_tests(D, E, S, cfg)
    plans = planned_comparison_sets(D, E, S);
    rows = cell(numel(plans),1); plot_specs = cell(numel(plans),1);
    for i = 1:numel(plans)
        P = plans{i};
        if P.kind == "between"
            C = collapse_by_participant(P.data);
            keep = C.group == P.a | C.group == P.b;
            if sum(keep) < 2 || numel(unique(C.group(keep))) < 2, continue; end
            [rows{i}, plot_specs{i}] = between_test(C.mu(keep,:,:), C.var(keep,:,:), C.group(keep), ...
                P.a, P.b, P.name, cfg, D, P.a, P.b);
        elseif P.kind == "mixed_event"
            [rows{i},plot_specs{i}]=mixed_event_test(P.data,P.a,P.b,P.name,cfg,D);
        else
            [A, B] = paired_conditions(P.data, P.a, P.b);
            if size(A.mu,1) < 2, continue; end
            [rows{i}, plot_specs{i}] = paired_test(A.mu,A.var,B.mu,B.var,P.a,P.b,P.name,cfg,D);
        end
    end
    rows = rows(~cellfun(@isempty,rows)); plot_specs = plot_specs(~cellfun(@isempty,plot_specs));
end

function plans = planned_comparison_sets(D, E, S)
    ii = find_condition_level(unique(D.condition), ["ii"]);
    preictal = find_condition_level(unique(D.condition), ["preict", "preictal"]);
    postictal = find_condition_level(unique(D.condition), ["postict", "postictal"]);
    pre = lower(D.condition) == lower(preictal); post = lower(D.condition) == lower(postictal);
    baseline = lower(D.condition) == lower(ii);
    pre_e = lower(E.condition) == lower(preictal); baseline_e = lower(E.condition) == lower(ii);
    pre_s = lower(S.condition) == lower(preictal);

    group_pair=@(a,b) complete_group_pair(D,a,b);
    sets = cell(10,1);
    sets{1}=group_pair("FS","ES");
    sets{2}=group_pair("FS","DD");
    sets{3}=group_pair("ES","DD");
    sets{4}=subset_distribution(E,pre_e & ismember(E.event_type,["NEAD","ES"]));
    sets{4}.condition=sets{4}.event_type;
    sets{5}=subset_distribution(E,ismember(E.group,["FS","DD"]) & (baseline_e | (pre_e & E.event_type=="NEAD")));
    sets{6}=subset_distribution(E,ismember(E.group,["ES","DD"]) & (baseline_e | (pre_e & E.event_type=="ES")));
    sets{7}=subset_distribution(E,E.group=="DD" & pre_e & ismember(E.event_type,["ES","NEAD"]));
    sets{7}.condition=sets{7}.event_type;
    sets{8}=subset_distribution(E,ismember(E.group,["ES","DD"]) & E.event_type=="ES" & (pre_e|lower(E.condition)==lower(postictal)));
    sets{9}=subset_distribution(E,ismember(E.group,["FS","DD"]) & E.event_type=="NEAD" & (pre_e|lower(E.condition)==lower(postictal)));
    sets{10}=subset_distribution(S,ismember(S.group,["FS","DD"]) & pre_s & S.event_type=="NEAD" & ismember(S.semiology,["hypo","hyper"]));
    sets{10}.condition(sets{10}.semiology=="hypo")="Hypomotor";
    sets{10}.condition(sets{10}.semiology=="hyper")="Hypermotor";

    names=["People: ES only vs FS only";"People: DD vs FS only";"People: DD vs ES only"; ...
        "Events: Preict functional seizure vs preict epileptic seizure"; ...
        "Functional events (FS or DD): II vs Preict";"Epileptic events (ES or DD): II vs Preict"; ...
        "Within DD: Preict epileptic vs preict functional event"; ...
        "Epileptic events (ES or DD): Preict vs Postict"; ...
        "Functional events (FS or DD): Preict vs Postict"; ...
        "Functional events (FS or DD): Preict Hypomotor vs Preict Hypermotor"];
    kinds=["between";"between";"between";"mixed_event";repmat("paired",6,1)];
    a=["FS";"FS";"ES";"NEAD";ii;ii;"NEAD";preictal;preictal;"Hypomotor"];
    b=["ES";"DD";"DD";"ES";preictal;preictal;"ES";postictal;postictal;"Hypermotor"];
    plans=cell(10,1);
    for i=1:10, plans{i}=struct('data',sets{i},'name',names(i),'kind',kinds(i),'a',a(i),'b',b(i)); end
end

function D2=complete_group_pair(D,a,b)
    D2=subset_distribution(D,ismember(D.group,[a,b]));
end

function D2 = subset_distribution(D, mask)
    mask=logical(mask(:)); D2=D;
    fields={'mu','var','participant','group','condition','event_type','semiology','n_maps','node_name'};
    for i=1:numel(fields)
        field=fields{i};
        if ismember(field,{'mu','var'}), D2.(field)=D.(field)(mask,:,:);
        elseif ismember(field,fieldnames(D)), D2.(field)=D.(field)(mask,:); end
    end
    if isfield(D,'manifest'), D2.manifest=D.manifest(mask,:); end
end

function ids = participants_with_all_conditions(participant, condition)
    levels=unique(condition(condition~="")); candidates=unique(participant); keep=false(size(candidates));
    for i=1:numel(candidates)
        present=unique(condition(participant==candidates(i)));
        keep(i)=all(ismember(levels,present));
    end
    ids=candidates(keep);
end

function P = collapse_by_participant(D)
    ids = unique(D.participant, 'stable');
    P.mu = nan(numel(ids),D.K,D.C); P.var = P.mu;
    P.participant = ids; P.group = strings(numel(ids),1);
    for i = 1:numel(ids)
        idx = D.participant == ids(i);
        [P.mu(i,:,:), P.var(i,:,:)] = pool_gaussians(D.mu(idx,:,:), D.var(idx,:,:), ones(sum(idx),1));
        P.group(i) = D.group(find(idx,1));
    end
end

function [A, B, ids, groups, semiology] = paired_conditions(D, condition_a, condition_b)
    ids_all = unique(D.participant, 'stable');
    ids = strings(0,1); groups = strings(0,1); semiology = strings(0,1);
    A.mu = nan(0,D.K,D.C); A.var = A.mu; B = A;
    for i = 1:numel(ids_all)
        ia = D.participant == ids_all(i) & D.condition == condition_a;
        ib = D.participant == ids_all(i) & D.condition == condition_b;
        if ~any(ia) || ~any(ib), continue; end
        ids(end+1,1) = ids_all(i); %#ok<AGROW>
        groups(end+1,1) = D.group(find(ia|ib,1)); %#ok<AGROW>
        semiology(end+1,1) = D.semiology(find(ia|ib,1)); %#ok<AGROW>
        [A.mu(end+1,:,:), A.var(end+1,:,:)] = pool_gaussians(D.mu(ia,:,:),D.var(ia,:,:),ones(sum(ia),1)); %#ok<AGROW>
        [B.mu(end+1,:,:), B.var(end+1,:,:)] = pool_gaussians(D.mu(ib,:,:),D.var(ib,:,:),ones(sum(ib),1)); %#ok<AGROW>
    end
end

function [T, spec] = between_test(mu, variance, labels, level_a, level_b, effect, cfg, D, display_a, display_b)
    if nargin < 10, display_a = level_a; display_b = level_b; end
    n = size(mu,1);
    [ma,va] = pool_gaussians(mu(labels==level_a,:,:),variance(labels==level_a,:,:),ones(sum(labels==level_a),1));
    [mb,vb] = pool_gaussians(mu(labels==level_b,:,:),variance(labels==level_b,:,:),ones(sum(labels==level_b),1));
    observed = bhattacharyya(ma,va,mb,vb);
    null = nan(cfg.n_permutations,D.K,D.C);
    for p = 1:cfg.n_permutations
        perm = labels(randperm(n));
        [m1,v1] = pool_gaussians(mu(perm==level_a,:,:),variance(perm==level_a,:,:),ones(sum(perm==level_a),1));
        [m2,v2] = pool_gaussians(mu(perm==level_b,:,:),variance(perm==level_b,:,:),ones(sum(perm==level_b),1));
        null(p,:,:) = bhattacharyya(m1,v1,m2,v2);
    end
    posterior_count = zeros(D.K,D.C);
    null95 = squeeze(prctile(null,95,1));
    for b = 1:cfg.n_posterior
        w = -log(max(rand(n,1),realmin));
        [m1,v1] = pool_gaussians(mu(labels==level_a,:,:),variance(labels==level_a,:,:),w(labels==level_a));
        [m2,v2] = pool_gaussians(mu(labels==level_b,:,:),variance(labels==level_b,:,:),w(labels==level_b));
        posterior_count = posterior_count + (squeeze(bhattacharyya(m1,v1,m2,v2)) > null95);
    end
    [T,sig] = result_table(effect,level_a,level_b,observed,null,null95,posterior_count,cfg,D,ma,va,mb,vb,n);
    spec = plot_spec(effect,display_a,display_b,squeeze(mb-ma),sig,D.state_labels);
end

function [T, spec] = paired_test(mu_a,var_a,mu_b,var_b,level_a,level_b,effect,cfg,D)
    n = size(mu_a,1);
    [ma,va] = pool_gaussians(mu_a,var_a,ones(n,1));
    [mb,vb] = pool_gaussians(mu_b,var_b,ones(n,1));
    observed = bhattacharyya(ma,va,mb,vb);
    null = nan(cfg.n_permutations,D.K,D.C);
    for p = 1:cfg.n_permutations
        swap = rand(n,1) < .5;
        ap=mu_a; av=var_a; bp=mu_b; bv=var_b;
        ap(swap,:,:)=mu_b(swap,:,:); av(swap,:,:)=var_b(swap,:,:);
        bp(swap,:,:)=mu_a(swap,:,:); bv(swap,:,:)=var_a(swap,:,:);
        [m1,v1]=pool_gaussians(ap,av,ones(n,1)); [m2,v2]=pool_gaussians(bp,bv,ones(n,1));
        null(p,:,:) = bhattacharyya(m1,v1,m2,v2);
    end
    posterior_count=zeros(D.K,D.C); null95=squeeze(prctile(null,95,1));
    for b=1:cfg.n_posterior
        w=-log(max(rand(n,1),realmin));
        [m1,v1]=pool_gaussians(mu_a,var_a,w); [m2,v2]=pool_gaussians(mu_b,var_b,w);
        posterior_count=posterior_count+(squeeze(bhattacharyya(m1,v1,m2,v2))>null95);
    end
    [T,sig]=result_table(effect,level_a,level_b,observed,null,null95,posterior_count,cfg,D,ma,va,mb,vb,n);
    spec=plot_spec(effect,level_a,level_b,squeeze(mb-ma),sig,D.state_labels);
end

function [T,spec]=mixed_event_test(E,level_a,level_b,effect,cfg,D)
    labels=E.condition; units=E.participant; n_units=numel(unique(units));
    weights=equal_participant_weights(units,ones(n_units,1));
    [ma,va]=pool_gaussians(E.mu(labels==level_a,:,:),E.var(labels==level_a,:,:),weights(labels==level_a));
    [mb,vb]=pool_gaussians(E.mu(labels==level_b,:,:),E.var(labels==level_b,:,:),weights(labels==level_b));
    observed=bhattacharyya(ma,va,mb,vb); null=nan(cfg.n_permutations,D.K,D.C);
    for p=1:cfg.n_permutations
        perm=hybrid_permute_labels(labels,units);
        [m1,v1]=pool_gaussians(E.mu(perm==level_a,:,:),E.var(perm==level_a,:,:),weights(perm==level_a));
        [m2,v2]=pool_gaussians(E.mu(perm==level_b,:,:),E.var(perm==level_b,:,:),weights(perm==level_b));
        null(p,:,:)=bhattacharyya(m1,v1,m2,v2);
    end
    posterior_count=zeros(D.K,D.C); null95=squeeze(prctile(null,95,1)); unit_levels=unique(units,'stable');
    for b=1:cfg.n_posterior
        unit_weights=-log(max(rand(numel(unit_levels),1),realmin));
        w=equal_participant_weights(units,unit_weights);
        [m1,v1]=pool_gaussians(E.mu(labels==level_a,:,:),E.var(labels==level_a,:,:),w(labels==level_a));
        [m2,v2]=pool_gaussians(E.mu(labels==level_b,:,:),E.var(labels==level_b,:,:),w(labels==level_b));
        posterior_count=posterior_count+(squeeze(bhattacharyya(m1,v1,m2,v2))>null95);
    end
    [T,sig]=result_table(effect,level_a,level_b,observed,null,null95,posterior_count,cfg,D,ma,va,mb,vb,n_units);
    spec=plot_spec(effect,level_a,level_b,squeeze(mb-ma),sig,D.state_labels);
end

function permuted=hybrid_permute_labels(labels,units)
    labels=string(labels(:)); units=string(units(:)); permuted=labels; unpaired=[];
    levels=unique(labels,'stable'); ul=unique(units,'stable');
    for i=1:numel(ul)
        idx=find(units==ul(i)); present=unique(labels(idx));
        if numel(present)==numel(levels) && numel(idx)==numel(levels)
            if rand<.5, permuted(idx)=flipud(labels(idx)); end
        else
            unpaired=[unpaired;idx(:)]; %#ok<AGROW>
        end
    end
    if numel(unpaired)>1, permuted(unpaired)=labels(unpaired(randperm(numel(unpaired)))); end
end

function weights=equal_participant_weights(units,unit_weights)
    levels=unique(units,'stable'); weights=zeros(numel(units),1);
    for i=1:numel(levels)
        idx=find(units==levels(i)); weights(idx)=unit_weights(i)/numel(idx);
    end
end

function [T,sig] = result_table(effect,a,b,observed,null,null95,count,cfg,D,ma,va,mb,vb,n)
    obs=squeeze(observed); p=(1+squeeze(sum(null>=reshape(obs,1,D.K,D.C),1)))./(cfg.n_permutations+1);
    post=(count+1)./(cfg.n_posterior+2); local_null=1-post; sig=false(D.K,D.C);
    for k=1:D.K, sig(k,:)=bayesian_fdr_mask(local_null(k,:),cfg.bayesian_fdr); end
    [state,channel]=ndgrid(1:D.K,1:D.C); N=D.K*D.C;
    difference=squeeze(mb-ma); width_a=sqrt(squeeze(va)); width_b=sqrt(squeeze(vb));
    state_label=reshape(string(D.state_labels(state(:))),[],1);
    channel_label=reshape(string(D.channel_labels(channel(:))),[],1);
    T=table(repmat(string(effect),N,1),repmat(string(a),N,1),repmat(string(b),N,1), ...
        strings(N,1),strings(N,1),state(:),state_label,channel(:),channel_label, ...
        obs(:),p(:),null95(:),post(:),local_null(:),repmat(cfg.bayesian_fdr,N,1),sig(:),difference(:), ...
        width_a(:),width_b(:),repmat(n,N,1), ...
        'VariableNames',{'effect','level_a','level_b','condition_a','condition_b','state_index','template_label', ...
        'channel_index','channel_label','bhattacharyya_distance','permutation_p','permutation_null95', ...
        'posterior_prob_above_null95','posterior_null_probability','bayesian_fdr_q','bayesian_fdr_significant','mean_difference_b_minus_a', ...
        'width_a','width_b','n_participants'});
end

function mask = bayesian_fdr_mask(local_null,q)
    mask=false(size(local_null)); valid=find(isfinite(local_null));
    if isempty(valid), return; end
    [v,order]=sort(local_null(valid)); keep=find(cumsum(v)./(1:numel(v))<=q,1,'last');
    if ~isempty(keep), mask(valid(order(1:keep)))=true; end
end

function pairs = planned_condition_pairs(conditions)
    ii = find_condition_level(conditions, ["ii"]);
    preictal = find_condition_level(conditions, ["preict", "preictal"]);
    postictal = find_condition_level(conditions, ["postict", "postictal"]);
    pairs = strings(0, 2);
    if ii ~= "" && preictal ~= "", pairs(end+1, :) = [ii preictal]; end
    if preictal ~= "" && postictal ~= "", pairs(end+1, :) = [preictal postictal]; end
end

function level = find_condition_level(levels, aliases)
    hit = find(ismember(lower(string(levels)), lower(string(aliases))), 1);
    if isempty(hit), level = ""; else, level = string(levels(hit)); end
end

function [m,v] = pool_gaussians(mu,variance,w)
    w=reshape(double(w),[],1,1); valid=isfinite(mu)&isfinite(variance); ww=w.*valid;
    denom=sum(ww,1); denom(denom<=0)=NaN;
    m=sum(ww.*replace_nan(mu,0),1)./denom;
    second=sum(ww.*replace_nan(variance+mu.^2,0),1)./denom;
    v=max(second-m.^2,1e-10);
end

function d = bhattacharyya(m1,v1,m2,v2)
    v1=max(v1,1e-10); v2=max(v2,1e-10);
    d=.25*log(.25.*(v1./v2+v2./v1+2))+.25.*(m1-m2).^2./(v1+v2);
end

function X = replace_nan(X,value), X(~isfinite(X))=value; end
function spec = plot_spec(effect,a,b,difference,sig,state_labels)
    spec=struct('effect',string(effect),'a',string(a),'b',string(b),'difference',difference,'significant',sig,'state_labels',{state_labels});
end

function plot_difference_maps(spec,H,D,plot_dir)
    if ~isfield(H,'common_chanlocs') || numel(H.common_chanlocs)~=D.C, return; end
    clim=max(abs(spec.difference(:)),[],'omitnan'); if ~isfinite(clim)||clim<=eps, clim=1; end
    utils=microstate_utilities(); [H.common_chanlocs,keep_idx]=utils.prepare_chanlocs_for_topoplot(H.common_chanlocs,D.C);
    if numel(keep_idx)~=D.C, return; end
    fig=figure('Visible','off','Color','white','Position',[50 50 max(1000,220*D.K) 330]);
    for k=1:D.K
        ax=subplot(1,D.K,k); locs=H.common_chanlocs; significant=find(spec.significant(k,:));
        if isempty(significant)
            topoplot(spec.difference(k,:),locs,'electrodes','off','maplimits',[-clim clim],'numcontour',6);
        else
            topoplot(spec.difference(k,:),locs,'electrodes','off','emarker2',{significant,'o','k',6,1},'maplimits',[-clim clim],'numcontour',6);
        end
        title(ax,sprintf('%s | n_sig=%d',spec.state_labels{k},numel(significant)),'Interpreter','none');
    end
    colormap(fig,'jet'); sgtitle(sprintf('%s: %s minus %s',spec.effect,spec.b,spec.a),'Interpreter','none','FontWeight','bold');
    exportgraphics(fig,fullfile(plot_dir,sprintf('%s__%s_minus_%s.png',safe_key(spec.effect),safe_key(spec.b),safe_key(spec.a))),'Resolution',300,'BackgroundColor','white');
    close(fig);
end

function [H,D] = load_plot_geometry(results_mat,T)
    global_mat=fullfile(fileparts(results_mat),'global','global_solution.mat');
    if ~isfile(global_mat), error('Global solution not found: %s',global_mat); end
    S=load(global_mat,'common_chanlocs'); H=struct('common_chanlocs',S.common_chanlocs);
    D=struct('K',max(double(T.state_index)),'C',max(double(T.channel_index)));
end

function specs = plot_specs_from_results(T,D)
    keys=string(T.effect)+"|"+string(T.level_a)+"|"+string(T.level_b)+"|"+string(T.condition_a)+"|"+string(T.condition_b);
    [groups,~]=findgroups(keys); specs=cell(max(groups),1);
    for i=1:max(groups)
        R=T(groups==i,:); difference=nan(D.K,D.C); significant=false(D.K,D.C);
        idx=sub2ind([D.K,D.C],double(R.state_index),double(R.channel_index));
        difference(idx)=double(R.mean_difference_b_minus_a); significant(idx)=logical(R.bayesian_fdr_significant);
        labels=arrayfun(@(k)char(string(R.template_label(find(double(R.state_index)==k,1)))),1:D.K,'UniformOutput',false);
        specs{i}=plot_spec(R.effect(1),R.level_a(1),R.level_b(1),difference,significant,labels);
    end
end

function out = safe_key(x), out=regexprep(char(x),'[^A-Za-z0-9_-]+','_'); end

function write_method_notes(path,cfg)
    fid=fopen(path,'w'); if fid<0, return; end; cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,['All maps are assigned to the same group-blind global centres; participant-condition clustering is not used.\n' ...
        'Channel widths are empirical SDs of assigned GFP-peak maps after polarity alignment to their global centre.\n' ...
        'Each solution/channel is treated as a univariate Gaussian. Factor contrasts use Bhattacharyya distance.\n' ...
        'Group labels are permuted between participants; condition labels are swapped within participants; interactions compare within-participant condition contrasts between groups.\n' ...
        'The practical null boundary is the channelwise 95th percentile of the permutation distribution.\n' ...
        'Bayesian-bootstrap posterior null probabilities are selected by expected Bayesian FDR <= %.3f.\n' ...
        'Permutations=%d; posterior draws=%d.\n'],cfg.bayesian_fdr,cfg.n_permutations,cfg.n_posterior);
end
