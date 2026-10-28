function c = template_prespecified_v2_constants()
%TEMPLATE_PRESPECIFIED_V2_CONSTANTS The frozen parameters of the pre-specified labelling rule.
%   These values are part of the registered analysis plan. They are deliberately constants in
%   code (not function arguments), so that a different value cannot be passed by accident; any
%   change alters the script fingerprint and invalidates stage-C outputs.
%
%   rule_id               identifier written to every output that depends on the labels
%   classes               the six MetaMaps classes the K=6 solution is mapped to (class D excluded a priori)
%   template_K            MetaMaps solution used as reference (K=7 = classes A-G)
%   strong_threshold      a state whose assigned |spatial correlation| is below this value is FLAGGED
%                         (labelling_quality = 'below_threshold'); it never stops the stage
%   allowed_modes         channel-matching modes accepted; 'first_common_channels' is never accepted
%   min_estimated_channels  fewer fitted-map channels than this make the template comparison meaningless
%   tie_tolerance         total-|r| tolerance within which assignments count as tied
%
%   v2 (supersedes v1, which stopped the stage when any assigned |r| < 0.5): state names are nominal and every
%   registered outcome is invariant to them, so low template similarity is reported, not enforced. Only
%   TECHNICAL failures that make the comparison meaningless stop the stage.
    c = struct();
    c.rule_id = 'template_prespecified_v2';
    c.classes = {'A', 'B', 'C', 'E', 'F', 'G'};
    c.excluded_classes = {'D'};
    c.template_K = 7;
    c.strong_threshold = 0.5;
    c.allowed_modes = {'labels', 'coordinates_idw_to_template'};
    c.min_estimated_channels = 8;
    c.tie_tolerance = 1e-9;
    c.K = 6;
    c.nominal_label_semantics = ['State names are nominal; primary and secondary outcomes do not depend on them; ' ...
        'below_threshold means descriptive per-state names are unreliable.'];
end
