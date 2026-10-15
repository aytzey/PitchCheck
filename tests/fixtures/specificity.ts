import type { SpecificityContrast } from "@/shared/types";

export const specificityContrast: SpecificityContrast = {
  available: true, medial_z: 1.84, z_min: 1, relative_change: 0.031, prior_points: 1.26,
  roi_multiplier: 0.625, grounded: true,
  rois: { dmpfc_chua2011: { z: 2.1, pct: 97.5, role: "primary" } },
  top_parcels: [{ name: "17Networks_LH_DefaultB_PFCd_1", z: 2.6 }],
  replaced: [{ span: "40 reps", category: "your reps", persona_fact: "moved 40 reps", grounded: true }],
  control_text: "You moved your reps onto a new CRM recently.", n_words: [10, 10],
  encoder: "meta-llama/Llama-3.2-3B", text_feature_compatible: true, direction: "higher",
  basis: "in_silico_specificity_contrast_v1", provenance: "literature_prior:chua2009+chua2011_rois;unvalidated",
  caveat: "Signed average-reader contrast on synthetic text timing; details may not be true or relevant.",
};
