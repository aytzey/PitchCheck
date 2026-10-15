"""Neural-only persuasion calibration for PitchCheck.

This module intentionally does not score words, patterns, CTAs, proof points, or
other text heuristics.  The persuasion prior is derived from TRIBE-predicted
fMRI response geometry only.  Text and persona are passed to the LLM as
untrusted context for semantic explanation and rewrite generation, not as a
deterministic scoring audit.
"""
from __future__ import annotations

import math
import os
from typing import Any

from tribe_service import roi_features


# ponytail: hand-set unvalidated prior; retune when paired outcome or human-rating data exists.
NEURO_AXIS_WEIGHTS = {
    "self_value": 0.28,
    "reward_affect": 0.24,
    "social_sharing": 0.18,
    "encoding_attention": 0.15,
    "processing_fluency": 0.15,
}

CALIBRATION_METHODOLOGY_VERSION = "neural_only_v2.3"

# Provenance registry for every cited finding and literature prior. Optional evidence_level
# (";"-joined) and modality appear only on entries read or abstract-checked; absent = unverified.
RESEARCH_SOURCES = [
    {
        "key": "falk_2010_persuasion_change",
        "citation": "Falk et al. 2010",
        "title": "Predicting persuasion-induced behavior change from the brain",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3027351/",
        "evidence_level": "individual_differences_in_sample",
        "modality": "fMRI; text+image slides with audio",
        "finding": "Between-person, N=20: an MPFC ROI (Soon et al. 2008 coordinates) during text+image sunscreen slides read aloud correlated with later change in self-reported sunscreen use (r=.49; .46 controlling attitude/intention change; split-half CV mean r=.48); precuneus ROI ns.",
    },
    {
        "key": "falk_2012_neural_focus_group",
        "citation": "Falk et al. 2012",
        "title": "From neural responses to population behavior",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3725133/",
        "finding": "Small-sample neural responses predicted population-level media effects.",
    },
    {
        "key": "venkatraman_2015_ad_success",
        "citation": "Venkatraman et al. 2015",
        "title": "Predicting advertising success beyond traditional measures",
        "url": "https://doi.org/10.1509/jmr.13.0593",
        "finding": "Neurophysiological measures improved market-response prediction beyond traditional measures.",
    },
    {
        "key": "baek_scholz_2017_sharing",
        "citation": "Baek, Scholz et al. 2017",
        "title": "The value of sharing information",
        "url": "https://doi.org/10.1177/0956797617695073",
        "finding": "Value, self-related, and social-cognition ROIs tracked selection and sharing preferences.",
    },
    {
        "key": "chan_2024_ad_liking",
        "citation": "Chan et al. 2024",
        "title": "Neural signals of video advertisement liking",
        "url": "https://doi.org/10.1177/00222437231194319",
        "finding": "Affective and social-cognition responses predicted ad liking with informative temporal dynamics.",
    },
    {
        "key": "chan_scholz_2023_cross_cultural_sharing",
        "citation": "Chan, Scholz et al. 2023",
        "title": "Neural signals predict information sharing across cultures",
        "url": "https://doi.org/10.1073/pnas.2313175120",
        "finding": "Self-, social- and value-related signals from scanned groups tracked population-level news sharing in US and Dutch samples, explaining variance beyond self-report ratings.",
    },
    {
        "key": "cohen_2024_reward_mentalizing",
        "citation": "Cohen et al. 2024",
        "title": "Reward and mentalizing circuits separately predict persuasiveness",
        "url": "https://doi.org/10.1038/s41598-024-62341-3",
        "finding": "Reward and mentalizing synchrony contributed dissociable persuasion evidence by narrative type.",
    },
    {
        "key": "scholz_chan_falk_2025_mega_analysis",
        "citation": "Scholz, Chan & Falk et al. 2025",
        "title": "Brain activity explains message effectiveness: A mega-analysis of 16 neuroimaging studies",
        "url": "https://doi.org/10.1093/pnasnexus/pgaf287",
        "finding": "Pooled across 16 fMRI datasets (572 people, 739 messages), reward- and social-processing activity was associated with message effectiveness in individuals and at scale, complementing self-report.",
    },
    {
        "key": "cao_reimann_2020_triangulation",
        "citation": "Cao & Reimann 2020",
        "title": "Data triangulation in consumer neuroscience",
        "url": "https://doi.org/10.3389/fpsyg.2020.550204",
        "finding": "Consumer-neuroscience interpretation should explicitly mitigate reverse-inference risk.",
    },
    {
        "key": "tribe_v2_foundation_model",
        "citation": "d'Ascoli et al. 2026",
        "title": "A foundation model of vision, audition, and language for in-silico neuroscience",
        "url": "https://github.com/facebookresearch/tribev2",
        "finding": "TRIBE v2 predicts average-subject fMRI response analogues on the fsaverage5 cortical mesh.",
    },
    {
        "key": "petty_cacioppo_1986_elm",
        "citation": "Petty & Cacioppo 1986",
        "title": "The Elaboration Likelihood Model of persuasion",
        "url": "https://doi.org/10.1016/S0065-2601(08)60214-2",
        "finding": "High-elaboration audiences respond to argument quality; low-elaboration audiences respond to peripheral cues.",
    },
    {
        "key": "tversky_kahneman_1981_framing",
        "citation": "Tversky & Kahneman 1981",
        "title": "The framing of decisions and the psychology of choice",
        "url": "https://doi.org/10.1126/science.7455683",
        "finding": "Describing logically equivalent outcomes as gains or losses reversed risky choices: risk aversion for gains, risk seeking for losses.",
    },
    {
        "key": "goldstein_cialdini_2008_social_norms",
        "citation": "Goldstein, Cialdini & Griskevicius 2008",
        "title": "A room with a viewpoint: using social norms to motivate conservation",
        "url": "https://doi.org/10.1086/586910",
        "finding": "In two hotel field experiments, descriptive-norm messages beat a standard environmental appeal, and norms about guests who had stayed in the same room (provincial norms) did best.",
    },
    {
        "key": "carpenter_2013_but_you_are_free",
        "citation": "Carpenter 2013",
        "title": 'A Meta-Analysis of the Effectiveness of the "But You Are Free" Compliance-Gaining Technique',
        "url": "https://doi.org/10.1080/10510974.2012.727941",
        "finding": "Adding an explicit freedom-to-refuse phrase raised compliance on average, mostly in face-to-face requests.",
    },
    {
        "key": "matz_2017_psychological_targeting",
        "citation": "Matz et al. 2017",
        "title": "Psychological targeting as an effective approach to digital mass persuasion",
        "url": "https://doi.org/10.1073/pnas.1710966114",
        "finding": "Three Facebook ad field experiments (>3.5M people): ads matched to extraversion or openness inferred from Likes produced up to 40% more clicks and 50% more purchases than mismatched or unpersonalized ads; the design drew published comments.",
    },
    {
        "key": "alter_oppenheimer_2009_fluency",
        "citation": "Alter & Oppenheimer 2009",
        "title": "Uniting the tribes of fluency to form a metacognitive nation",
        "url": "https://doi.org/10.1177/1088868309341564",
        "finding": "Easier-to-process messages are judged more true, more likable, and less risky.",
    },
    {
        "key": "ntoumanis_2024_naturalistic_isc",
        "citation": "Ntoumanis et al. 2024",
        "title": "Deciphering the neural responses to a naturalistic persuasive message",
        "url": "https://doi.org/10.1073/pnas.2401317121",
        "evidence_level": "individual_differences_in_sample",
        "modality": "fMRI; one audio healthy-eating call",
        "finding": "N=50: listeners who resisted were more synchronized (vmPFC, precuneus, SMG, pgACC); no region was more synchronized in persuaded listeners; ISC correlated negatively with ΔWTP (r≈−.39).",
    },
    {
        "key": "li_2026_dacc_dynamic_attitude",
        "citation": "Li et al. 2026",
        "title": "The role of dorsal anterior cingulate cortex in dynamic attitude changes in naturalistic settings",
        "url": "https://doi.org/10.1038/s42003-026-09794-6",
        "evidence_level": "individual_differences_in_sample;within_subject_item_level",
        "modality": "fMRI; persuasive videos (N=35), debate video (N=37)",
        "finding": "Inter-subject dACC dynamics and dACC–DMN coupling tracked moment-to-moment attitude change; vmPFC/dmPFC similarity did not in the continuous debate.",
    },
    {
        "key": "knutson_2007_neural_purchase",
        "citation": "Knutson et al. 2007",
        "title": "Neural predictors of purchases",
        "url": "https://doi.org/10.1016/j.neuron.2006.11.010",
        "evidence_level": "within_subject_item_level",
        "modality": "fMRI; product photos then prices (N=26)",
        "finding": "Preference activated NAcc and MPFC activation tracked reduced prices (price differential); right-insula activation during the price period predicted not buying, though its correlation with price differential was not significant (conjoined Z=−1.36). NAcc, MPFC and insula each predicted purchases beyond self-report.",
    },
    {
        "key": "falk_2010_jocn_mentalizing_network",
        "citation": "Falk, Rameson et al. 2010 (JoCN)",
        "title": "The neural correlates of persuasion: a common network across cultures and media",
        "url": "https://doi.org/10.1162/jocn.2009.21363",
        "evidence_level": "within_subject_condition_contrast",
        "modality": "fMRI; text passages (US N=15, Korean N=14), video (Study 3)",
        "finding": "Passages each reader rated persuasive drew more dmPFC, bilateral pSTS and temporal-pole activity in both samples (p<.001 uncorrected); insula was higher for unpersuasive passages.",
    },
    {
        "key": "plassmann_2007_wtp_ofc",
        "citation": "Plassmann et al. 2007",
        "title": "Orbitofrontal cortex encodes willingness to pay in everyday economic transactions",
        "url": "https://doi.org/10.1523/JNEUROSCI.2131-07.2007",
        "evidence_level": "within_subject_item_level",
        "modality": "fMRI; food photos, real bids (N=19)",
        "finding": "mOFC and dlPFC activity encoded willingness to pay during real bidding.",
    },
    {
        "key": "chua_2011_tailored_self_related",
        "citation": "Chua et al. 2011",
        "title": "Self-related neural response to tailored smoking-cessation messages predicts quitting",
        "url": "https://doi.org/10.1038/nn.2761",
        "evidence_level": "individual_differences_in_sample",
        "modality": "fMRI; audio-visual statements tailored from motivated smokers' surveys (N=91 scanned, 87 follow-up)",
        "finding": "dmPFC response to tailored vs neutral statements predicted quitting at 4 months (in-sample individual-level OR=1.31); the combined three-region OR=1.36 included angular cortex, precuneus alone p=.078. Memory did not predict quitting; no self-report comparison. Small single-laboratory health sample; disclosed commercial tailoring interests; not a validated message-level effect.",
    },
    {
        "key": "klucharev_2008_expert_power",
        "citation": "Klucharev et al. 2008",
        "title": "Brain mechanisms of persuasion: how 'expert power' modulates memory and attitudes",
        "url": "https://doi.org/10.1093/scan/nsn022",
        "evidence_level": "within_subject_item_level",
        "modality": "fMRI; expert/celebrity + object photos (24 women scanned; behaviour analysed with N=23, t(22); fMRI n=16/18)",
        "finding": "Expert pairing improved memory and attitudes; memory and attitude were largely independent (r=.05); objects later disliked drew stronger responses.",
    },
    {
        "key": "vezich_2017_framing_action",
        "citation": "Vezich et al. 2017",
        "title": "Modulating the neural bases of persuasion: why/how, gain/loss, and users/non-users",
        "url": "https://doi.org/10.1093/scan/nsw113",
        "evidence_level": "within_subject_condition_contrast;individual_differences_in_sample",
        "modality": "fMRI; 40 text ads read with audio (N=36)",
        "finding": "MPFC response to gain-framed \"why\" ads (vs facts) tracked next-week sunscreen use beyond intentions (r=.34, one-tailed), more strongly in non-users (difference marginal); loss frames null; a separate pretest rated loss frames marginally more persuasive.",
    },
    {
        "key": "diplinio_2025_affective_cognitive_matching",
        "citation": "Di Plinio et al. 2025",
        "title": "Brain and behavior in persuasion: the role of affective-cognitive matching",
        "url": "https://doi.org/10.1111/psyp.70088",
        "evidence_level": "individual_differences_in_sample",
        "modality": "fMRI; read product texts (N=35)",
        "finding": "A larger affective-minus-cognitive response difference went with lower evaluation (std β=−.167): activity was higher under mismatch; orientation alone did not predict evaluations.",
    },
    {
        "key": "haddock_2008_affect_cognition_matching",
        "citation": "Haddock et al. 2008",
        "title": "Should persuasion be affective or cognitive? The moderating effects of need for affect and need for cognition",
        "url": "https://doi.org/10.1177/0146167208314871",
        "evidence_level": "behavioral",
        "modality": "three experiments; affect- vs cognition-based messages",
        "finding": "Affective messages persuaded people high in need for affect, cognitive messages people high in need for cognition.",
    },
    {
        "key": "hernandez_vergara_2025_dogmatic_tone",
        "citation": "Hernández-Vergara et al. 2025",
        "title": "Neural bases of psychological reactance: an fMRI study on dogmatic and suggestive health communications",
        "url": "https://doi.org/10.1080/00913367.2025.2564236",
        "evidence_level": "within_subject_condition_contrast",
        "modality": "fMRI; 10–30-word health texts (N=37; submitted version)",
        "finding": "Dogmatic messages raised dMPFC, IFG/anterior insula, fusiform and parietal activity yet were rated less persuasive (2.6 vs 3.3); suggestive messages engaged vmPFC/OFC.",
    },
    {
        "key": "yomogida_2017_norm_persuasion",
        "citation": "Yomogida et al. 2017",
        "title": "The neural basis of changing social norms through persuasion",
        "url": "https://doi.org/10.1038/s41598-017-16572-2",
        "evidence_level": "within_subject_condition_contrast",
        "modality": "fMRI; 5-sentence persuasive texts (N=27)",
        "finding": "All persuasive texts recruited left dlPFC and dMPFC versus neutral text; norm-targeted persuasion also engaged temporal poles, TPJ and dMPFC.",
    },
    {
        "key": "falk_2011_smoking_replication",
        "citation": "Falk et al. 2011",
        "title": "Neural activity during health messaging predicts reductions in smoking above and beyond self-report",
        "url": "https://doi.org/10.1037/a0022259",
        "evidence_level": "individual_differences_in_sample",
        "modality": "fMRI; antismoking TV ads (N=28)",
        "finding": "MPFC response predicted reduced expired CO beyond intentions and self-efficacy; neural and self-report measures explained independent variance.",
    },
    {
        "key": "cooper_2015_self_value_mpfc",
        "citation": "Cooper et al. 2015",
        "title": "Brain activity in self- and value-related regions in response to online antismoking messages predicts behavior change",
        "url": "https://doi.org/10.1027/1864-1105/a000146",
        "evidence_level": "individual_differences_in_sample",
        "modality": "fMRI; silent animated banner ads (50 scanned, 46 retained, N=42 analysed)",
        "finding": "Self- and value-related MPFC subregions predicted smoking reduction independent of intention, self-efficacy and risk perception.",
    },
    {
        "key": "falk_2015_self_affirmation",
        "citation": "Falk et al. 2015",
        "title": "Self-affirmation alters the brain's response to health messages and subsequent behavior change",
        "url": "https://doi.org/10.1073/pnas.1500247112",
        "evidence_level": "within_subject_condition_contrast;individual_differences_in_sample",
        "modality": "fMRI; text+pictogram health messages after a values-affirmation task (N=45 with fMRI and accelerometer data; 67 enrolled)",
        "finding": "Affirming values before exposure increased VMPFC response to the messages, which predicted later activity increase; the manipulation was a pre-task, not message wording.",
    },
    {
        "key": "cooper_2017_value_connectivity",
        "citation": "Cooper et al. 2017",
        "title": "Coherent activity between brain regions that code for value is linked to the malleability of human behavior",
        "url": "https://doi.org/10.1038/srep43250",
        "evidence_level": "individual_differences_in_sample",
        "modality": "fMRI; affirmation dataset (N=44)",
        "finding": "Behavior change related to value-network coupling with a subcortical node (PPI p=.036), not to self-reflection MPFC/PCC activity or connectivity.",
    },
    {
        "key": "pegors_2017_rsa_mpfc",
        "citation": "Pegors et al. 2017",
        "title": "Predicting behavior change from persuasive messages using neural representational similarity and social network analyses",
        "url": "https://doi.org/10.1016/j.neuroimage.2017.05.063",
        "evidence_level": "individual_differences_in_sample",
        "modality": "fMRI; anti-smoking images (N=40 analysed, 50 scanned; reuses Cooper 2015 participants)",
        "finding": "MPFC pattern similarity to self-related and social-exclusion processing related to smoking change, with direction depending on the smoker's social network.",
    },
    {
        "key": "huskey_2017_persuasion_network",
        "citation": "Huskey et al. 2017",
        "title": "The persuasion network is modulated by drug-use risk and predicts anti-drug message effectiveness",
        "url": "https://doi.org/10.1093/scan/nsx126",
        "evidence_level": "within_subject_condition_contrast;message_level_aggregate",
        "modality": "fMRI; anti-drug video PSAs (N=28)",
        "finding": "Argument strength and viewer risk changed connectivity in a persuasion network; message-level connectivity predicted responses in independent samples.",
    },
    {
        "key": "chua_2009_tailoring",
        "citation": "Chua et al. 2009",
        "title": "Neural correlates of message tailoring and self-relatedness in smoking cessation programming",
        "url": "https://doi.org/10.1016/j.biopsych.2008.08.030",
        "evidence_level": "within_subject_condition_contrast",
        "modality": "fMRI; audio + on-screen smoking messages (N=24)",
        "finding": "High-tailored messages engaged rMPFC and precuneus/PCC (partly as less deactivation) versus low-tailored (N=24 motivated smokers, uncorrected thresholds); behavior was not tested. Small single-laboratory health sample with disclosed commercial tailoring interests.",
    },
    {
        "key": "cascio_2015_peer_recommendations",
        "citation": "Cascio et al. 2015",
        "title": "Neural correlates of susceptibility to group opinions in online word-of-mouth recommendations",
        "url": "https://doi.org/10.1509/jmr.13.0611",
        "evidence_level": "within_subject_item_level;individual_differences_in_sample",
        "modality": "fMRI; male adolescents re-rating apps after peer ratings (N=65)",
        "finding": "Conformity involved valuation regions within person and a between-person TPJ susceptibility signal; divergent peer opinions changed ratings far more than agreeing ones; individuals varied widely.",
    },
    {
        "key": "falk_2016_campaign_success",
        "citation": "Falk et al. 2016",
        "title": "Functional brain imaging predicts public health campaign success",
        "url": "https://doi.org/10.1093/scan/nsv108",
        "evidence_level": "message_level_aggregate",
        "modality": "fMRI; antismoking images+text (N=50); email campaign",
        "finding": "Aggregated self-localizer MPFC responses forecast population-level email campaign response (~400,000 emails), complementing self-report; the link was content-dependent.",
    },
]

RESEARCH_BASIS = [
    f"{source['citation']}: {source['finding']}"
    for source in RESEARCH_SOURCES
]

# source_keys list only the literature an axis is named after; sources that narrow its scope are
# cited in the caveat. These strings reach the LLM, so they never name subcortical structures.
NEURO_AXIS_META = {
    "self_value": {
        "label": "Self/value response proxy",
        "analogue": "Average-reader composite of whole-brain |prediction| shape (sustain, focus, late and peak ratios), named after mPFC/vmPFC self/value findings; computed from the text alone, identical for every persona",
        "caveat": "Literature prior: medial-prefrontal links to later behavior were individual-level, small-sample and content- and audience-dependent (Vezich 2017; Pegors 2017); tailoring effects came from messages built from each reader's own data, which TRIBE never sees (Chua 2011).",
        "source_keys": [
            "falk_2010_persuasion_change",
            "falk_2011_smoking_replication",
            "cooper_2015_self_value_mpfc",
            "falk_2016_campaign_success",
            "chua_2011_tailored_self_related",
            "falk_2015_self_affirmation",
            "falk_2012_neural_focus_group",
            "baek_scholz_2017_sharing",
            "chan_scholz_2023_cross_cultural_sharing",
        ],
    },
    "reward_affect": {
        "label": "Affective salience proxy",
        "analogue": "Average-reader composite of whole-brain |prediction| intensity and variability (valence-agnostic), named after reward/valuation findings in ad and message studies",
        "caveat": "Not liking, willingness to pay or purchase intent: measured responses were larger for objects later rated unfavourably (Klucharev 2008, photo study), and mOFC/dlPFC tracked value only during real bids on food photos (Plassmann 2007).",
        "source_keys": [
            "venkatraman_2015_ad_success",
            "chan_2024_ad_liking",
            "cohen_2024_reward_mentalizing",
            "scholz_chan_falk_2025_mega_analysis",
        ],
    },
    "social_sharing": {
        "label": "Social-cognition proxy",
        "analogue": "Average-reader composite of whole-brain |prediction| peaks, transitions and spread, named after dmPFC/pSTS/temporal-pole and TPJ social-cognition findings",
        "caveat": "Not a social-proof detector: norm-targeted text raised temporal-pole/TPJ/dMPFC responses whether or not it persuaded (Yomogida 2017), and dMPFC rose for dogmatic texts rated less persuasive (Hernández-Vergara 2025).",
        "source_keys": [
            "falk_2010_jocn_mentalizing_network",
            "baek_scholz_2017_sharing",
            "chan_scholz_2023_cross_cultural_sharing",
            "chan_2024_ad_liking",
            "cohen_2024_reward_mentalizing",
            "scholz_chan_falk_2025_mega_analysis",
        ],
    },
    "encoding_attention": {
        "label": "Encoding/attention proxy",
        "analogue": "Average-reader composite of whole-brain |prediction| arc, sustain, early and peak ratios; a salience/encoding heuristic, not a recall estimate",
        "caveat": "Memorability is not persuasion: tailored messages were remembered better but memory did not predict quitting (Chua 2011), and recall and attitude were nearly independent (Klucharev 2008).",
        "source_keys": ["chan_2024_ad_liking", "tribe_v2_foundation_model"],
    },
    "processing_fluency": {
        "label": "Processing-fluency proxy",
        "analogue": "Inverse of a whole-brain friction heuristic (low sustain/focus/spread and temporal noise of |prediction|)",
        "caveat": "A comprehension hint: persuasive argument text recruits more prefrontal activity than neutral text (Yomogida 2017), so a lower value may mean the draft argues something, not that it is hard to read.",
        "source_keys": ["alter_oppenheimer_2009_fluency", "cao_reimann_2020_triangulation"],
    },
}

SCIENTIFIC_CAVEATS = [
    "TRIBE returns predicted neural-response analogues, not measured fMRI from this recipient.",
    "TRIBE predictions are average-subject model outputs, not individualized recipient measurements.",
    "Signals and axes are average-reader statistics of a cortex-only prediction (fsaverage5): the six signals use whole-brain |prediction|, which carries no direction, and any ROI terms are signed means over atlas parcels of the same prediction; subcortical structures are not modelled and region names point to the literature, so reverse inference stays limited.",
    "The score is a neural prior from in-silico fMRI geometry, not a validated recipient-level outcome probability; its weights and thresholds are hand-set priors, and personalization fit and credibility follow the language model's read of the persona you described.",
    "TRIBE sees only the message text; recipient fit, source credibility, prior attitude and susceptibility to peers come only from the persona-aware semantic read.",
    "A higher, broader or more shared predicted response is not evidence of preference or persuasion: listeners who resisted a message responded more alike (Ntoumanis 2024), and measured activity was higher for unpersuasive, dogmatic or mismatched messages (Falk, Rameson 2010; Hernández-Vergara 2025; Di Plinio 2025).",
    "Neural weights are literature priors: most cited links predicted which people changed in small health-message studies; only a few predicted message-level outcomes (Falk 2012, 2016; Chan, Scholz 2023; Scholz 2025), all from measured, not model-predicted, responses.",
    "Direct text mode models hearing the pitch read aloud at ~150 wpm, not the recipient's self-paced reading.",
]


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return lo
    if math.isnan(numeric) or math.isinf(numeric):
        return lo
    return max(lo, min(hi, numeric))


def _round(value: float, digits: int = 1) -> float:
    return round(clamp(value), digits)


def _signal(neural_signals: dict[str, float], key: str, default: float = 50.0) -> float:
    try:
        return clamp(float(neural_signals.get(key, default)))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


QUALITY_WARNING_CAPS = {
    "near_zero_prediction_response": 0.35,
    "low_voxel_count_prediction": 0.55,
    "low_temporal_resolution": 0.65,
    "flat_temporal_trace": 0.72,
}


def calibration_quality_weight(evidence: dict[str, Any] | None = None) -> float:
    """Return how strongly neural evidence should pull away from neutral.

    Scientific caveats such as average-subject prediction and synthetic word
    order are exposed to users, but they are not score penalties by themselves.
    This weight only shrinks the score when the prediction matrix is too small,
    flat, near-zero, or otherwise weak as evidence.
    """
    if not isinstance(evidence, dict):
        return 1.0

    diagnostics = evidence.get("calibration_quality")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    warning_values = evidence.get("warnings") or []
    if not isinstance(warning_values, (list, tuple, set)):
        warning_values = [warning_values]
    diagnostic_warning_values = diagnostics.get("warnings") or []
    if not isinstance(diagnostic_warning_values, (list, tuple, set)):
        diagnostic_warning_values = [diagnostic_warning_values]
    warnings = {
        str(warning)
        for warning in warning_values
        if warning
    }
    warnings.update(
        str(warning)
        for warning in diagnostic_warning_values
        if warning
    )

    weight = 1.0
    for warning, cap in QUALITY_WARNING_CAPS.items():
        if warning in warnings:
            weight = min(weight, cap)

    segments = _safe_int(diagnostics.get("segments"), 0)
    if segments == 1:
        weight = min(weight, 0.55)
    elif segments == 2:
        weight = min(weight, 0.65)

    voxel_count = _safe_int(diagnostics.get("voxel_count"), 0)
    if 0 < voxel_count < 1000:
        weight = min(weight, 0.55)

    global_mean_abs = _safe_float(diagnostics.get("global_mean_abs"), 0.0)
    global_peak_abs = _safe_float(diagnostics.get("global_peak_abs"), 0.0)
    if (
        ("global_mean_abs" in diagnostics or "global_peak_abs" in diagnostics)
        and (global_mean_abs <= 1e-7 or global_peak_abs <= 1e-7)
    ):
        weight = min(weight, QUALITY_WARNING_CAPS["near_zero_prediction_response"])

    temporal_std_ratio = _safe_float(diagnostics.get("temporal_std_ratio"), 1.0)
    arc_ratio = _safe_float(diagnostics.get("arc_ratio"), 1.0)
    if segments > 2 and temporal_std_ratio < 0.02 and arc_ratio < 0.05:
        weight = min(weight, QUALITY_WARNING_CAPS["flat_temporal_trace"])

    return clamp(weight, 0.25, 1.0)


def quality_adjusted_score(score: float, evidence: dict[str, Any] | None = None) -> float:
    """Shrink low-quality neural evidence toward neutral instead of overclaiming."""
    weight = calibration_quality_weight(evidence)
    return clamp(50.0 + (clamp(score) - 50.0) * weight)


def analyze_persuasion_text(message: str, persona: str, platform: str = "general") -> dict[str, Any]:
    """Return compatibility metadata without text scoring.

    The previous implementation used keyword and regex heuristics.  That path is
    deliberately disabled so the prompt and final calibration are not biased by
    brittle text features or prompt-injection-like surface forms.
    """
    platform_key = (platform or "general").strip().lower()
    return {
        "overall_text_score": 50.0,
        "feature_scores": {},
        "detected_strategies": [],
        "missing_elements": [],
        "warnings": [],
        "prompt_injection_risk": 0.0,
        "readability": {},
        "matched_persona_terms": [],
        "strategy_counts": {},
        "platform": platform_key,
        "input_metadata": {
            "message_character_count": len(message or ""),
            "persona_character_count": len(persona or ""),
        },
        "methodology_version": CALIBRATION_METHODOLOGY_VERSION,
        "methodology": "text_heuristics_removed_neural_only_calibration",
        "research_basis": RESEARCH_BASIS,
        "research_sources": research_sources(),
    }


def neural_score_from_signals(neural_signals: dict[str, float]) -> float:
    """Convert six TRIBE-derived signal scores to a conservative neural prior."""
    ee = _signal(neural_signals, "emotional_engagement")
    pr = _signal(neural_signals, "personal_relevance")
    sp = _signal(neural_signals, "social_proof_potential")
    mem = _signal(neural_signals, "memorability")
    ac = _signal(neural_signals, "attention_capture")
    cf = _signal(neural_signals, "cognitive_friction")
    return clamp(
        pr * 0.25
        + ee * 0.20
        + mem * 0.18
        + ac * 0.15
        + sp * 0.12
        + (100.0 - cf) * 0.10
    )


def neuro_axes_from_analysis(neural_signals: dict[str, float], evidence: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Build neuro-persuasive axes using TRIBE-derived neural signals only."""
    del evidence
    ee = _signal(neural_signals, "emotional_engagement")
    pr = _signal(neural_signals, "personal_relevance")
    sp = _signal(neural_signals, "social_proof_potential")
    mem = _signal(neural_signals, "memorability")
    ac = _signal(neural_signals, "attention_capture")
    cf = _signal(neural_signals, "cognitive_friction")
    inv_cf = 100.0 - cf

    raw_scores = {
        "self_value": clamp(pr * 0.55 + ee * 0.18 + mem * 0.15 + inv_cf * 0.12),
        "reward_affect": clamp(ee * 0.48 + pr * 0.20 + ac * 0.18 + mem * 0.14),
        "social_sharing": clamp(sp * 0.50 + pr * 0.18 + ee * 0.14 + mem * 0.10 + ac * 0.08),
        "encoding_attention": clamp(mem * 0.42 + ac * 0.34 + ee * 0.14 + inv_cf * 0.10),
        "processing_fluency": clamp(inv_cf * 0.62 + mem * 0.14 + pr * 0.14 + ac * 0.10),
    }

    evidence_lines = {
        "self_value": [
            f"personal_relevance={pr:.0f}",
            f"emotional_engagement={ee:.0f}",
            f"memorability={mem:.0f}",
        ],
        "reward_affect": [
            f"emotional_engagement={ee:.0f}",
            f"personal_relevance={pr:.0f}",
            f"attention_capture={ac:.0f}",
        ],
        "social_sharing": [
            f"social_cognition_proxy={sp:.0f}",
            f"personal_relevance={pr:.0f}",
            f"emotional_engagement={ee:.0f}",
        ],
        "encoding_attention": [
            f"memorability={mem:.0f}",
            f"attention_capture={ac:.0f}",
            f"emotional_engagement={ee:.0f}",
        ],
        "processing_fluency": [
            f"inverse_cognitive_friction={inv_cf:.0f}",
            f"memorability={mem:.0f}",
            f"personal_relevance={pr:.0f}",
        ],
    }

    return {
        key: {
            "label": NEURO_AXIS_META[key]["label"],
            "score": _round(score),
            "weight": NEURO_AXIS_WEIGHTS[key],
            "contribution": round((score - 50.0) * NEURO_AXIS_WEIGHTS[key], 1),
            "analogue": NEURO_AXIS_META[key]["analogue"],
            "evidence": evidence_lines[key],
            "caveat": NEURO_AXIS_META[key]["caveat"],
            "source_keys": NEURO_AXIS_META[key]["source_keys"],
        }
        for key, score in raw_scores.items()
    }


def neuro_axis_score_from_axes(axes: dict[str, dict[str, Any]]) -> float:
    score = 0.0
    total_weight = 0.0
    for key, weight in NEURO_AXIS_WEIGHTS.items():
        item = axes.get(key, {})
        score += clamp(_safe_float(item.get("score", 50.0), 50.0)) * weight
        total_weight += weight
    return clamp(score / max(total_weight, 1e-9))


# Literature prior, UNVALIDATED: direction of the closest message-level finding.
ROI_PRIOR_WEIGHT = clamp(_safe_float(os.getenv("PITCHCHECK_ROI_PRIOR_WEIGHT"), 0.15), 0.0, 0.5)
ROI_ENCODER_MISMATCH_FACTOR = clamp(_safe_float(os.getenv("PITCHCHECK_ROI_ENCODER_MISMATCH_FACTOR"), 0.5), 0.0, 1.0)
SPECIFICITY_SCORE_WEIGHT = max(0.0, _safe_float(os.getenv("PITCHCHECK_SPECIFICITY_SCORE_WEIGHT"), 1.5))
SPECIFICITY_Z_MIN = max(0.0, _safe_float(os.getenv("PITCHCHECK_SPECIFICITY_Z_MIN"), 1.0))
SPECIFICITY_REFINE_PENALTY = clamp(_safe_float(os.getenv("PITCHCHECK_SPECIFICITY_REFINE_PENALTY"), 2.0), 0.0, 2.0)


def specificity_prior_points(contrast: dict | None) -> float:
    """Unvalidated literature prior: persona grounding, z dead zone, then a signed ±3 ceiling."""
    if not contrast or contrast.get("available") is not True or contrast.get("grounded") is not True:
        return 0.0
    z = _safe_float(contrast.get("medial_z"), 0.0)
    excess = max(0.0, min(abs(z), SPECIFICITY_Z_MIN + 2.0) - SPECIFICITY_Z_MIN)
    return round(clamp(math.copysign(excess, z) * SPECIFICITY_SCORE_WEIGHT, -3.0, 3.0), 2)


# shortcut: provisional unit defaults; replace with per-encoder benchmark/Stage-0 artifacts after a GPU run.
ROI_Z_SCALE = 1.0
ROI_UNIT_PROVENANCE = "provisional_unit_defaults:ws07;gpu_not_run"
ROI_PRIOR = {
    "vmpfc_frontopolar": (+1, 0.30, 0.0, ("falk_2010_persuasion_change", "vezich_2017_framing_action")),
    "dmpfc": (+1, 0.20, 0.0, ("falk_2010_jocn_mentalizing_network", "chua_2011_tailored_self_related")),
    "r_ifg_ai": (-1, 0.20, 0.0, ("hernandez_vergara_2025_dogmatic_tone",)),
    "mofc": (+1, 0.15, 0.0, ("plassmann_2007_wtp_ofc", "hernandez_vergara_2025_dogmatic_tone")),
    "insula": (-1, 0.15, 0.0, ("knutson_2007_neural_purchase",)),
    "pcc_precuneus": (+1, 0.05, 0.0, ("chua_2009_tailoring", "chua_2011_tailored_self_related")),
}


def roi_prior_from_summary(summary, text_feature_compatible=None, multipliers=None, encoder=None):
    """K4: multipliers shrink only the numerator; a neutral prior leaves the anchor unchanged."""
    rois = summary.get("rois") if isinstance(summary, dict) else None
    if ROI_PRIOR_WEIGHT <= 0 or not isinstance(rois, dict):
        return None
    avail = [k for k in ROI_PRIOR if isinstance(rois.get(k), dict) and
             isinstance(rois[k].get("z"), (int, float)) and math.isfinite(rois[k]["z"])]
    if not avail:
        return None
    calibration = roi_features.roi_calibration(encoder)
    centers = {k: _safe_float(calibration.get("centers", {}).get(k), ROI_PRIOR[k][2]) for k in avail}
    scale = _safe_float(calibration.get("ROI_Z_SCALE"), ROI_Z_SCALE)
    if scale <= 1e-12:
        scale = ROI_Z_SCALE
    m = {k: clamp(_safe_float((multipliers or {}).get(k), 1.0), 0.0, 1.0) for k in avail}
    terms = {k: ROI_PRIOR[k][0] * ROI_PRIOR[k][1] * m[k] * (rois[k]["z"] - centers[k]) for k in avail}
    raw = sum(terms.values()) / sum(ROI_PRIOR[k][1] for k in avail)
    mismatch = text_feature_compatible is False
    weight = ROI_PRIOR_WEIGHT * (ROI_ENCODER_MISMATCH_FACTOR if mismatch else 1.0)
    return {"score": round(50 + 50 * math.tanh(raw / scale), 1), "raw": round(raw, 4),
            "weight": round(weight, 4), "applied": weight > 0, "multipliers": m,
            "terms": {k: round(v, 4) for k, v in terms.items()}, "version": "roi_prior_v1",
            "centers": centers, "ROI_Z_SCALE": scale, "encoder": encoder,
            "unit_calibration_provenance": calibration.get("provenance", ROI_UNIT_PROVENANCE),
            "provenance": "literature_prior_unvalidated" + (";encoder_mismatch" if mismatch else "")}


def evidence_score_from_analysis(neural_signals: dict[str, float], evidence: dict[str, Any] | None = None) -> float:
    axis_score = neuro_axis_score_from_axes(neuro_axes_from_analysis(neural_signals, evidence))
    roi = evidence.get("roi_prior") if isinstance(evidence, dict) else None
    if isinstance(roi, dict) and roi.get("applied"):
        axis_score = clamp(axis_score + roi["weight"] * (clamp(roi["score"]) - 50))
    return quality_adjusted_score(axis_score, evidence)


def calibration_confidence(neural_score: float, text_score: float, evidence: dict[str, Any]) -> float:
    """Estimate confidence from neural signal strength and prediction quality.

    ``text_score`` is accepted for backward compatibility and ignored.
    """
    del text_score
    distance_from_mid = abs(clamp(neural_score) - 50.0) / 50.0
    confidence = 0.62 + distance_from_mid * 0.22
    quality_weight = calibration_quality_weight(evidence)
    if quality_weight < 0.99:
        confidence = min(confidence, 0.45 + quality_weight * 0.38)
    return max(0.35, min(0.90, confidence))


def confidence_reasons(
    neural_score: float,
    text_score: float,
    evidence: dict[str, Any],
    axes: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    del text_score
    reasons = [
        "text_heuristic_audit_disabled",
        "tribe_predicted_fmri_primary",
        "reverse_inference_caveat_applied",
    ]
    quality_weight = calibration_quality_weight(evidence)
    if quality_weight < 0.75:
        reasons.append("low_prediction_quality_shrunk_to_neutral")
    elif quality_weight < 0.99:
        reasons.append("prediction_quality_caveat_applied")

    warnings = evidence.get("warnings", []) if isinstance(evidence, dict) else []
    if not isinstance(warnings, (list, tuple)):
        warnings = [warnings] if warnings else []
    for warning in warnings[:3]:
        reasons.append(f"warning_{warning}")

    if axes:
        strongest = max(axes.items(), key=lambda item: float(item[1].get("score", 50.0)))[0]
        weakest = min(axes.items(), key=lambda item: float(item[1].get("score", 50.0)))[0]
        reasons.append(f"strongest_neural_axis_{strongest}")
        reasons.append(f"weakest_neural_axis_{weakest}")
    if neural_score >= 70:
        reasons.append("high_neural_persuasion_prior")
    elif neural_score <= 40:
        reasons.append("low_neural_persuasion_prior")
    else:
        reasons.append("moderate_neural_persuasion_prior")
    return sorted(set(reasons))


def scientific_caveats() -> list[str]:
    return SCIENTIFIC_CAVEATS[:]


def research_sources() -> list[dict[str, str]]:
    return [dict(source) for source in RESEARCH_SOURCES]
