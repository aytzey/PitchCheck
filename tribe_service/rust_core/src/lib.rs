use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

const DIRECT_TEXT_TIMING_BASIS: &str = "synthetic_word_order";
const REAL_TIME_TIMING_BASIS: &str = "real_time_seconds";
const TRIBE_PREDICTION_SUBJECT_BASIS: &str = "average_subject";
const TRIBE_CORTICAL_MESH: &str = "fsaverage5";
const TRIBE_HEMODYNAMIC_LAG_SECONDS: f64 = 5.0;
const TRIBE_RESPONSE_KIND: &str = "tribe_predicted_fmri_analogue";

#[derive(Debug, Clone)]
struct PredictionStats {
    segments: usize,
    voxels: usize,
    global_mean_abs: f64,
    global_peak_abs: f64,
    temporal_means: Vec<f64>,
    temporal_peaks: Vec<f64>,
    spatial_means: Vec<f64>,
}

#[derive(Debug, Clone)]
struct FeatureSet {
    global_mean_abs: f64,
    global_peak_abs: f64,
    temporal_std: f64,
    early_mean: f64,
    late_mean: f64,
    max_temporal_delta: f64,
    spatial_spread: f64,
    focus_ratio: f64,
    sustain_ratio: f64,
    arc_ratio: f64,
}

#[derive(Debug, Clone)]
struct SignalSet {
    emotional_engagement: f64,
    personal_relevance: f64,
    social_proof_potential: f64,
    memorability: f64,
    attention_capture: f64,
    cognitive_friction: f64,
}

fn finite_abs(value: f32) -> f64 {
    f64::from(value).abs()
}

fn clamp(value: f64, lo: f64, hi: f64) -> f64 {
    if !value.is_finite() {
        return lo;
    }
    value.max(lo).min(hi)
}

fn safe_ratio(numerator: f64, denominator: f64, default: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() <= 1e-9 {
        return default;
    }
    numerator / denominator
}

fn band_score(value: f64, lo: f64, hi: f64) -> f64 {
    if (hi - lo).abs() < 1e-9 {
        return 50.0;
    }
    clamp((value - lo) / (hi - lo) * 100.0, 0.0, 100.0)
}

fn weighted_signal(scores: &[(f64, f64)], floor: f64, ceiling: f64) -> f64 {
    let mut total_weight = 0.0;
    let mut weighted_sum = 0.0;
    for (score, weight) in scores {
        let sanitized_score = clamp(*score, 0.0, 100.0);
        let sanitized_weight = clamp(*weight, 0.0, 1_000_000.0).max(0.0);
        total_weight += sanitized_weight;
        weighted_sum += sanitized_score * sanitized_weight;
    }
    if total_weight < 1e-9 {
        return 50.0;
    }
    let raw = weighted_sum / total_weight;
    clamp(
        floor + (ceiling - floor) * (clamp(raw, 0.0, 100.0) / 100.0),
        0.0,
        100.0,
    )
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_population(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let avg = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - avg;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn round_to(value: f64, digits: i32) -> f64 {
    let factor = 10_f64.powi(digits);
    (value * factor).round() / factor
}

fn timing_metadata(text_input_mode: Option<&str>) -> (&'static str, &'static str, &'static str) {
    let mode = text_input_mode.unwrap_or("").trim().to_ascii_lowercase();
    if mode == "direct" {
        return (
            DIRECT_TEXT_TIMING_BASIS,
            "synthetic word-order segment",
            "Direct text mode skips TTS/WhisperX. Segment order follows the pitch text; segment positions are not real elapsed seconds.",
        );
    }
    (
        REAL_TIME_TIMING_BASIS,
        "second",
        "Audio/TTS timing path with speech-event alignment; segment positions are time-based.",
    )
}

fn collect_stats(predictions: PyReadonlyArray2<'_, f32>) -> PyResult<PredictionStats> {
    let shape = predictions.shape();
    let segments = shape[0];
    let voxels = shape[1];
    if segments == 0 || voxels == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Prediction matrix is empty",
        ));
    }
    let values = predictions.as_slice()?;

    let mut temporal_means = Vec::with_capacity(segments);
    let mut temporal_peaks = Vec::with_capacity(segments);
    let mut spatial_sums = vec![0.0; voxels];
    let mut global_sum = 0.0;
    let mut global_peak_abs = 0.0;

    for row in values.chunks_exact(voxels) {
        let mut row_sum = 0.0;
        let mut row_peak = 0.0;
        for (col_idx, value) in row.iter().enumerate() {
            let abs_value = finite_abs(*value);
            row_sum += abs_value;
            spatial_sums[col_idx] += abs_value;
            global_sum += abs_value;
            if abs_value > row_peak {
                row_peak = abs_value;
            }
            if abs_value > global_peak_abs {
                global_peak_abs = abs_value;
            }
        }
        temporal_means.push(row_sum / voxels as f64);
        temporal_peaks.push(row_peak);
    }

    let spatial_means = spatial_sums
        .into_iter()
        .map(|sum| sum / segments as f64)
        .collect::<Vec<_>>();

    Ok(PredictionStats {
        segments,
        voxels,
        global_mean_abs: global_sum / (segments * voxels) as f64,
        global_peak_abs,
        temporal_means,
        temporal_peaks,
        spatial_means,
    })
}

fn extract_feature_set(stats: &PredictionStats) -> FeatureSet {
    let n_segments = stats.segments;
    let n_voxels = stats.voxels;
    let temporal_mean = mean(&stats.temporal_means);

    let temporal_std = if n_segments > 1 {
        std_population(&stats.temporal_means)
    } else {
        0.0
    };

    let q1 = (n_segments / 4).max(1);
    let early_mean = mean(&stats.temporal_means[..q1]);
    let late_slice = if q1 < n_segments {
        &stats.temporal_means[n_segments - q1..]
    } else {
        &stats.temporal_means[..]
    };
    let late_mean = mean(late_slice);

    let max_temporal_delta = if n_segments > 1 {
        stats
            .temporal_means
            .windows(2)
            .map(|window| (window[1] - window[0]).abs())
            .fold(0.0, f64::max)
    } else {
        0.0
    };

    let spatial_mean = mean(&stats.spatial_means);
    let spatial_spread = if n_voxels > 1 {
        stats
            .spatial_means
            .iter()
            .filter(|value| **value > spatial_mean)
            .count() as f64
            / n_voxels as f64
    } else {
        0.0
    };

    let top_k = (n_voxels / 10).max(1);
    let mut sorted_spatial = stats.spatial_means.clone();
    let top_start = n_voxels - top_k;
    sorted_spatial.select_nth_unstable_by(top_start, |a, b| a.total_cmp(b));
    let top_mean = mean(&sorted_spatial[n_voxels - top_k..]);
    let focus_ratio = safe_ratio(top_mean, stats.global_mean_abs, 1.0);

    let sustain_ratio = if n_segments > 1 {
        stats
            .temporal_means
            .iter()
            .filter(|value| **value >= temporal_mean)
            .count() as f64
            / n_segments as f64
    } else {
        0.5
    };

    let arc_ratio = if n_segments > 1 {
        let min_value = stats
            .temporal_means
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_value = stats
            .temporal_means
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        safe_ratio(max_value - min_value, temporal_mean, 0.0)
    } else {
        0.0
    };

    FeatureSet {
        global_mean_abs: stats.global_mean_abs,
        global_peak_abs: stats.global_peak_abs,
        temporal_std,
        early_mean,
        late_mean,
        max_temporal_delta,
        spatial_spread,
        focus_ratio,
        sustain_ratio,
        arc_ratio,
    }
}

fn derive_signals(features: &FeatureSet) -> SignalSet {
    let gma = features.global_mean_abs.max(1e-9);
    let peak_r = features.global_peak_abs.max(0.0) / gma;
    let ts_r = features.temporal_std.max(0.0) / gma;
    let early_r = features.early_mean.max(0.0) / gma;
    let late_r = features.late_mean.max(0.0) / gma;
    let delta_r = features.max_temporal_delta.max(0.0) / gma;
    let ss = clamp(features.spatial_spread, 0.0, 1.0);
    let fr = features.focus_ratio.max(0.0);
    let sr = clamp(features.sustain_ratio, 0.0, 1.0);
    let ar = features.arc_ratio.max(0.0);

    let emotional_engagement = weighted_signal(
        &[
            (band_score(peak_r, 5.0, 12.0), 0.35),
            (band_score(ts_r, 0.05, 0.5), 0.25),
            (band_score(ar, 0.1, 0.8), 0.20),
            (band_score(delta_r, 0.1, 1.0), 0.20),
        ],
        8.0,
        92.0,
    );

    let personal_relevance = weighted_signal(
        &[
            (band_score(sr, 0.4, 0.75), 0.35),
            (band_score(fr, 2.0, 5.0), 0.30),
            (band_score(late_r, 0.8, 1.3), 0.20),
            (band_score(peak_r, 6.0, 10.0), 0.15),
        ],
        8.0,
        92.0,
    );

    let social_proof_potential = weighted_signal(
        &[
            (band_score(peak_r, 6.5, 11.0), 0.35),
            (band_score(delta_r, 0.15, 0.8), 0.25),
            (band_score(ts_r, 0.08, 0.4), 0.20),
            (band_score(ss, 0.25, 0.42), 0.20),
        ],
        8.0,
        92.0,
    );

    let memorability = weighted_signal(
        &[
            (band_score(ar, 0.15, 0.65), 0.30),
            (band_score(sr, 0.45, 0.75), 0.25),
            (band_score(peak_r, 6.0, 10.0), 0.25),
            (band_score(fr, 2.5, 4.5), 0.20),
        ],
        8.0,
        92.0,
    );

    let attention_capture = weighted_signal(
        &[
            (band_score(early_r, 0.85, 1.25), 0.35),
            (band_score(peak_r, 6.0, 11.0), 0.30),
            (band_score(ss, 0.25, 0.42), 0.20),
            (band_score(delta_r, 0.1, 0.7), 0.15),
        ],
        8.0,
        92.0,
    );

    let cognitive_friction = weighted_signal(
        &[
            (100.0 - band_score(sr, 0.35, 0.7), 0.35),
            (100.0 - band_score(fr, 1.8, 4.0), 0.30),
            (100.0 - band_score(ss, 0.22, 0.40), 0.20),
            (band_score(ts_r, 0.3, 0.6), 0.15),
        ],
        4.0,
        84.0,
    );

    SignalSet {
        emotional_engagement,
        personal_relevance,
        social_proof_potential,
        memorability,
        attention_capture,
        cognitive_friction,
    }
}

fn features_to_dict<'py>(py: Python<'py>, features: &FeatureSet) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("global_mean_abs", features.global_mean_abs)?;
    dict.set_item("global_peak_abs", features.global_peak_abs)?;
    dict.set_item("temporal_std", features.temporal_std)?;
    dict.set_item("early_mean", features.early_mean)?;
    dict.set_item("late_mean", features.late_mean)?;
    dict.set_item("max_temporal_delta", features.max_temporal_delta)?;
    dict.set_item("spatial_spread", features.spatial_spread)?;
    dict.set_item("focus_ratio", features.focus_ratio)?;
    dict.set_item("sustain_ratio", features.sustain_ratio)?;
    dict.set_item("arc_ratio", features.arc_ratio)?;
    Ok(dict)
}

fn signals_to_dict<'py>(py: Python<'py>, signals: &SignalSet) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("emotional_engagement", signals.emotional_engagement)?;
    dict.set_item("personal_relevance", signals.personal_relevance)?;
    dict.set_item("social_proof_potential", signals.social_proof_potential)?;
    dict.set_item("memorability", signals.memorability)?;
    dict.set_item("attention_capture", signals.attention_capture)?;
    dict.set_item("cognitive_friction", signals.cognitive_friction)?;
    Ok(dict)
}

fn summary_to_dict<'py>(
    py: Python<'py>,
    stats: &PredictionStats,
    text_input_mode: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let (trace_basis, segment_label, trace_note) = timing_metadata(text_input_mode);
    let dict = PyDict::new_bound(py);
    dict.set_item("segments", stats.segments)?;
    dict.set_item("voxel_count", stats.voxels)?;
    dict.set_item("global_mean_abs", stats.global_mean_abs)?;
    dict.set_item("global_peak_abs", stats.global_peak_abs)?;
    dict.set_item(
        "temporal_trace",
        stats
            .temporal_means
            .iter()
            .map(|value| round_to(*value, 4))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "temporal_peaks",
        stats
            .temporal_peaks
            .iter()
            .map(|value| round_to(*value, 4))
            .collect::<Vec<_>>(),
    )?;

    let mut ranked = stats
        .spatial_means
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    let rank_desc = |left: &(usize, f64), right: &(usize, f64)| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    };
    let top_n = stats.voxels.min(6);
    if top_n < ranked.len() {
        ranked.select_nth_unstable_by(top_n, rank_desc);
    }
    ranked[..top_n].sort_by(rank_desc);
    let top_voxel_indices = ranked
        .iter()
        .take(top_n)
        .map(|(idx, _)| *idx)
        .collect::<Vec<_>>();
    let top_voxel_values = ranked
        .iter()
        .take(top_n)
        .map(|(_, value)| round_to(*value, 4))
        .collect::<Vec<_>>();

    dict.set_item("top_voxel_indices", top_voxel_indices)?;
    dict.set_item("top_voxel_values", top_voxel_values)?;
    dict.set_item("response_kind", TRIBE_RESPONSE_KIND)?;
    dict.set_item("prediction_subject_basis", TRIBE_PREDICTION_SUBJECT_BASIS)?;
    dict.set_item("cortical_mesh", TRIBE_CORTICAL_MESH)?;
    dict.set_item("hemodynamic_lag_seconds", TRIBE_HEMODYNAMIC_LAG_SECONDS)?;
    dict.set_item("temporal_trace_basis", trace_basis)?;
    dict.set_item("temporal_segment_label", segment_label)?;
    dict.set_item("temporal_trace_note", trace_note)?;
    Ok(dict)
}

#[pyfunction]
fn extract_features(py: Python<'_>, predictions: PyReadonlyArray2<'_, f32>) -> PyResult<PyObject> {
    let stats = collect_stats(predictions)?;
    let features = extract_feature_set(&stats);
    Ok(features_to_dict(py, &features)?.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (predictions, text_input_mode=None))]
fn summarize_fmri_output(
    py: Python<'_>,
    predictions: PyReadonlyArray2<'_, f32>,
    text_input_mode: Option<&str>,
) -> PyResult<PyObject> {
    let stats = collect_stats(predictions)?;
    Ok(summary_to_dict(py, &stats, text_input_mode)?.into_py(py))
}

#[pyfunction]
fn derive_persuasion_signals(py: Python<'_>, features: &Bound<'_, PyDict>) -> PyResult<PyObject> {
    let get = |key: &str, default: f64| -> f64 {
        features
            .get_item(key)
            .ok()
            .flatten()
            .and_then(|value| value.extract::<f64>().ok())
            .filter(|value| value.is_finite())
            .unwrap_or(default)
    };
    let feature_set = FeatureSet {
        global_mean_abs: get("global_mean_abs", 0.01),
        global_peak_abs: get("global_peak_abs", 0.0),
        temporal_std: get("temporal_std", 0.0),
        early_mean: get("early_mean", 0.0),
        late_mean: get("late_mean", 0.0),
        max_temporal_delta: get("max_temporal_delta", 0.0),
        spatial_spread: get("spatial_spread", 0.0),
        focus_ratio: get("focus_ratio", 1.0),
        sustain_ratio: get("sustain_ratio", 0.5),
        arc_ratio: get("arc_ratio", 0.0),
    };
    let signals = derive_signals(&feature_set);
    Ok(signals_to_dict(py, &signals)?.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (predictions, text_input_mode=None))]
fn prediction_analysis(
    py: Python<'_>,
    predictions: PyReadonlyArray2<'_, f32>,
    text_input_mode: Option<&str>,
) -> PyResult<PyObject> {
    let stats = collect_stats(predictions)?;
    let features = extract_feature_set(&stats);
    let signals = derive_signals(&features);
    let dict = PyDict::new_bound(py);
    dict.set_item("raw_features", features_to_dict(py, &features)?)?;
    dict.set_item(
        "fmri_summary",
        summary_to_dict(py, &stats, text_input_mode)?,
    )?;
    dict.set_item("neural_signals", signals_to_dict(py, &signals)?)?;
    Ok(dict.into_py(py))
}

#[pyfunction]
fn extract_balanced_json_object(content: &str) -> Option<String> {
    let start = content.find('{')?;
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escape = false;

    for (offset, ch) in content[start..].char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
        } else if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
            if depth == 0 {
                return Some(content[start..start + offset + ch.len_utf8()].to_string());
            }
        }
    }
    None
}

#[pymodule]
fn _pitchcheck_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_fmri_output, m)?)?;
    m.add_function(wrap_pyfunction!(derive_persuasion_signals, m)?)?;
    m.add_function(wrap_pyfunction!(prediction_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(extract_balanced_json_object, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_signal_compresses_midpoint() {
        let score = weighted_signal(&[(100.0, 1.0), (0.0, 1.0)], 8.0, 92.0);
        assert!((score - 50.0).abs() < 1e-9);
    }

    #[test]
    fn derive_signals_stays_in_range() {
        let features = FeatureSet {
            global_mean_abs: 0.25,
            global_peak_abs: 0.6,
            temporal_std: 0.08,
            early_mean: 0.2,
            late_mean: 0.3,
            max_temporal_delta: 0.12,
            spatial_spread: 0.07,
            focus_ratio: 0.35,
            sustain_ratio: 0.6,
            arc_ratio: 1.2,
        };
        let signals = derive_signals(&features);
        let values = [
            signals.emotional_engagement,
            signals.personal_relevance,
            signals.social_proof_potential,
            signals.memorability,
            signals.attention_capture,
            signals.cognitive_friction,
        ];
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().all(|value| (0.0..=100.0).contains(value)));
    }

    #[test]
    fn json_extractor_handles_strings_and_nesting() {
        let content = r#"prefix {"a": "{not end}", "b": {"c": 1}} suffix"#;
        assert_eq!(
            extract_balanced_json_object(content).as_deref(),
            Some(r#"{"a": "{not end}", "b": {"c": 1}}"#)
        );
    }
}
