use std::collections::HashSet;
use serde::{ ser::SerializeStruct, Serialize };
use std::time::Duration;
use cggmp21::progress::{ PerfReport, RoundDuration, StageDuration };
use uuid::Uuid;

pub fn generate_uuid_v5_from_execution_id(execution_id: &[u8]) -> Uuid {
    Uuid::new_v5(&Uuid::NAMESPACE_DNS, &execution_id)
}

pub fn all_unique<T: Eq + std::hash::Hash>(arr: &[T]) -> bool {
    let mut seen = HashSet::new();
    for item in arr {
        if !seen.insert(item) {
            return false; // If insert returns false, the item is already in the set
        }
    }
    true
}

pub fn generate_array_i16(n: usize) -> Vec<i16> {
    (0..n).map(|x| x as i16).collect()
}

pub fn generate_array_u16(n: usize) -> Vec<u16> {
    (0..n).map(|x| x as u16).collect()
}

#[derive(Debug, Clone)]

pub struct MyPerfReport(pub PerfReport);

// Helper function to format duration and percentage for computation and I/O
fn json_duration_with_percent(duration: Duration, total: Duration) -> String {
    format!("{:.2?} ({:.1}%)", duration, percentage(duration, total))
}
fn percentage(part: Duration, total: Duration) -> f64 {
    (part.as_secs_f64() / total.as_secs_f64()) * 100.0
}
fn json_io(
    total_io: Duration,
    total: Duration,
    total_send: Duration,
    total_recv: Duration
) -> serde_json::Value {
    serde_json::json!({
        "total_io": format!("{:.2?} ({:.1}%)", total_io, percentage(total_io, total)),
        "send": format!("{:.2?}", total_send),
        "receive": format!("{:.2?}", total_recv)
    })
}

fn json_stage(stage: &StageDuration, total: Duration) -> serde_json::Value {
    serde_json::json!({
        "stage_name": stage.name,
        "duration": format!("{:.2?}", stage.duration),
        "percentage": format!("{:.1}%", percentage(stage.duration, total))
    })
}

fn json_round(index: usize, round: &RoundDuration) -> serde_json::Value {
    let total_duration = round.computation + round.sending + round.receiving;

    let stages_json: Vec<_> = round.stages
        .iter()
        .map(|stage| json_stage(stage, total_duration))
        .collect();

    // Calculate the total duration of the stages and the "unstaged" time
    let stages_total = round.stages
        .iter()
        .map(|s| s.duration)
        .sum::<Duration>();
    let unstaged = round.computation - stages_total;

    serde_json::json!({
        "round_name": round.round_name.unwrap_or(&format!("Round {}", index)),
        "total_duration": format!("{:.2?}", total_duration),
        "computation": format!("{:.2?} ({:.1}%)", round.computation, percentage(round.computation, total_duration)),
        "stages": stages_json,
        "unstaged": format!("{:.2?} ({:.1}%)", unstaged, percentage(unstaged, total_duration)),
        "io": json_io(round.sending + round.receiving, total_duration, round.sending, round.receiving)
    })
}

impl Serialize for MyPerfReport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: serde::Serializer {
        let mut state = serializer.serialize_struct("PerfReport", 3)?;
        let total_computation =
            self.0.setup +
            self.0.rounds
                .iter()
                .map(|r| r.computation)
                .sum::<Duration>();
        let total_send = self.0.rounds
            .iter()
            .map(|r| r.sending)
            .sum::<Duration>();

        let total_recv = self.0.rounds
            .iter()
            .map(|r| r.receiving)
            .sum::<Duration>();

        let total_io = total_send + total_recv;
        let total = total_computation + total_io;
        state.serialize_field("total_duration", &format!("{:.2?}", total))?;
        state.serialize_field(
            "computation",
            &json_duration_with_percent(total_computation, total)
        )?;
        state.serialize_field("io", &json_io(total_io, total, total_send, total_recv))?;

        let stages_json: Vec<_> = self.0.setup_stages
            .iter()
            .map(|stage| json_stage(stage, self.0.setup))
            .collect();
        state.serialize_field("setup_stages", &stages_json)?;

        let rounds_json: Vec<_> = self.0.rounds
            .iter()
            .enumerate()
            .map(|(i, round)| json_round(i + 1, round))
            .collect();
        state.serialize_field("rounds", &rounds_json)?;
        state.end()
    }
}
