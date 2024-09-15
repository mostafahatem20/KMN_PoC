use std::collections::HashSet;

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
