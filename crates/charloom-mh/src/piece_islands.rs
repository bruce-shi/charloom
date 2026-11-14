//! Remove unmarked garment label islands embedded in another surface.
//! Size alone is not a reason to remove a buckle, button or separate cuff:
//! require a small component with a clear, different boundary owner and no
//! direct guidance seed. Work on welded scan topology, before extraction.
use crate::geom::v;
use charloom_core::Vec3;
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub struct Input<'a> {
    pub positions: &'a [Vec3],
    pub triangles: &'a [[u32; 3]],
    pub weld_id: &'a [usize],
    pub weld_count: usize,
    pub piece: &'a [Option<usize>],
    pub skin: &'a [bool],
    pub ordinary_cloth: &'a [bool],
    /// A direct, strong polygon/stroke seed of this same piece.
    pub marked: &'a [bool],
}

pub fn refine(input: &Input) -> (Vec<(usize, Option<usize>)>, Value) {
    let n = input.triangles.len();
    let mut start = vec![0usize; input.weld_count + 1];
    for tri in input.triangles {
        for &i in tri {
            start[input.weld_id[i as usize] + 1] += 1;
        }
    }
    for i in 0..input.weld_count {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut incident = vec![0usize; start[input.weld_count]];
    for (t, tri) in input.triangles.iter().enumerate() {
        for &i in tri {
            let w = input.weld_id[i as usize];
            incident[fill[w]] = t;
            fill[w] += 1;
        }
    }
    let around = |i: u32| {
        let w = input.weld_id[i as usize];
        &incident[start[w]..start[w + 1]]
    };
    let area = |t: usize| {
        let [a, b, c] = input.triangles[t].map(|i| v(input.positions[i as usize]));
        (b - a).cross(&(c - a)).norm() * 0.5
    };
    let mut seen = vec![false; n];
    let mut components = vec![];
    let mut largest = vec![0.0f64; input.ordinary_cloth.len()];
    for seed in 0..n {
        let Some(piece) = input.piece[seed] else { continue };
        if seen[seed] || !input.ordinary_cloth[piece] {
            continue;
        }
        let mut component = vec![seed];
        seen[seed] = true;
        let mut k = 0;
        while k < component.len() {
            for &i in &input.triangles[component[k]] {
                for &t in around(i) {
                    if !seen[t] && input.piece[t] == Some(piece) {
                        seen[t] = true;
                        component.push(t);
                    }
                }
            }
            k += 1;
        }
        let area: f64 = component.iter().map(|&t| area(t)).sum();
        largest[piece] = largest[piece].max(area);
        components.push((piece, component, area));
    }
    let mut changes = vec![];
    let mut removed = vec![];
    let mut marked_kept = 0usize;
    for (piece, component, area) in &components {
        if component.iter().any(|&t| input.marked[t]) {
            marked_kept += 1;
            continue;
        }
        // Physical area, not triangle count: dense scan fragments must not
        // survive merely because they have more tessellation than a cuff.
        if *area >= largest[*piece] * 0.04 {
            continue;
        }
        let mut border = BTreeMap::<Option<usize>, usize>::new();
        for &t in component {
            for &i in &input.triangles[t] {
                for &u in around(i) {
                    if input.piece[u] != Some(*piece)
                        && (input.skin[u] || input.piece[u].is_some())
                    {
                        *border.entry(input.piece[u]).or_default() += 1;
                    }
                }
            }
        }
        let total: usize = border.values().sum();
        let Some((&owner, &votes)) = border.iter().max_by_key(|(_, count)| **count) else {
             
            continue;
        };
        if votes * 4 < total * 3 {
            continue;
        }
        changes.extend(component.iter().map(|&t| (t, owner)));
        removed.push(json!({"from_piece": piece, "to_piece": owner,
            "triangles": component.len(), "area_cm2": area * 1e4,
            "boundary_share": votes as f64 / total as f64}));
    }
    (changes, json!({"components": components.len(), "marked_components_kept": marked_kept,
        "reassigned": removed}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmarked_island_returns_to_skin_but_marked_button_and_separate_cuff_stay() {
        let positions = vec![
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            [2., 0., 0.], [2.01, 0., 0.], [2., 0.01, 0.], [2.01, 0.01, 0.],
            [3., 0., 0.], [3.01, 0., 0.], [3., 0.01, 0.],
        ];
        let triangles = vec![[0, 1, 2], [3, 4, 5], [4, 6, 5], [7, 8, 9]];
        let weld_id: Vec<_> = (0..positions.len()).collect();
        let piece = vec![Some(0), Some(0), None, Some(0)];
        let skin = vec![false, false, true, false];
        let cloth = vec![true];
        let mut marked = vec![false; 4];
        let run = |marked: &[bool]| refine(&Input { positions: &positions, triangles: &triangles,
            weld_id: &weld_id, weld_count: positions.len(), piece: &piece,
            skin: &skin, ordinary_cloth: &cloth, marked });
        assert_eq!(run(&marked).0, vec![(1, None)]);
        marked[1] = true;
        assert!(run(&marked).0.is_empty());
    }
}
