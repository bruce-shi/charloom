//! Native execution uses Rayon; a browser Worker owns a single WASM instance.
//! This avoids creating OS threads (or requiring cross-origin shared memory)
//! while preserving the exact iterator order and numerical implementation.
#[cfg(not(target_arch = "wasm32"))]
pub use rayon::prelude::*;

#[cfg(target_arch = "wasm32")]
pub trait SequentialSlice<T> {
    fn par_iter(&self) -> std::slice::Iter<'_, T>;
}
#[cfg(target_arch = "wasm32")]
impl<T> SequentialSlice<T> for [T] {
    fn par_iter(&self) -> std::slice::Iter<'_, T> {
        self.iter()
    }
}
#[cfg(target_arch = "wasm32")]
pub trait SequentialIntoIterator: IntoIterator + Sized {
    fn into_par_iter(self) -> Self::IntoIter {
        self.into_iter()
    }
}
#[cfg(target_arch = "wasm32")]
impl<T: IntoIterator> SequentialIntoIterator for T {}

pub struct Timer {
    #[cfg(not(target_arch = "wasm32"))]
    start: std::time::Instant,
}
impl Timer {
    pub fn start() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            start: std::time::Instant::now(),
        }
    }
    pub fn seconds(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start.elapsed().as_secs_f64()
        }
        // The Worker fills in performance.now() timing. std::time panics on wasm32.
        #[cfg(target_arch = "wasm32")]
        {
            0.
        }
    }
}
