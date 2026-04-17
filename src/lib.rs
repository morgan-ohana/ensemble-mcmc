//! An affine-invariant ensemble MCMC sampler based on the stretch move algorithm
//! of [Goodman & Weare (2010)](https://msp.org/camcos/2010/5-1/p04.xhtml).
//!
//! The sampler maintains an ensemble of walkers that explore the parameter space
//! in parallel. Each walker proposes a move by stretching along the direction to
//! a randomly chosen partner walker, making the algorithm invariant to affine
//! transformations of the parameter space and requiring no tuning of step sizes.
//!
//! The affine nature of the stretch move algorithm makes it ideal for posteriors with complex or elongated degeneracies.
//!
//! # Quick start
//! ```rust
//! use ensemble_mcmc::{MCMCCore, MCMCSettings, mcmc};
//!
//! // Fit the mean and standard deviation of a Gaussian to some observed data.
//! struct GaussianFit {
//!     data: Vec<f64>,
//! }
//!
//! impl MCMCCore for GaussianFit {
//!     fn get_bounds(&self) -> &[[f64; 2]] {
//!         &[[-10.0, 10.0], [0.01, 5.0]]  // [mu, sigma]
//!     }
//!
//!     fn get_log_likelihood(&self, params: &[f64]) -> f64 {
//!         let (mu, sigma) = (params[0], params[1]);
//!         let n = self.data.len() as f64;
//!         -n * sigma.ln()
//!             - self.data.iter().map(|x| (x - mu).powi(2)).sum::<f64>()
//!               / (2.0 * sigma.powi(2))
//!     }
//! }
//!
//! let model = GaussianFit { data: vec![1.2, 0.8, 1.5, 0.9, 1.1] };
//! let output = mcmc(&model, MCMCSettings::default());
//! // best_params[0] ≈ 1.1 (sample mean), best_params[1] ≈ 0.25 (sample std)
//! println!("Best params: {:?}", output.best_params);
//! ```

use log;
use rand::{Rng, RngExt, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize, deserialize, rancor::Error};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

/// Internal serialization-friendly representation of [`MCMCOutput`].
///
/// `Vec<Vec<f64>>` causes rkyv offset overflow for large chains, so we flatten
/// the chain to a single `Vec<f64>` and store `n_params` for reconstruction.
/// This type is not part of the public API.
#[derive(Archive, Deserialize, Serialize, serde::Serialize, serde::Deserialize)]
struct FlattenedMCMCOutput {
    best_params: Vec<f64>,
    n_params: usize,
    flattened_chain: Vec<f64>,
    log_likelihoods: Vec<f64>,
    gelman_rubin: Vec<f64>,
}

impl FlattenedMCMCOutput {
    fn init(output: &MCMCOutput) -> Self {
        let output = output.clone();
        FlattenedMCMCOutput {
            best_params: output.best_params,
            n_params: output.chain[0].len(),
            flattened_chain: output.chain.into_iter().flatten().collect(),
            log_likelihoods: output.log_likelihoods,
            gelman_rubin: output.gelman_rubin,
        }
    }
    fn unpack(self) -> MCMCOutput {
        MCMCOutput {
            best_params: self.best_params,
            chain: self
                .flattened_chain
                .chunks(self.n_params)
                .map(|c| c.to_vec())
                .collect(),
            log_likelihoods: self.log_likelihoods,
            gelman_rubin: self.gelman_rubin,
        }
    }
}

/// The output of a completed MCMC run.
///
/// Contains the full sample chain, log-likelihoods, the highest-likelihood
/// parameter set found, and Gelman-Rubin convergence diagnostics.
#[derive(Clone, Debug)]
pub struct MCMCOutput {
    /// The parameter vector with the highest log-likelihood across all walkers
    /// and steps.
    pub best_params: Vec<f64>,
    /// All post-burn-in samples from all walkers, concatenated. Each entry is
    /// one sample, i.e. a vector of length `n_params`. The total length is
    /// `num_walkers * num_steps`.
    pub chain: Vec<Vec<f64>>,
    /// Log-likelihood values corresponding to each sample in `chain`.
    pub log_likelihoods: Vec<f64>,
    /// Gelman-Rubin R-hat convergence statistic, one value per parameter.
    /// Values close to 1.0 indicate good convergence; values above ~1.1
    /// suggest the chains have not mixed well and more steps are needed.
    pub gelman_rubin: Vec<f64>,
}

impl MCMCOutput {
    /// Save the output to a binary file using rkyv serialization.
    ///
    /// The file can be reloaded with [`MCMCOutput::load`]. Binary files are
    /// more compact and faster to read/write than JSON for large chains.
    ///
    /// # Errors
    /// Returns an error if the file cannot be created or serialization fails.
    pub fn save(&self, file_name: &str) -> anyhow::Result<()> {
        let flattened_output = FlattenedMCMCOutput::init(self);

        let file = File::create(file_name)?;
        let mut writer = BufWriter::new(file);

        let bytes = rkyv::to_bytes::<Error>(&flattened_output)
            .map_err(|e| anyhow::anyhow!("Serialization failed: {}", e))?;

        writer.write_all(&bytes)?;
        log::info!("Binary file saved as {file_name}");

        Ok(())
    }

    /// Save the output to a human-readable JSON file.
    ///
    /// Useful for inspecting results or loading them from other languages.
    /// For large chains, prefer [`MCMCOutput::save`] which produces smaller files.
    ///
    /// # Errors
    /// Returns an error if the file cannot be created or serialization fails.
    pub fn save_as_json(&self, file_name: &str) -> anyhow::Result<()> {
        let flattened_output = FlattenedMCMCOutput::init(self);

        let file = File::create(file_name)?;
        let writer = BufWriter::new(file);

        // Serialize to JSON
        serde_json::to_writer_pretty(writer, &flattened_output)
            .map_err(|e| anyhow::anyhow!("JSON serialization failed: {}", e))?;

        log::info!("JSON File saved as {file_name}");

        Ok(())
    }

    /// Load a previously saved output from a binary file.
    ///
    /// The file must have been created with [`MCMCOutput::save`].
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened, is corrupt, or was not
    /// produced by this library.
    pub fn load(file_name: &str) -> anyhow::Result<MCMCOutput> {
        let file = File::open(file_name)?;
        let mut reader = BufReader::new(file);

        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;

        let archived = rkyv::access::<ArchivedFlattenedMCMCOutput, Error>(&buffer)
            .map_err(|e| anyhow::anyhow!("Data verification failed: {}", e))?;

        let flattened_output: FlattenedMCMCOutput =
            deserialize::<FlattenedMCMCOutput, Error>(archived)?;
        log::info!("Binary file loaded from {file_name}");

        Ok(flattened_output.unpack())
    }

    /// Load a previously saved output from a JSON file.
    ///
    /// The file must have been created with [`MCMCOutput::save_as_json`].
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened, is not valid JSON, or
    /// does not match the expected format.
    pub fn load_from_json(file_name: &str) -> anyhow::Result<MCMCOutput> {
        let file = File::open(file_name)?;
        let reader = BufReader::new(file);

        let flattened_output: FlattenedMCMCOutput = serde_json::from_reader(reader)
            .map_err(|e| anyhow::anyhow!("JSON deserialization failed: {}", e))?;

        log::info!("JSON file loaded from {file_name}");

        Ok(flattened_output.unpack())
    }
}

/// The interface your model must implement to be sampled with [`mcmc`].
///
/// # Example
///
/// ```rust
/// use ensemble_mcmc::{MCMCCore, MCMCSettings, mcmc};
///
/// // Fit the mean and standard deviation of a Gaussian to some observed data.
/// struct GaussianFit {
///     data: Vec<f64>,
/// }
///
/// impl MCMCCore for GaussianFit {
///     fn get_bounds(&self) -> &[[f64; 2]] {
///         &[[-10.0, 10.0], [0.01, 5.0]]  // [mu, sigma]
///     }
///
///     fn get_log_likelihood(&self, params: &[f64]) -> f64 {
///         let (mu, sigma) = (params[0], params[1]);
///         let n = self.data.len() as f64;
///         -n * sigma.ln()
///             - self.data.iter().map(|x| (x - mu).powi(2)).sum::<f64>()
///               / (2.0 * sigma.powi(2))
///     }
/// }
///
/// let model = GaussianFit { data: vec![1.2, 0.8, 1.5, 0.9, 1.1] };
/// let output = mcmc(&model, MCMCSettings::default());
/// // best_params[0] ≈ 1.1 (sample mean), best_params[1] ≈ 0.25 (sample std)
/// println!("Best params: {:?}", output.best_params);
/// ```
pub trait MCMCCore: Sync {
    /// Returns the hard prior bounds for each parameter as `[min, max]` pairs.
    ///
    /// Proposals that fall outside these bounds are always rejected. Walkers
    /// are also initialised uniformly within these bounds, so they should
    /// encompass the region where significant posterior probability mass lives.
    /// The length of the returned slice determines `n_params`.
    fn get_bounds(&self) -> &[[f64; 2]];

    /// Returns the natural logarithm of the likelihood for the given parameter
    /// vector.
    ///
    /// `params` is guaranteed to be within the bounds returned by
    /// [`get_bounds`](MCMCCore::get_bounds). The function does not need to be
    /// normalised — only differences in log-likelihood matter for acceptance.
    ///
    /// This function will be called from multiple threads simultaneously, so
    /// any shared state must be either immutable or protected by a lock.
    fn get_log_likelihood(&self, params: &[f64]) -> f64;
}

#[derive(Clone)]
struct Walker {
    params: Vec<f64>,
    log_prob: f64,
    rng: Pcg64,
}

/// Samples the stretch factor `z` from the distribution pdf ~ 1/sqrt(z) on [1/a, a].
///
/// Uses inverse CDF sampling. The derivation is:
/// - cdf = (sqrt(z) - 1/sqrt(a)) / (sqrt(a) - 1/sqrt(a))
/// - Solving for z: z = (1/sqrt(a) + u * (sqrt(a) - 1/sqrt(a)))^2
fn sample_z(rng: &mut impl Rng, a: f64) -> f64 {
    // pdf ~ 1/sqrt(z) [1/a, a]
    // cdf ~ 2sqrt(z) + C [1/a, a]
    // Normalizing on the range it must be: cdf = (sqrt(z) - 1/sqrt(a)) / (sqrt(a) - 1/sqrt(a))
    // u is random sample from cdf, so u * (sqrt(a) - 1/sqrt(a)) + 1/sqrt(a) = sqrt(z)
    // => z = (1/sqrt(a) + u * (sqrt(a) - 1/sqrt(a)))^2
    let u: f64 = rng.random();
    let b = 1.0 / a.sqrt();
    (b + u * (a.sqrt() - b)).powi(2)
}

/// Performs one parallel stretch-move step across the entire walker ensemble.
///
/// Each walker independently proposes a move by stretching toward a randomly
/// chosen partner. The snapshot `walkers_old` ensures all walkers read from
/// the same state regardless of parallel update order.
fn stretch_move_parallel(walkers: &mut [Walker], core: &impl MCMCCore, a: f64) {
    let n = walkers.len();
    let bounds = core.get_bounds();
    let d = bounds.len();

    // Copy walkers so reads are consistent during parallel update
    let walkers_old = walkers.to_vec();

    walkers.par_iter_mut().enumerate().for_each(|(i, walker)| {
        // choose partner
        let mut j = walker.rng.random_range(0..n);
        while j == i {
            j = walker.rng.random_range(0..n);
        }

        let z = sample_z(&mut walker.rng, a);
        let proposal: Vec<f64> = walker
            .params
            .iter()
            .zip(&walkers_old[j].params)
            .map(|(&xi, &xj)| xj + z * (xi - xj))
            .collect();

        let in_bounds = proposal
            .iter()
            .zip(bounds)
            .all(|(&p, b)| p >= b[0] && p <= b[1]);

        if !in_bounds {
            // if out of bounds just stop so walker stays put ie rejecting move
            return;
        }

        let proposed_log_prob = core.get_log_likelihood(&proposal);

        let log_accept = (d as f64 - 1.0) * z.ln() + proposed_log_prob - walker.log_prob;

        if log_accept >= 0.0 || walker.rng.random::<f64>() < log_accept.exp() {
            walker.params = proposal;
            walker.log_prob = proposed_log_prob;
        }
    });
}

/// Configuration for an MCMC run.
///
/// All fields have sensible defaults via [`MCMCSettings::default`]. Use struct
/// update syntax to override only the fields you care about:
///
/// ```rust
/// use ensemble_mcmc::MCMCSettings;
///
/// let settings = MCMCSettings {
///     num_steps: 50_000,
///     ..MCMCSettings::default()
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MCMCSettings {
    /// Number of steps to record after burn-in. The total number of
    /// likelihood evaluations is `(num_steps + burn_in) * num_walkers`.
    /// Default: 10000.
    pub num_steps: usize,
    /// Number of initial steps to discard while the ensemble moves away from
    /// its starting position. Default: 1000.
    pub burn_in: usize,
    /// Number of walkers in the ensemble. Must be at least 3; values below
    /// `2 * n_params` will produce a warning. Default: 32.
    pub num_walkers: usize,
    /// The stretch scale factor `a` controlling the range of the proposal
    /// distribution. Must be greater than 1. Larger values mean bigger
    /// proposed steps. The standard value from Goodman & Weare is 2.0.
    /// Default: 2.0.
    pub scale_factor: f64,
}

impl Default for MCMCSettings {
    fn default() -> Self {
        MCMCSettings {
            num_steps: 10000,
            burn_in: 1000,
            num_walkers: 32,
            scale_factor: 2.0,
        }
    }
}

/// Run the affine-invariant ensemble MCMC sampler and return the results.
///
/// Walkers are initialised by sampling uniformly within the bounds returned
/// by [`MCMCCore::get_bounds`]. After burn-in, all walker positions are
/// recorded and returned in [`MCMCOutput::chain`].
///
/// Convergence is assessed automatically using the split Gelman-Rubin R-hat
/// statistic. The chain for each walker is split in half, giving `2 *
/// num_walkers` sub-chains that are compared. R-hat values close to 1.0
/// indicate good convergence.
///
/// Progress is logged via the log crate if a logger has been configured.
///
/// # Panics
/// Panics if `settings.num_walkers < 3` or `settings.scale_factor <= 1.0`.
///
/// # Warnings
/// Prints a warning to stderr if `num_walkers < 2 * n_params`, as the
/// ensemble may fail to explore all directions in parameter space.
pub fn mcmc(core: &impl MCMCCore, settings: MCMCSettings) -> MCMCOutput {
    if settings.num_walkers < 3 {
        panic!(
            "With less than 3 walkers the ensemble stepper cannot properly function. At least twice your number of parameters is recommended to avoid convergence issues"
        )
    } else if settings.num_walkers < 2 * core.get_bounds().len() {
        log::warn!(
            "To properly explore the whole parameter space your walker number ({}) should be at least twice the dimension of your parameter space ({}). You may experience convergence issues.",
            settings.num_walkers,
            core.get_bounds().len()
        )
    }

    if settings.scale_factor <= 1.0 {
        panic!(
            "Scale factor must be greater than 1 for the stretch step probabilily distribution to be well-posed. Your scale factor is {}",
            settings.scale_factor
        )
    }

    let mut rng = Pcg64::seed_from_u64(42);

    // initialize walkers
    let mut walkers: Vec<Walker> = (0..settings.num_walkers)
        .map(|i| {
            let params: Vec<f64> = core
                .get_bounds()
                .iter()
                .map(|b| b[0] + rng.random::<f64>() * (b[1] - b[0]))
                .collect();
            let log_prob = core.get_log_likelihood(&params);

            Walker {
                params,
                log_prob,
                rng: Pcg64::seed_from_u64(42 + i as u64 * 7919),
            }
        })
        .collect();

    let mut chains: Vec<Vec<Vec<f64>>> =
        vec![Vec::with_capacity(settings.num_steps); settings.num_walkers];
    let mut log_likelihoods: Vec<Vec<f64>> =
        vec![Vec::with_capacity(settings.num_steps); settings.num_walkers];

    for step in 0..(settings.num_steps + settings.burn_in) {
        stretch_move_parallel(&mut walkers, core, settings.scale_factor);

        if step >= settings.burn_in {
            for w in 0..settings.num_walkers {
                chains[w].push(walkers[w].params.clone());
                log_likelihoods[w].push(walkers[w].log_prob);
            }
        }

        if step % 1000 == 0 {
            if step < settings.burn_in {
                log::info!("Burn in step {step}")
            } else {
                log::info!("Step {}", step - settings.burn_in);
            }
        }
    }

    // Split each walker's chain in half, giving 2*n_walkers chains
    let half = settings.num_steps / 2;
    let split: Vec<Vec<Vec<f64>>> = chains
        .iter()
        .flat_map(|walker_chain| {
            // truncate to even length so both halves are equal
            let trimmed = &walker_chain[..half * 2];
            vec![trimmed[..half].to_vec(), trimmed[half..].to_vec()]
        })
        .collect();
    let gelman_rubin = calculate_gelman_rubin(&split);

    let combined_chain: Vec<Vec<f64>> = chains.into_iter().flatten().collect();
    let combined_log_likelihoods: Vec<f64> = log_likelihoods.into_iter().flatten().collect();

    // Best params
    let best_idx = combined_log_likelihoods
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let best_params = combined_chain[best_idx].clone();

    MCMCOutput {
        best_params,
        chain: combined_chain,
        log_likelihoods: combined_log_likelihoods,
        gelman_rubin,
    }
}

/// Compute the Gelman-Rubin R-hat convergence statistic for a set of chains.
///
/// R-hat compares the within-chain variance to the between-chain variance. A
/// value of 1.0 means the chains are indistinguishable from each other;
/// values above ~1.1 indicate poor mixing and the need for more samples.
///
/// All chains must have the same length and the same number of parameters.
///
/// Returns one R-hat value per parameter. If a chain is completely stuck
/// (zero within-chain variance), the corresponding R-hat is `f64::INFINITY`.
pub fn calculate_gelman_rubin(chains: &[Vec<Vec<f64>>]) -> Vec<f64> {
    let m = chains.len() as f64; // number of chains
    let n = chains[0].len() as f64; // Samples per chain
    let n_params = chains[0][0].len();

    let mut r_hat = vec![0.0; n_params];

    for param_idx in 0..n_params {
        // Collect samples for this parameter
        let param_samples: Vec<Vec<f64>> = chains
            .iter()
            .map(|chain| chain.iter().map(|p| p[param_idx]).collect())
            .collect();

        // Calculate within-chain variance
        let mut chain_means = Vec::new();
        let mut within_var = 0.0;

        for samples in &param_samples {
            let mean: f64 = samples.iter().sum::<f64>() / n;
            chain_means.push(mean);

            let variance: f64 =
                samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            within_var += variance;
        }
        within_var /= m;

        // Calculate between-chain variance
        let overall_mean: f64 = chain_means.iter().sum::<f64>() / m;

        let between_var: f64 = chain_means
            .iter()
            .map(|&mean| (mean - overall_mean).powi(2))
            .sum::<f64>()
            / (m - 1.0);

        // Calculate pooled variance
        let pooled_var = ((n - 1.0) / n) * within_var + between_var;

        // R-hat statistic (should approach 1.0 as chains converge)
        r_hat[param_idx] = if within_var > 0.0 {
            (pooled_var / within_var).sqrt()
        } else {
            f64::INFINITY
        };
    }

    r_hat
}
