An affine-invariant ensemble MCMC sampler based on the stretch move algorithm
of [Goodman & Weare (2010)](https://msp.org/camcos/2010/5-1/p04.xhtml).

The sampler maintains an ensemble of walkers that explore the parameter space
in parallel. Each walker proposes a move by stretching along the direction to
a randomly chosen partner walker, making the algorithm invariant to affine
transformations of the parameter space and requiring no tuning of step sizes.

The affine nature of the stretch move algorithm makes it ideal for posteriors with complex or elongated degeneracies.

# Quick start
```rust
use ensemble_mcmc::{MCMCCore, MCMCSettings, mcmc};

// Fit the mean and standard deviation of a Gaussian to some observed data.
struct GaussianFit {
    data: Vec<f64>,
}

impl MCMCCore for GaussianFit {
    fn get_bounds(&self) -> &[[f64; 2]] {
        &[[-10.0, 10.0], [0.01, 5.0]]  // [mu, sigma]
    }

    fn get_log_likelihood(&self, params: &[f64]) -> f64 {
        let (mu, sigma) = (params[0], params[1]);
        let n = self.data.len() as f64;
        -n * sigma.ln()
           - self.data.iter().map(|x| (x - mu).powi(2)).sum::<f64>()
              / (2.0 * sigma.powi(2))
    }
}

let model = GaussianFit { data: vec![1.2, 0.8, 1.5, 0.9, 1.1] };
let output = mcmc(&model, MCMCSettings::default());
// best_params[0] ≈ 1.1 (sample mean), best_params[1] ≈ 0.25 (sample std)
println!("Best params: {:?}", output.best_params);
```
