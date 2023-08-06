/* 
    Model: Binomial Bandit for Sequential Bayesian A/B Testing 
    Author: A. Jordan Nafa
    Date: 2023-08-07
*/
data {
   int<lower=1> N;                          // Number of Arms
   int<lower=1> T;                          // Number of Prior Periods

   // Data for the Model
   array[N] int<lower=1> trials;            // Number of Trials
   array[N] int<lower=0> successes;         // Number of Successes

   // Mapping of Arms to Observations
   array[N] int<lower=1,upper=N> arm;       // Arm for each observation
   
   // Mapping of Arms to Prior Parameters
   vector<lower=1>[N] prior_alpha;          // Prior Successes for each arm
   vector<lower=1>[N] prior_beta;           // Prior Failures for each arm
}

transformed data {
   // Number of Failures
   array[N] int<lower=0> failures;
   for (n in 1:N) {
      failures[n] = trials[n] - successes[n];
   }

   // Data for the Conversion Rate
   vector<lower=1>[N] beta;
   vector<lower=1>[N] alpha;
   for (n in 1:N) {
      beta[n] = prior_beta[n] + failures[n];
      alpha[n] = prior_alpha[n] + successes[n];
   }
}

parameters {
    // Parameters for the Model
    vector<lower=0, upper=1>[N] theta;       // Probability of Success for Each Arm
}

model {
    // Likelihood for the Model
    for (n in 1:N) {
        // Sample the Conversion Rate
        theta ~ beta(alpha[n], beta[n]);

        // Sample the Number of Successes
        successes[n] ~ binomial(trials[n], theta[n]);
    }
}

generated quantities {
    // Draw from the Posterior Predictive Distribution
    array[N] real alpha_ppd = binomial_rng(trials, theta);
    array[N] real beta_ppd;
    for (n in 1:N) {
        beta_ppd[n] = trials[n] - alpha_ppd[n];
    }

    // Calculate the Posterior Predictive Conversion Rate
    array[N] real theta_ppd = beta_rng(alpha_ppd, beta_ppd);

    // Calculate the Posterior Probability of Being the Best Policy
    vector[N] post_prob;
    {
        real best_prob = max(theta_ppd);
        for (n in 1:N) {
            post_prob[n] = (theta_ppd[n] >= best_prob);
        }
        // Uniform in the case of ties
        post_prob = post_prob / sum(post_prob);
    }
}





