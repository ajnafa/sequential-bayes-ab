/* 
    Model: Binomial Bandit for Sequential Bayesian A/B Testing 
    with an Autoregressive Formulation and Logit Link Function
    Author: A. Jordan Nafa
    License: MIT
    Date: 2023-08-07
*/

data {
   int<lower=1> N;                          // Number of Observations
   int<lower=1> K;                          // Number of Arms
   int<lower=1> T;                          // Number of Observed Periods
   int<lower=1> T_pred;                     // Number of Predicted Periods

   // Data for the Model
   array[N] int<lower=1> trials;            // Number of Trials
   array[N] int<lower=0> successes;         // Number of Successes

   // Mapping of Arms to Observations and Periods
   array[N] int<lower=1,upper=K> kk;        // Arm for each observation
   array[N] int<lower=1,upper=T> tt;        // Period for each observation
}

transformed data {
   // Total Number of Periods
   int<lower=T + 1> M = T + T_pred;
}

parameters {
   // Model Parameters
   array[K] vector[T] mu;                   // Mean of the Latent Distribution
   vector<lower=0>[K] sigma;                // Scale of the Latent Distribution
}

transformed parameters {
   array[K] vector[T] theta;                // Probability of Success
   for (k in 1:K) {
      theta[k, 1:T] = inv_logit(mu[k, 1:T]);
   }
}

model {
    // Priors for the Model Parameters
    for (k in 1:K) {
        sigma[k] ~ lognormal(0, 1);
        mu[k, 1] ~ normal(0, sigma[k]);
        for (t in 2:T) {
            mu[k, t] ~ normal(mu[k, t-1], sigma[k]);
        }
    }

    // Likelihood for the Data
    for (n in 1:N) {
        successes[n] ~ binomial(trials[n], theta[kk[n], tt[n]]);
    }
}

generated quantities {
    // Posterior Predictive Distribution
    array[N] real<lower=0> y_rep;
    for (n in 1:N) {
        y_rep[n] = binomial_rng(trials[n], theta[kk[n], tt[n]]);
    }

    // Posterior Predictive Distribution for the Full Series
    array[K, M] real mu_pred;
    array[M * K] real mu_pred_vec;
    for (k in 1:K) {
        mu_pred[k, 1:T] = normal_rng(mu[k, 1:T], sigma[k]);
        for (t in (T+1):M) {
            mu_pred[k, t] = normal_rng(mean(mu_pred[k, (t-3):(t-1)]), sigma[k]);
        }
        mu_pred_vec[((k-1)*M + 1):(k*M)] = mu_pred[k, 1:M];
    }

    // Posterior Predictive Distribution for the Probability of Success
    array[K*M] real theta_pred_vec = inv_logit(mu_pred_vec);
    array[K, M] real theta_pred;
    for (k in 1:K) {
        theta_pred[k, 1:M] = theta_pred_vec[((k-1)*M + 1):(k*M)];
    }

    // Difference in the Probability of Success
    array[M] real delta_theta_pred;
    for (t in 1:M) {
        delta_theta_pred[t] = theta_pred[2, t] - theta_pred[1, t];
    }

    // Probability of Best at Each Period
    array[M] simplex[K] prob_best;
    {
        for (t in 1:M) {
            real best_arm = max(theta_pred[1:K, t]);
            for (k in 1:K) {
                prob_best[t, k] = (theta_pred[k, t] >= best_arm);
            }
            prob_best[t, 1:K] = prob_best[t, 1:K] / sum(prob_best[t, 1:K]);
        }
    }
}
