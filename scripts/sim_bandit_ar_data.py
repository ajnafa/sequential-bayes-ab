from numpy.random import binomial, normal, multivariate_normal, poisson
from numpy import array, zeros, ones, exp, repeat
from polars import DataFrame, col, Int64, Float64, Series, when, from_pandas
from cmdstanpy import CmdStanModel, install_cmdstan, cmdstan_path, set_cmdstan_path
import plotnine as gg

##############################################################################
########################### Helper Functions #################################
##############################################################################

# Define a function for the inverse logit transformation
def inv_logit(x: float):
    pi = exp(x) / (1 + exp(x))
    return pi

# Define a function for the Bernoulli distribution
def bernoulli(n: int, p: float):
    return binomial(1, p, n)

##############################################################################
################################ Stan Model ##################################
##############################################################################

# Compile the Stan Model
install_cmdstan(compiler=True, cores=4, progress=True)
model = CmdStanModel(stan_file='models/binomial_bandit_mstep.stan')

##############################################################################
################## Simulation of Data for Fixed Utility ######################
##############################################################################

# Define the dimensions of the experiment
n: Int64 = 5000     # Average number of new players per day
T: Int64 = 90       # Maximum Number of days to run the experiment for
het = False         # Whether to include temporal heterogeneity

# Define the treatment effect and baseline conversion rate
tau: Float64 = 0.8       # Treatment Effect
alpha: Float64 = -5.0    # Baseline Conversion Rate
prob: Float64 = 0.5      # Allocation Probability 
truth = zeros(T)         # True effect for each period

# Define the data frame to store the results
data = DataFrame(
    {
        'period': 0, 
        'treat': 0, 
        'purchase': 0,
        'players': 0
    },
    schema={
        "period": Int64, 
        "treat": Int64, 
        "purchase": Int64,
        "players": Int64
    }
)

# Define the data frame to store the results
track_data = DataFrame(
    {
        'next_period': 0, 
        'arm': 0,
        'policy': '', 
        'allocation_probs': 0.0,
        'cum_purchases': 0,
        'cum_players': 0,
        'current_period': 0
    },
    schema={
        "next_period": Int64, 
        "arm": Int64,
        "policy": str, 
        "allocation_probs": Float64,
        "cum_purchases": Int64,
        "cum_players": Int64,
        "current_period": Int64
    }
)

for t in range(1, T):
    
    # Simulate the number of new players
    new_players = poisson(n)

    # Simulate the allocation of new players to each arm
    X = bernoulli(n=new_players, p=prob)

    # Treatment Assignment Mechanism
    muA = alpha + 0*tau; sdA = normal(0, 0.5)  # Mean and Std. Dev for the Control
    muB = alpha + 1*tau; sdB = normal(0, 1)  # Mean and Std. Dev for Arm 1
        
    # Simulate the potential outcomes for each arm
    mu = [muA, muB]

    rho = 0.00      # Correlation between the potential outcomes

    # Define the covariance matrix
    cov = [
        [sdA**2, rho*sdA*sdB],
        [rho*sdA*sdB, sdB**2]
    ]

    # Simulate the Potential Outcomes for each arm
    mu = multivariate_normal(mu, cov, new_players)

    # Simulate the observed outcomes for each arm
    YA = mu[:, 0]        # Potential Outcome if A = 1, A' = 0
    YB = mu[:, 1]        # Potential Outcome if A = 0, B = 1
    
    # Realization of the Potential Outcomes on the Latent Scale
    Y_obs = YA * (1 - X) + YB * X
    Y_mis = YA * X + YB * (1 - X)
    
    # Store the true effect for each period
    truth[t] = (inv_logit(YB) - inv_logit(YA)).mean()
            
    # Simulate the observed outcomes at time t
    Y = bernoulli(new_players, p = inv_logit(Y_obs))
    
    # Store the results in a data frame
    period: Int64 = repeat(t, len(Y))    
    data_t = DataFrame([
        Series('period', period, dtype = Int64),
        Series('treat', X, dtype = Int64),
        Series('purchase', Y, dtype = Int64)
    ])
    
    # Aggregate the data counts and trials
    data_t = (
        data_t
        .groupby(['period', 'treat'], maintain_order=True)
        .agg(
            [
                col('purchase').sum().alias('purchase').cast(Int64), 
                col('treat').count().alias('players').cast(Int64)
            ]
        )
    )
    
    # Append the data to the main data frame
    data = data.vstack(data_t, in_place=True)

    
    if t > 10: 
        
        # Current Data up to time t
        current_data = (
            data
            .with_columns(arm_id = col('treat') + 1)
            .filter(col('period') > 0)
        )

        # Data to Pass to Stan
        stan_data = {
            'N': current_data.shape[0],
            'T': current_data['period'].max(),
            'K': 2,
            'T_pred': 7,
            'trials': current_data['players'].to_numpy(),
            'successes': current_data['purchase'].to_numpy(),
            'kk': current_data['arm_id'].to_numpy(),
            'tt': current_data['period'].to_numpy()
        }
        
        # Fit the model
        fit = model.sample(data=stan_data, parallel_chains=4)
        
        # Extract the posterior probabilities for the best arm at the next period
        prob_best = fit.draws_pd(vars = "prob_best").mean(axis=0)
        prob_indicies = prob_best.index.to_series()
        prob_best = (
            DataFrame({
                'prob_best': from_pandas(prob_best),
                'prob_indicies': from_pandas(prob_indicies)
            })
            # Split the indicies into arms and periods
            .with_columns(prob_indicies = col('prob_indicies').str.extract_all(r'(\d+)'))
            # Extract the arm and period identifiers
            .with_columns(
                arm = col('prob_indicies').apply(lambda x: x[1]).cast(Int64),
                period = col('prob_indicies').apply(lambda x: x[0]).cast(Int64)
            )
            .drop('prob_indicies')
            # Convert the arm identifiers to strings
            .with_columns(
                policy = when(col('arm') == 1)
                .then('Control')
                .otherwise('Treat')
            )
            # Keep only the forecasts for the next period
            .filter(col('period') > t)
            # Group by the arm and policy
            .groupby(by = ['arm', 'policy'])
            # Take the mean of the forecasted probabilities for each arm
            .agg(col('prob_best').mean().alias('allocation_probs'))
        )
        
        # Update the allocation probability
        prob = prob_best.filter(col('policy') == 'Treat')['allocation_probs'].to_numpy()
        
        # Store the results in a data frame
        prob_best.insert_at_idx(0, Series('next_period', repeat(t + 1, prob_best.shape[0]), dtype = Int64))
        
        # Extract the data from the current and previous periods
        current_data = (
            current_data
            .groupby(['arm_id'], maintain_order=True)
            .agg(
                [
                    col('purchase').sum().alias('cum_purchases'),
                    col('players').sum().alias('cum_players'),
                    col('period').max().alias('current_period')
                ]
            )
        )
        
        # Join the data to the posterior probabilities
        prob_best = prob_best.join(current_data, left_on = 'arm', right_on = 'arm_id')
        
        # Append the data to the main data frame
        track_data.vstack(prob_best, in_place=True)
        
# Filter out the empty period
track_data = track_data.filter(col('next_period') > 0)
        
##############################################################################
########################## Posterior Analysis ################################
##############################################################################

# Plot of the Posterior Allocation Probabilities
(
    gg.ggplot(track_data.to_pandas())
    + gg.aes(x='current_period') 
    + gg.facet_wrap('~policy')
    + gg.geom_point(gg.aes(y='allocation_probs', color='policy'))
    + gg.theme_bw(base_size=14, base_family='serif')
    + gg.labs(
        x='Period', 
        y='Posterior Probability of the Optimal Policy',
        subtitle="Dots are the average of the posterior probabilities of each action being optimal based on m-step prediction at each period."
    )
)
        
# Plot of performance of the optimal policy in the final model
delta_pred = fit.draws_pd(vars = 'delta_theta_pred').median(axis=0)
delta_pred_upper = fit.draws_pd(vars = 'delta_theta_pred').quantile(q=0.84, axis=0)
delta_pred_lower = fit.draws_pd(vars = 'delta_theta_pred').quantile(q=0.16, axis=0)
indicies = delta_pred.index.to_series()

preds = {
   'delta_pred': from_pandas(delta_pred),
   'delta_pred_upper': from_pandas(delta_pred_upper),
   'delta_pred_lower': from_pandas(delta_pred_lower),
   'indicies': from_pandas(indicies)
}

# Create data frame and split names into arms and periods
preds_df = (
    DataFrame(preds)
    .with_columns(
        period = col('indicies').str.extract(r'(\d+)').cast(Int64)
    )
)

# Plot of the Posterior Allocation Probabilities
(
    gg.ggplot(preds_df.to_pandas())
    + gg.aes(x='period') 
    + gg.geom_pointrange(gg.aes(y='delta_pred', ymin='delta_pred_lower', ymax='delta_pred_upper'))
    + gg.scale_x_continuous(breaks=[1, 15, 30, 45, 60, 75, 90, 105, 120])
    + gg.scale_y_continuous(
        breaks=[-0.04, -0.02,  0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
        limits=[-0.04, 0.12]
    )
    + gg.labs(
        x='Period', 
        y='Difference in Conversion Probability',
        title = 'Posterior Predictive Forecasts of the Difference in Daily Conversion Probability',
        subtitle="Red dashed line indicates start of out-of-sample forecasts, blue line is the true long-run average effect."
    )
    + gg.geom_vline(xintercept=90, linetype='dashed', color='red')
    + gg.geom_hline(yintercept=truth.mean(), linetype='dashed', color='blue')
    + gg.theme_bw(base_size=14, base_family='serif')
    + gg.theme(plot_caption = gg.element_text(size=10, family='serif', va='top'))
)



