from numpy.random import binomial, normal, multivariate_normal, poisson
from numpy import array, zeros, ones, exp, repeat
from polars import DataFrame, col, Int64, Float64, Series, when
from cmdstanpy import CmdStanModel, install_cmdstan
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
# install_cmdstan(compiler=True, cores=4, progress=True)
model = CmdStanModel(stan_file='models/binomial_bandit.stan')

##############################################################################
################## Simulation of Data for Fixed Utility ######################
##############################################################################

# Define the dimensions of the experiment
n: Int64 = 2500     # Average number of new players per day
T: Int64 = 60       # Maximum Number of days to run the experiment for
het = False         # Whether to include temporal heterogeneity

# Define the treatment effect and baseline conversion rate
tau: Float64 = 0.8       # Treatment Effect
alpha: Float64 = -5.0    # Baseline Conversion Rate
prob: Float64 = 0.5      # Allocation Probability 

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
        'period': 0, 
        'treat': 0, 
        'purchase': 0,
        'players': 0,
        'arm_id': 0,
        'prob_best': 0.0,
        'successes_ppd': 0.0,
        'conversion_prob': 0.0,
        'prior_alpha': 0,
        'prior_beta': 0
    },
    schema={
        "period": Int64, 
        "treat": Int64, 
        "purchase": Int64,
        "players": Int64,
        "arm_id": Int64,
        "prob_best": Float64,
        "successes_ppd": Float64,
        "conversion_prob": Float64,
        "prior_alpha": Int64,
        "prior_beta": Int64
    }
)

for t in range(1, T):
    
    # Simulate the number of new players
    new_players = poisson(n)

    # Simulate the allocation of new players to each arm
    X = bernoulli(n=new_players, p=prob)

    # Treatment Assignment Mechanism
    muA = alpha + 0*tau; sdA = normal(0, 1)  # Mean and Std. Dev for the Control
    muB = alpha + 1*tau; sdB = normal(0, 1.5)  # Mean and Std. Dev for Arm 1
        
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
    
    if t > 2: 
        # Extract the data from the current and previous periods
        current_data = (
            data
            .filter(col('period') == t)
            .with_columns(arm_id = col('treat') + 1)
        )
        
        prev_data = (
            data
            .filter(col('period') == t - 1)
            .with_columns(arm_id = col('treat') + 1)
        )
        
        # Data to Pass to Stan
        stan_data = {
            'N': current_data.shape[0],
            'T': 1,
            'trials': current_data['players'].to_numpy(),
            'successes': current_data['purchase'].to_numpy(),
            'arm': current_data['arm_id'].to_numpy(),
            'prior_alpha': prev_data['purchase'].to_numpy() + 1,
            'prior_beta': prev_data['players'].to_numpy() - prev_data['purchase'].to_numpy() + 1
        }
        
        # Fit the model
        fit = model.sample(data=stan_data, parallel_chains=4)
        
        # Extract the posterior probabilities
        prob_best = fit.draws_pd(vars = 'post_prob').mean(axis=0)
        successes = fit.draws_pd(vars = 'alpha_ppd').median(axis=0)
        conversion_prob = fit.draws_pd(vars = 'theta_ppd').median(axis=0)
        current_data = current_data.insert_at_idx(5, Series('prob_best', prob_best))
        current_data = current_data.insert_at_idx(6, Series('successes_ppd', successes))
        current_data = current_data.insert_at_idx(7, Series('conversion_prob', conversion_prob))
        current_data = current_data.insert_at_idx(8, Series('prior_alpha', stan_data['prior_alpha']))
        current_data = current_data.insert_at_idx(9, Series('prior_beta', stan_data['prior_beta']))
        track_data = track_data.vstack(current_data, in_place=False)
        
        # Update the allocation probability
        prob = current_data.filter(col('treat') == 1)['prob_best'].to_numpy()

##############################################################################
########################## Posterior Analysis ################################
##############################################################################

# Posterior Analysis
plot_data = (
    track_data
    .filter(col('period') > 1)
    .with_columns(
        arm = when(col('treat') == 0)
        .then('Control')
        .otherwise('Treat')
    )
)

# Plot of the Posterior Allocation Probabilities
(
    gg.ggplot(plot_data.to_pandas())
    + gg.aes(x='period', y='prob_best', color='arm')
    + gg.geom_line()
    + gg.geom_point()
    + gg.theme_bw() 
    + gg.scale_x_continuous(breaks=[1, 15, 30, 45, 60])
    + gg.labs(x='Period', y='Posterior Probability')
)

# Plot of the Posterior Allocation Probabilities
(
    gg.ggplot(plot_data.to_pandas())
    + gg.aes(x='period', y='conversion_prob', color='arm')
    + gg.geom_line()
    + gg.geom_point()
    + gg.theme_bw() 
    + gg.scale_x_continuous(breaks=[1, 15, 30, 45, 60])
    + gg.labs(x='Period', y='Probability of Conversion')
)