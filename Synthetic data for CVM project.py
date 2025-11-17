# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:03:06 2025

@author: dkont
"""

# generate_synthetic_data.py
# This script creates a synthetic dataset for a CVM project.
# goal here is to simulate customer behavior, transactions, and a marketing campaign
# to support RFM segmentation, clustering, and churn prediction.
# My script mimics real-world banking data complexities,for obvious reasons i did not work with actual from my banking job.

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42) # reproducible for testing

#data generation
n_customers = 3000                 # How many customers im simulating
start_date = datetime(2023, 1, 1)  # Beginning of the transaction history
end_date = datetime(2025, 10, 31)  # End of transaction history
campaign_date = datetime(2025, 10, 1) # When the tech campaign was launched
snapshot_date = datetime(2025, 11, 1) # the today for RFM calculations (day after end_date usually)
avg_tx_per_customer = 45           # Average number of transactions per customer over the period
# ----------------------------------------



# adding some kind of realistic date patterns for transactions 
# Real bank data isn't flat of course,so adding seasonality and spikes.
all_dates = pd.date_range(start_date, end_date)
transaction_date_weights = np.ones(len(all_dates))

for i, date in enumerate(all_dates):
    # Higher activity on weekends (more time to spend work money :D)
    if date.weekday() >= 5: # Saturday or Sunday
        transaction_date_weights[i] *= 1.8 # boosted weekend transactions by 80%
    
    # small boost around month-end/start (usually payday is around that time)
    if date.day in [1, 2, 28, 29, 30, 31]:
        transaction_date_weights[i] *= 1.3
        
    # Major holidays (eg,Christmas)
    if date.month == 12: # December
        transaction_date_weights[i] *= 2.2 # Big spike!
    if date.month in [7, 8]: # Summer vacation months
        transaction_date_weights[i] *= 1.4

# Normalize these weights so they sum to 1, turning them into probabilities
transaction_date_weights /= transaction_date_weights.sum()
print("Transaction seasonality weights prepared.")


# generation of customer data
customer_ids = np.arange(100000, 100000 + n_customers) # sequential IDs, as used in banking

# Basic demographics
ages = np.random.randint(18, 80, size=n_customers) # strictly ages must be >=18 
genders = np.random.choice(['M', 'F', 'X'], size=n_customers, p=[0.48, 0.48, 0.04]) 

# GDPR consent is crucial also
gdpr_email = np.random.binomial(1, 0.65, size=n_customers) # 65% opt in for email
gdpr_sms = np.random.binomial(1, 0.55, size=n_customers)   # 55% opt in for SMS
gdpr_inapp = np.random.binomial(1, 0.75, size=n_customers) # inapp often has higher consent

# Overdue obligations: mostly zero, but some customers have delays
# This distribution is more realistic than just a random spread.
n_zero_overdue = int(n_customers * 0.8) # 80% of customers are current
n_medium_overdue = int(n_customers * 0.15) # 15% are 1-60 days overdue
n_high_overdue = n_customers - n_zero_overdue - n_medium_overdue # The remaining 5% are 61-180 days overdue

overdue_zero = np.zeros(n_zero_overdue)
overdue_medium = np.random.randint(1, 61, size=n_medium_overdue)
overdue_high = np.random.randint(61, 181, size=n_high_overdue)
overdue_days = np.concatenate([overdue_zero, overdue_medium, overdue_high])
np.random.shuffle(overdue_days) # Important to shuffle after concatenating,to prevent unrealistic correlations


# pre existing customer segments and card types
# This influences behavior and often how we'd market to them.
segment_types = np.random.choice(
    ['Affluent', 'Mass Market', 'Private Banking', 'Student'],
    size=n_customers,
    p=[0.15, 0.60, 0.05, 0.20] # Define proportions for each segment
)
card_types = np.random.choice(
    ['Gold Credit', 'Standard Credit', 'Debit Plus', 'Standard Debit'],
    size=n_customers,
    p=[0.2, 0.3, 0.3, 0.2] # What kind of cards do they hold?
)

# Start building the customers DF
customers = pd.DataFrame({
    'customer_id': customer_ids,
    'age': ages,
    'gender': genders,
    'gdpr_email': gdpr_email,
    'gdpr_sms': gdpr_sms,
    'gdpr_inapp': gdpr_inapp,
    'overdue_days': overdue_days,
    'bank_segment': segment_types,
    'card_type': card_types
})

# Loyalty points balance to vary by bank segment
# Affluent and Private Banking customers should generally have more points.
def get_points_base_scale(segment):
    if segment == 'Private Banking': return 1500 # High base for private clients
    if segment == 'Affluent': return 800      # Good base for affluent
    if segment == 'Mass Market': return 200       # Standard for mass market
    if segment == 'Student': return 50           # Lower for students
    return 100 # Default/fallback

customers['points_scale_helper'] = customers['bank_segment'].apply(get_points_base_scale)
customers['points_balance'] = customers.apply(
    lambda row: int(np.round(np.random.exponential(scale=row['points_scale_helper']))),
    axis=1
)
customers = customers.drop(columns=['points_scale_helper']) # Clean up our helper column

print("Customer master table generated with segments and loyalty points.")


# Transaction data generation 

# merchant categories and their typical transaction sizes
categories = ['Technology', 'Groceries', 'Food & Beverage', 'Clothing', 'Services', 'Travel', 'Other']

# aaverage transaction amounts by category. (Groceries are small, Travel is big)
category_avg_amounts = {
    'Technology': 120.0, 'Groceries': 45.0, 'Food & Beverage': 25.0,
    'Clothing': 70.0, 'Services': 60.0, 'Travel': 350.0, 'Other': 30.0
}

# customer segments also influence overall spending levels
segment_overall_spend_multipliers = {
    'Private Banking': 2.5, 'Affluent': 1.8, 'Mass Market': 1.0, 'Student': 0.7
}

# category preferences can also differ by segment
# Students might buy more F&B, Affluent might buy more Tech/Travel.
p_mass_market =      [0.10, 0.25, 0.20, 0.15, 0.10, 0.10, 0.10]
p_affluent_private = [0.20, 0.15, 0.15, 0.15, 0.10, 0.20, 0.05]
p_student =          [0.15, 0.20, 0.30, 0.10, 0.05, 0.05, 0.15]


all_transactions_rows = []
transaction_id_counter = 1



#loop through each customer to generate their transactions
for idx, customer_row in customers.iterrows():
    cust_id = customer_row['customer_id']
    customer_segment = customer_row['bank_segment']
    
  #get relevant multipliers and category probabilities for customer
    current_spend_multiplier = segment_overall_spend_multipliers.get(customer_segment, 1.0)
    
    if customer_segment in ['Affluent', 'Private Banking']:
        current_category_probs = p_affluent_private
    elif customer_segment == 'Student':
        current_category_probs = p_student
    else: # Default for Mass Market
        current_category_probs = p_mass_market
        
    num_transactions = np.random.poisson(avg_tx_per_customer) #each customer gets a varying number of transactions
    
    for _ in range(max(num_transactions, 1)): #ensure at least one transaction for every customer
        #pick a transaction date keeping in mindseasonal weights
        tx_date = np.random.choice(all_dates, p=transaction_date_weights)
        
        #select category based on segment preferences
        tx_category = np.random.choice(categories, p=current_category_probs)
        
        #calculate amount,exponential distribution for realism (many small, few large)
        # Scaled by category typical amount and customer's overall spend multiplier
        base_category_scale = category_avg_amounts.get(tx_category, 50.0)
        tx_amount = np.round(np.random.exponential(scale=base_category_scale) * current_spend_multiplier, 2)
        tx_amount = max(tx_amount, 1.0) # Ensure no zero or negative transactions
        
        tx_id = f"TX{transaction_id_counter:07d}" #unique transaction ID
        transaction_id_counter += 1
        points_earned = int(np.floor(tx_amount)) #1 point per euro
        
        all_transactions_rows.append([tx_id, cust_id, tx_date, tx_amount, tx_category, points_earned])

transactions = pd.DataFrame(all_transactions_rows, columns=['transaction_id', 'customer_id', 'date', 'amount', 'category', 'points_earned'])
transactions['date'] = pd.to_datetime(transactions['date']) # Make sure date is a proper datetime object

print(f"Done! Generated {len(transactions)} individual transactions.")


#Customer-Level Feature Aggregation 
# before i get to RFM,add some more customer-level features based on transactions.
# for example, what proportion of their spending is on technology?
tech_counts = transactions[transactions['category'] == 'Technology'].groupby('customer_id').size().rename('tech_tx_count')
total_counts = transactions.groupby('customer_id').size().rename('total_tx_count')

#merge these back and calculate the ratio
customer_activity_summary = pd.concat([tech_counts, total_counts], axis=1).fillna(0)
customer_activity_summary['tech_purchase_ratio'] = customer_activity_summary['tech_tx_count'] / customer_activity_summary['total_tx_count']
customer_activity_summary = customer_activity_summary.reset_index()

#add these new features to main customers df
customers = customers.merge(customer_activity_summary[['customer_id', 'tech_purchase_ratio', 'total_tx_count']], on='customer_id', how='left')
customers['tech_purchase_ratio'] = customers['tech_purchase_ratio'].fillna(0) # For customers with no tech purchases
customers['total_tx_count'] = customers['total_tx_count'].fillna(0).astype(int) #should match frequency later


#campaign Targeting logic & control group
# Define who is eligible for "Tech_Points" campaign.
# The bank has some rules: target top tech buyers, no overdue debt, age > 18.
tech_buyer_threshold = customers['tech_purchase_ratio'].quantile(0.70) #target the top 30% of tech buyers
customers['is_candidate'] = ((customers['tech_purchase_ratio'] >= tech_buyer_threshold) &
                             (customers['overdue_days'] < 60) & #no customers with significant overdue debt
                             (customers['age'] >= 18)) #this is already ensured, but good to include for clarity

#Correct campaign evaluation, we need a control group
#some eligible candidates wont receive the campaign
#hold out 20% eligible candidates as a control group
customers['is_control_group'] = (np.random.rand(n_customers) < 0.20) & customers['is_candidate']
customers['is_control_group'] = customers['is_control_group'].astype(int) #convert boolean to int (0 or 1)


#determine the best channel to send the campaign, based on customer GDPR consent
def pick_send_channel(row):
    available_channels = []
    if row['gdpr_email'] == 1: available_channels.append('email')
    if row['gdpr_sms'] == 1: available_channels.append('sms')
    if row['gdpr_inapp'] == 1: available_channels.append('in_app')
    if not available_channels: return None #if no consent, no channel
    return np.random.choice(available_channels) #pick one if multiple available
customers['send_channel'] = customers.apply(pick_send_channel, axis=1)

#a customer is "targeted" if they are a candidate,not in the control group,and we have a channel
customers['targeted'] = (customers['is_candidate'] &
                         (customers['is_control_group'] == 0) & #exclude control group from targeting
                         customers['send_channel'].notnull())
customers['targeted'] = customers['targeted'].astype(int) #convert boolean to int

print(f"Campaign eligibility and targeting defined.")
print(f" -> Total candidates: {customers['is_candidate'].sum()}")
print(f" -> Customers in Target Group (received campaign): {customers['targeted'].sum()}")
print(f" -> Customers in Control Group (eligible but held out): {customers['is_control_group'].sum()}")


#simulate Campaign Clicks
#not everyone who gets the campaign clicks on it, must simulate that behavior
def calculate_click_probability(row):
    if row['targeted'] == 0: #only targeted customers can click
        return 0.0
    
    base_click_prob = 0.04 #baseline probability
    
    #factors that might increase click probability
    tech_affinity_boost = row['tech_purchase_ratio'] * 0.4 #tech buyers are more interested in tech offers
    loyalty_points_boost = min(0.15, row['points_balance'] / 5000) #customers with more points might be more engaged
    
    #different channels have different effectiveness
    channel_effectiveness = {'email': 0.9, 'sms': 1.1, 'in_app': 1.3}
    channel_factor = channel_effectiveness.get(row['send_channel'], 1.0) #default to 1.0 if channel is somehow missing
    
    #combine factors,nsure probability stays between 0 and 1
    prob = (base_click_prob + tech_affinity_boost + loyalty_points_boost) * channel_factor
    return min(prob, 0.85) #cap max probability to 85%

customers['click_prob'] = customers.apply(calculate_click_probability, axis=1)
customers['campaign_clicked'] = (np.random.rand(len(customers)) < customers['click_prob']).astype(int) #simulate click based on probability


# Create a separate campaign log, just how a bank stores these events
campaign_log_rows = []
for _, row_data in customers.iterrows():
    #log any customer who was a candidate, whether they were targeted or in control
    if row_data['is_candidate']:
        campaign_log_rows.append({
            'customer_id': row_data['customer_id'],
            'campaign_name': 'Tech_Points_40_to_1_euro_2025Q4',
            'campaign_date': campaign_date.strftime('%Y-%m-%d'),
            'send_channel': row_data['send_channel'] if row_data['targeted'] == 1 else 'N/A_Control', # Control group didn't get a "send" per se
            'is_target_group': row_data['targeted'],
            'is_control_group': row_data['is_control_group'],
            'clicked': row_data['campaign_clicked'] #only targeted customers can have 1 here
        })
campaign = pd.DataFrame(campaign_log_rows)


#calculate RFM

#Recency,Days since last transaction
last_transaction_date = transactions.groupby('customer_id')['date'].max().rename('last_tx_date')
#Frequency,ttl number of transactions
total_frequency = transactions.groupby('customer_id').size().rename('frequency')
#Monetary,ttl amount spent
total_monetary = transactions.groupby('customer_id')['amount'].sum().rename('monetary')

#rfm df
rfm_raw = pd.concat([last_transaction_date, total_frequency, total_monetary], axis=1).reset_index()

#customers with no transactions (they wont appear in the groupby initially)
rfm_raw['last_tx_date'] = rfm_raw['last_tx_date'].replace(pd.NaT, pd.Timestamp(start_date)) #if no transactions, assume "start_date" for calculation base
rfm_raw['recency'] = (pd.to_datetime(snapshot_date) - rfm_raw['last_tx_date']).dt.days
#for customers with no transactions,their recency will be max possible
rfm_raw['recency'] = rfm_raw['recency'].fillna((snapshot_date - start_date).days).astype(int)

#relevant RFM columns for merging
rfm_for_merge = rfm_raw[['customer_id', 'recency', 'frequency', 'monetary']]

#merge rfm data back into the main customers df
customers = customers.merge(rfm_for_merge, on='customer_id', how='left')
#fill the NaNs for any customer who might not have had any transactions(should be rare with poisson, but robust)
customers['frequency'] = customers['frequency'].fillna(0).astype(int)
customers['monetary'] = customers['monetary'].fillna(0.0)

print("RFM metrics calculated for all customers as of snapshot date.")


#simulate churn label for next 90 days
#need a target variable for churn prediction model
#churn probability higher for inactive customers, lower for engaged ones
#also clicking the campaign should reduce churn risk

#normalize RFM factors so they contribute proportionally to churn probability
recency_normalized = customers['recency'] / (customers['recency'].max() + 1) #higher recency ->higher churn risk
frequency_normalized = 1 - (customers['frequency'] / (customers['frequency'].max() + 1)) #lower frequency ->higher churn risk
points_balance_normalized = 1 - (customers['points_balance'] / (customers['points_balance'].max() + 1)) #more points ->lower churn risk

#the campaign click is a positive signaso it should reduce churn probability
click_effect_on_churn = np.where(customers['campaign_clicked'] == 1, -0.25, 0.0) # clicking reduces churn prob by 25%

# combine churn probability
customers['churn_prob'] = 0.15 + \
                          0.5 * recency_normalized + \
                          0.25 * frequency_normalized + \
                          0.1 * points_balance_normalized + \
                          click_effect_on_churn

#probability  (eg, min 2%, max 95%)
customers['churn_prob'] = customers['churn_prob'].clip(0.02, 0.95)

#actual churn based on the calculated probability
customers['is_churn_next_90d'] = (np.random.rand(len(customers)) < customers['churn_prob']).astype(int)



output_directory = Path("data")
output_directory.mkdir(exist_ok=True)


customers_final_export = customers.copy().drop(columns=['is_candidate', 'click_prob'])

#Export DataFrames to CSV files
customers_final_export.to_csv(output_directory / 'customers.csv', index=False)
transactions.to_csv(output_directory / 'transactions.csv', index=False)
campaign.to_csv(output_directory / 'campaign.csv', index=False)

print("\n" + "=" * 50)
print(f"Data generation complete! Files saved to '{output_directory.resolve()}'")
print(f" - customers.csv ({len(customers_final_export)} rows)")
print(f" - transactions.csv ({len(transactions)} rows)")
print(f" - campaign.csv ({len(campaign)} rows)")
print("=" * 50)