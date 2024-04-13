
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
from scipy import optimize

#Calculate the share of the total number of movies for each country
total_movies_dk = filtered_bio_dk_movies['Value'].sum()
total_movies_usa = filtered_bio_usa_movies['Value'].sum()
total_movies_eu = filtered_bio_eu_movies['Value'].sum()

share_movies_dk = total_movies_dk / (total_movies_dk + total_movies_usa + total_movies_eu)
share_movies_usa = total_movies_usa / (total_movies_dk + total_movies_usa + total_movies_eu)
share_movies_eu = total_movies_eu / (total_movies_dk + total_movies_usa + total_movies_eu)


#Calculate the share of the total number of tickets sold for each country by censorships
total_tickets_dk = filtered_bio_dk_tickets['Value'].sum()
total_tickets_usa = filtered_bio_usa_tickets['Value'].sum()
total_tickets_eu = filtered_bio_eu_tickets['Value'].sum()

share_tickets_dk = total_tickets_dk / (total_tickets_dk + total_tickets_usa + total_tickets_eu)
share_tickets_usa = total_tickets_usa / (total_tickets_dk + total_tickets_usa + total_tickets_eu)
share_tickets_eu = total_tickets_eu / (total_tickets_dk + total_tickets_usa + total_tickets_eu)



# Fistly we define a list of selected countries
selected_countries = ['Danmark', 'USA', 'EU-28 udenfor Danmark']

# Then we iterate over selected countries, creating a share in terms of censorship categories for each country
for country in selected_countries:
    # Filter the data for the current country and type
    filtered_data_movies = filtered_bio_cencorship[(filtered_bio_cencorship['Country'] == country) & (filtered_bio_cencorship['Type'] == 'Film (antal)')]
# Create a single plot
fig, ax = plt.subplots(figsize=(10, 6))

# Fistly we define a list of selected countries
selected_countries = ['Danmark', 'USA', 'EU-28 udenfor Danmark']

# Create an empty list to store the filtered dataframes for each country
filtered_data_per_country = []

# Iterate over selected countries
for country in selected_countries:
    # Filter the data for the current country and type
    filtered_data_movies = filtered_bio_cencorship[(filtered_bio_cencorship['Country'] == country) & (filtered_bio_cencorship['Type'] == 'Film (antal)')]
    
    # Calculate the total number of movies for the current country
    total_movies_country = filtered_data_movies['Value'].sum()
    
    # Calculate the share of movies for each censorship category
    filtered_data_movies['Share'] = filtered_data_movies['Value'] / total_movies_country
    
    # Append the filtered dataframe to the list
    filtered_data_per_country.append(filtered_data_movies)

# Concatenate the filtered dataframes for all countries
filtered_data_all_countries = pd.concat(filtered_data_per_country)