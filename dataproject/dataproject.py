import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from types import SimpleNamespace
from scipy import optimize
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display


class Dataproject_functions:
    def __init__(self, filename):
        self.filename = filename
        self.bio = None
        self.bio_cleaned = None
        self.bio_melted = None

    def load_data(self):
        self.bio = pd.read_excel(self.filename, skiprows=2)

    def clean_data(self):
        self.bio['Unnamed: 1'] = self.bio['Unnamed: 1'].fillna(method='ffill')
        self.bio['Unnamed: 2'] = self.bio['Unnamed: 2'].fillna(method='ffill')
        self.bio['Unnamed: 3'] = self.bio['Unnamed: 3'].fillna(method='ffill')
        drop_these = ['Unnamed: 0']
        self.bio.drop(drop_these, axis=1, inplace=True)
        self.bio.rename(columns={'Unnamed: 1':'Country', 'Unnamed: 2':'Censorship', 'Unnamed: 3':'Type', 'Unnamed: 4':'Cinema_movies'}, inplace=True)

    def melt_data(self):
        self.bio_melted = self.bio.melt(id_vars=['Country', 'Censorship', 'Type', 'Cinema_movies'], var_name='Year', value_name='Value')
        self.bio_melted['Year'] = self.bio_melted['Year'].str.replace('bio', '').astype(int)
        self.bio_melted = self.bio_melted[self.bio_melted['Year'] >= 2015]

    def filter_data(self, selected_countries, selected_censorship, selected_type, selected_cinema_movies):
        self.filtered_bio = self.bio_melted[(self.bio_melted['Country'].isin(selected_countries)) & 
                                             (self.bio_melted['Censorship'] == selected_censorship) & 
                                             (self.bio_melted['Type'].isin(selected_type)) &
                                             (self.bio_melted['Cinema_movies'] == selected_cinema_movies)]

    def group_data(self):
        self.summary_stats = self.filtered_bio.groupby(['Country','Type'])['Value'].describe()

    def compute_summary_stats_censorship(self):
        self.summary_stats_censorship = self.filtered_bio_cencorship.groupby(['Country', 'Censorship'])['Value'].describe()

    def print_summary_stats(self):
        if self.summary_stats is not None:
            print("Summary Statistics:")
            print(self.summary_stats)
        else:
            print("Summary statistics have not been computed yet.")

    def print_summary_stats_censorship(self):
        if self.summary_stats_censorship is not None:
            print("Summary Statistics (Censorship):")
            print(self.summary_stats_censorship)
        else:
            print("Censorship summary statistics have not been computed yet.")

    def plot_movies_by_country_combined(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define filtered DataFrame for Denmark
        filtered_bio_dk_movies = self.filtered_bio[(self.filtered_bio['Country'] == 'Danmark') & (self.filtered_bio['Type'] == 'Film (antal)')]

        # Define filtered DataFrame for USA
        filtered_bio_usa_movies = self.filtered_bio[(self.filtered_bio['Country'] == 'USA') & (self.filtered_bio['Type'] == 'Film (antal)')]

        # Define filtered DataFrame for EU-28 excluding Denmark
        filtered_bio_eu_movies = self.filtered_bio[(self.filtered_bio['Country'] == 'EU-28 udenfor Danmark') & (self.filtered_bio['Country'] != 'Danmark') & (self.filtered_bio['Type'] == 'Film (antal)')]

        # Plot for Danmark
        ax.plot(filtered_bio_dk_movies['Year'], filtered_bio_dk_movies['Value'], label='Danmark')

        # Plot for USA
        ax.plot(filtered_bio_usa_movies['Year'], filtered_bio_usa_movies['Value'], label='USA')

        # Plot for EU-28 udenfor Danmark
        ax.plot(filtered_bio_eu_movies['Year'], filtered_bio_eu_movies['Value'], label='EU-28 udenfor Danmark')

        # Set labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of movies')
        ax.set_title('Figure 1: Number of movies by country')

        # Add legend
        ax.legend()

        # Show the plot
        plt.show()

    def plot_tickets_by_country_combined(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define filtered DataFrame for Denmark
        filtered_bio_dk_tickets = self.filtered_bio[(self.filtered_bio['Country'] == 'Danmark') & (self.filtered_bio['Type'] == 'Solgte billetter (1.000)')]

        # Define filtered DataFrame for USA
        filtered_bio_usa_tickets = self.filtered_bio[(self.filtered_bio['Country'] == 'USA') & (self.filtered_bio['Type'] == 'Solgte billetter (1.000)')]

        # Define filtered DataFrame for EU-28 excluding Denmark
        filtered_bio_eu_tickets = self.filtered_bio[(self.filtered_bio['Country'] == 'EU-28 udenfor Danmark') & (self.filtered_bio['Country'] != 'Danmark') & (self.filtered_bio['Type'] == 'Solgte billetter (1.000)')]

        # Plot for Danmark
        ax.plot(filtered_bio_dk_tickets['Year'], filtered_bio_dk_tickets['Value'], label='Danmark')

        # Plot for USA
        ax.plot(filtered_bio_usa_tickets['Year'], filtered_bio_usa_tickets['Value'], label='USA')

        # Plot for EU-28 udenfor Danmark
        ax.plot(filtered_bio_eu_tickets['Year'], filtered_bio_eu_tickets['Value'], label='EU-28 udenfor Danmark')

        # Set labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of tickets sold (1.000)')
        ax.set_title('Figure 2: Number of tickets sold by country')

        # Add legend
        ax.legend()

        # Show the plot
        plt.show()

    def filter_and_group_data_censorship(self, selected_countries, selected_censorship_censorship, selected_type_censorship):
        self.filtered_bio_cencorship = self.bio_melted[(self.bio_melted['Country'].isin(selected_countries)) & 
                                                        (self.bio_melted['Censorship'].isin(selected_censorship_censorship)) &
                                                        (self.bio_melted['Type'].isin(selected_type_censorship))]
        self.summary_stats_censorship = self.filtered_bio_cencorship.groupby(['Country', 'Censorship'])['Value'].describe()

    def plot_censorship_graph(self, filtered_bio_cencorship):
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot for 2019
        # Define filtered DataFrame for Denmark for the year 2019
        filtered_bio_dk_2019 = filtered_bio_cencorship[(filtered_bio_cencorship['Country'] == 'Danmark') & (filtered_bio_cencorship['Year'] == 2019)] 

        # Define filtered DataFrame for USA for the year 2019
        filtered_bio_usa_2019 = filtered_bio_cencorship[(filtered_bio_cencorship['Country'] == 'USA') & (filtered_bio_cencorship['Year'] == 2019)]

        # Define filtered DataFrame for EU-28 excluding Denmark for the year 2019
        filtered_bio_eu_2019 = filtered_bio_cencorship[(filtered_bio_cencorship['Country'] == 'EU-28 udenfor Danmark') & (filtered_bio_cencorship['Country'] != 'Danmark') & (filtered_bio_cencorship['Year'] == 2019)]

        # Group the filtered DataFrames by the movie type 'Cinema_movie' and sum the values for 2019
        grouped_bio_dk_2019 = filtered_bio_dk_2019.groupby('Censorship')['Value'].sum().reset_index()
        grouped_bio_usa_2019 = filtered_bio_usa_2019.groupby('Censorship')['Value'].sum().reset_index()
        grouped_bio_eu_2019 = filtered_bio_eu_2019.groupby('Censorship')['Value'].sum().reset_index()

        # Define the width of each bar
        bar_width = 0.2

        # Define the positions for the bars for 2019
        bar_positions_dk_2019 = np.arange(len(grouped_bio_dk_2019['Censorship']))
        bar_positions_usa_2019 = np.arange(len(grouped_bio_usa_2019['Censorship'])) + bar_width
        bar_positions_eu_2019 = np.arange(len(grouped_bio_eu_2019['Censorship'])) + 2 * bar_width

        # Plot for Danmark for 2019
        ax1.bar(bar_positions_dk_2019, grouped_bio_dk_2019['Value'], width=bar_width, label='Danmark')

        # Plot for USA for 2019
        ax1.bar(bar_positions_usa_2019, grouped_bio_usa_2019['Value'], width=bar_width, label='USA')

        # Plot for EU-28 udenfor Danmark for 2019
        ax1.bar(bar_positions_eu_2019, grouped_bio_eu_2019['Value'], width=bar_width, label='EU-28 udenfor Danmark')

        # Set x-axis tick positions and labels for 2019
        ax1.set_xticks(bar_positions_usa_2019)
        ax1.set_xticklabels(grouped_bio_usa_2019['Censorship'])

        # Set labels and title for 2019
        ax1.set_xlabel('Censorship Category')
        ax1.set_ylabel('Number of tickets sold (1.000)')
        ax1.set_title('Figure 3a: Number of tickets sold by censorship category for the year 2019')

        # Rotate x-axis labels for better readability for 2019
        ax1.tick_params(axis='x', rotation=90)

        # Add legend for 2019
        ax1.legend()

        # Plot for 2020
        # Define filtered DataFrame for Denmark for the year 2020
        filtered_bio_dk_2020 = filtered_bio_cencorship[(filtered_bio_cencorship['Country'] == 'Danmark') & (filtered_bio_cencorship['Year'] == 2020)] 

        # Define filtered DataFrame for USA for the year 2020
        filtered_bio_usa_2020 = filtered_bio_cencorship[(filtered_bio_cencorship['Country'] == 'USA') & (filtered_bio_cencorship['Year'] == 2020)]

        # Define filtered DataFrame for EU-28 excluding Denmark for the year 2020
        filtered_bio_eu_2020 = filtered_bio_cencorship[(filtered_bio_cencorship['Country'] == 'EU-28 udenfor Danmark') & (filtered_bio_cencorship['Country'] != 'Danmark') & (filtered_bio_cencorship['Year'] == 2020)]

        # Group the filtered DataFrames by the movie type 'Cinema_movie' and sum the values for 2020
        grouped_bio_dk_2020 = filtered_bio_dk_2020.groupby('Censorship')['Value'].sum().reset_index()
        grouped_bio_usa_2020 = filtered_bio_usa_2020.groupby('Censorship')['Value'].sum().reset_index()
        grouped_bio_eu_2020 = filtered_bio_eu_2020.groupby('Censorship')['Value'].sum().reset_index()

        # Define the positions for the bars for 2020
        bar_positions_dk_2020 = np.arange(len(grouped_bio_dk_2020['Censorship'])) + bar_width * 3
        bar_positions_usa_2020 = np.arange(len(grouped_bio_usa_2020['Censorship'])) + bar_width * 3 + bar_width
        bar_positions_eu_2020 = np.arange(len(grouped_bio_eu_2020['Censorship'])) + bar_width * 3 + 2 * bar_width

        # Plot for Danmark for 2020
        ax2.bar(bar_positions_dk_2020, grouped_bio_dk_2020['Value'], width=bar_width, label='Danmark 2020')

        # Plot for USA for 2020
        ax2.bar(bar_positions_usa_2020, grouped_bio_usa_2020['Value'], width=bar_width, label='USA 2020')

        # Plot for EU-28 udenfor Danmark for 2020
        ax2.bar(bar_positions_eu_2020, grouped_bio_eu_2020['Value'], width=bar_width, label='EU-28 udenfor Danmark 2020')

        # Set x-axis tick positions and labels for 2020
        ax2.set_xticks(bar_positions_usa_2020)
        ax2.set_xticklabels(grouped_bio_usa_2020['Censorship'])

        # Set labels and title for 2020
        ax2.set_xlabel('Censorship Category')
        ax2.set_ylabel('Number of tickets sold (1.000)')
        ax2.set_title('Figure 3b: Number of tickets sold by censorship category for the year 2020')

        # Rotate x-axis labels for better readability for 2020
        ax2.tick_params(axis='x', rotation=90)

        # Add legend for 2020
        ax2.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Set the same y-axis limits for both subplots
        max_value_2019 = max(grouped_bio_dk_2019['Value'].max(), grouped_bio_usa_2019['Value'].max(), grouped_bio_eu_2019['Value'].max())
        max_value_2020 = max(grouped_bio_dk_2020['Value'].max(), grouped_bio_usa_2020['Value'].max(), grouped_bio_eu_2020['Value'].max())
        max_value = max(max_value_2019, max_value_2020)
        ax1.set_ylim(0, max_value)
        ax2.set_ylim(0, max_value)

        # Show the plot
        plt.show()

    def compute_movie_shares(self):
        # Calculate the total number of movies for each country
        total_movies_dk = self.filtered_bio[(self.filtered_bio['Country'] == 'Danmark') & (self.filtered_bio['Type'] == 'Film (antal)')]['Value'].sum()
        total_movies_usa = self.filtered_bio[(self.filtered_bio['Country'] == 'USA') & (self.filtered_bio['Type'] == 'Film (antal)')]['Value'].sum()
        total_movies_eu = self.filtered_bio[(self.filtered_bio['Country'] == 'EU-28 udenfor Danmark') & (self.filtered_bio['Type'] == 'Film (antal)')]['Value'].sum()

        # Calculate shares
        total_movies = total_movies_dk + total_movies_usa + total_movies_eu
        share_movies_dk = total_movies_dk / total_movies
        share_movies_usa = total_movies_usa / total_movies
        share_movies_eu = total_movies_eu / total_movies

        return share_movies_dk, share_movies_usa, share_movies_eu

    def compute_ticket_shares(self):
        # Calculate the total number of tickets sold for each country
        total_tickets_dk = self.filtered_bio[(self.filtered_bio['Country'] == 'Danmark') & (self.filtered_bio['Type'] == 'Solgte billetter (1.000)')]['Value'].sum()
        total_tickets_usa = self.filtered_bio[(self.filtered_bio['Country'] == 'USA') & (self.filtered_bio['Type'] == 'Solgte billetter (1.000)')]['Value'].sum()
        total_tickets_eu = self.filtered_bio[(self.filtered_bio['Country'] == 'EU-28 udenfor Danmark') & (self.filtered_bio['Type'] == 'Solgte billetter (1.000)')]['Value'].sum()

        # Calculate shares
        total_tickets = total_tickets_dk + total_tickets_usa + total_tickets_eu
        share_tickets_dk = total_tickets_dk / total_tickets
        share_tickets_usa = total_tickets_usa / total_tickets
        share_tickets_eu = total_tickets_eu / total_tickets

        return share_tickets_dk, share_tickets_usa, share_tickets_eu

    def print_shares(self):
        share_movies_dk, share_movies_usa, share_movies_eu = self.compute_movie_shares()
        share_tickets_dk, share_tickets_usa, share_tickets_eu = self.compute_ticket_shares()

        print(f'The share of the total number of movies for Denmark is {share_movies_dk:.2f}')
        print(f'The share of the total number of movies for USA is {share_movies_usa:.2f}')
        print(f'The share of the total number of movies for EU-28 excluding Denmark is {share_movies_eu:.2f}')

        print(f'The share of the total number of tickets sold for Denmark is {share_tickets_dk:.2f}')
        print(f'The share of the total number of tickets sold for USA is {share_tickets_usa:.2f}')
        print(f'The share of the total number of tickets sold for EU-28 excluding Denmark is {share_tickets_eu:.2f}')
    
    def plot_censorship_graph_new(self, year, selected_countries):
        # List to store the filtered dataframes for each country
        filtered_data_per_country = []
        
        # Filter the data for the provided year
        filtered_bio_censorship_year = self.filtered_bio_cencorship[self.filtered_bio_cencorship['Year'] == int(year)]

        # Iterate over selected countries
        for country in selected_countries:
            # Filter the data for the current country and type
            filtered_data_movies = filtered_bio_censorship_year[(filtered_bio_censorship_year['Country'] == country) & 
                                                                (filtered_bio_censorship_year['Type'] == 'Film (antal)')]

            if not filtered_data_movies.empty:
                # Aggregate the data by country and censorship category
                aggregated_data = filtered_data_movies.groupby(['Country', 'Censorship']).agg({'Value': 'sum'}).reset_index()

                # Calculate the total number of movies for the current country and year
                total_movies_country = aggregated_data['Value'].sum()

                if total_movies_country > 0:
                    # Calculate the share of movies for each censorship category
                    aggregated_data['Share'] = aggregated_data['Value'] / total_movies_country * 100

                    # Append the aggregated dataframe to the list
                    filtered_data_per_country.append(aggregated_data)
        
        # Concatenate the filtered dataframes for all countries
        if filtered_data_per_country:
            filtered_data_all_countries = pd.concat(filtered_data_per_country)

            # Create the single plot
            plt.figure(figsize=(10, 6))

            # Create the plot
            sns.barplot(data=filtered_data_all_countries, x='Censorship', y='Share', hue='Country')
            plt.title(f'Share of Movies by Censorship Category for Selected Countries (Year: {year})')
            plt.xlabel('Censorship Category')
            plt.ylabel('Share')
            plt.xticks(rotation=90)

            # Show the plot
            plt.tight_layout()
            plt.show()
        else:
            print("No data available for the selected countries and year.")

    def plot_number_of_movies(self, country):
        # Create a single plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define filtered DataFrame for the selected country
        filtered_bio_country_movies = self.filtered_bio[(self.filtered_bio['Country'] == country) & (self.filtered_bio['Type'] == 'Film (antal)')]

        # Plot for the selected country
        ax.plot(filtered_bio_country_movies['Year'], filtered_bio_country_movies['Value'], label=country)

        # Set labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of movies')
        ax.set_title(f'Number of movies by country: {country}')

        # Add legend
        ax.legend()

        # Show the plot
        plt.show()

    # Define the method plot_number_of_movies_interactive
    def plot_number_of_movies_interactive(self, selected_countries):
        # Create a dropdown widget for the selected country
        country_widget = widgets.Dropdown(
            options=selected_countries,
            description='Country:',
            disabled=False
        )

        # Create an interactive plot
        interactive_plot = widgets.interactive(self.plot_number_of_movies, country=country_widget)
        
        # Display the interactive plot
        display(interactive_plot)
