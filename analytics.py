
import pandas as pd
import matplotlib.pyplot as plt

def plot():
    # Load the dataset
    data = pd.read_csv("analysis.csv")

    # Group the data by 'County' and calculate the average rent for each county
    avg_rent_per_county = data.groupby('County')['Rent'].mean().reset_index()

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(avg_rent_per_county['County'], avg_rent_per_county['Rent'], color='skyblue')
    
    # Add titles and labels
    plt.xlabel('County')
    plt.ylabel('Average Rent')
    plt.title('Average Rent per County')

    # Save the plot as an image
    plt.savefig('plot.png')

plot()
