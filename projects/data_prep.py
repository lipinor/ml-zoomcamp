import pandas as pd

df22 = pd.read_csv('data/MY2022_Fuel_Consumption_Ratings.csv')
df21 = pd.read_csv('./data/MY2021_Fuel_Consumption_Ratings.csv')
df20 = pd.read_csv('./data/MY2020_Fuel_Consumption_Ratings.csv')
df19 = pd.read_csv('./data/MY2019_Fuel_Consumption_Ratings.csv')
df18 = pd.read_csv('./data/MY2018_Fuel_Consumption_Ratings.csv')

df = pd.concat([df22, df21, df20, df19, df18])
df.columns = df.columns.str.lower()

df.to_csv('data/co2_emission_2018_2022.csv')