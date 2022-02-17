import pandas as pd
import numpy as np

from pygenesys.technology.technology import Technology

from pygenesys.commodity.resource import electricity, ethos
from pygenesys.commodity.emissions import co2eq
from pygenesys.data.eia_data import get_eia_generators, get_existing_capacity
from pygenesys.utils.tsprocess import choose_distribution_method
curr_data = get_eia_generators()

# Calculate the capacity factor distributions

existing_solar = get_existing_capacity(curr_data, 'IL', 'Solar Photovoltaic')
existing_wind = get_existing_capacity(curr_data, 'IL', 'Onshore Wind Turbine')

data_path = '/home/sdotson/research/2021-dotson-ms/data/fixed_cost_projections_bfill.csv'
fixed_df = pd.read_csv(data_path, parse_dates=True, index_col='year')
data_path = '/home/sdotson/research/2021-dotson-ms/data/capital_cost_projections_bfill.csv'
capital_df = pd.read_csv(data_path, parse_dates=True, index_col='year')
nrel_years = np.array(fixed_df.index.year).astype('int')

solar_capital = np.array(capital_df['UtilityPV']).astype('float')
solar_capital = dict(zip(nrel_years, solar_capital))
solar_fixed = np.array(fixed_df['UtilityPV']).astype('float')
solar_fixed = dict(zip(nrel_years, solar_fixed))

wind_capital = np.array(capital_df['LandbasedWind']).astype('float')
wind_capital = dict(zip(nrel_years, wind_capital))
wind_fixed = np.array(fixed_df['LandbasedWind']).astype('float')
wind_fixed = dict(zip(nrel_years, wind_fixed))


def solar_farm_constructor(datafiles,
                           N_seasons,
                           N_hours,
                           region='IL',
                           input_comm=ethos,
                           output_comm=electricity,
                           efficiency=1.0,
                           tech_lifetime=25,
                           loan_lifetime=10,
                           existing=existing_solar,
                           emissions={co2eq:4.8e-5},
                           cost_fixed=solar_fixed,
                           cost_invest=solar_capital):




    solar_farm_techs = []
    method = choose_distribution_method(N_seasons, N_hours)

    # divide the existing capacity by the number of regions
    N_regions = len(datafiles)
    existing = {k: v / N_regions for k, v in existing.items()}

    for n, file in enumerate(datafiles):

        solar_cf = method(file, N_seasons, N_hours, kind='cf')
        solarfarm = Technology(tech_name = f'SOLAR_FARM_{n+1}',
                               units = 'MWe',
                               tech_sector='electricity',
                               category='renewable',
                               tech_label='p',
                               description=file,
                               capacity_to_activity=8.76,
                               curtailed_tech=True)
        solarfarm.add_regional_data(region=region,
                                     input_comm=input_comm,
                                     output_comm=output_comm,
                                     efficiency=efficiency,
                                     tech_lifetime=tech_lifetime,
                                     loan_lifetime=loan_lifetime,
                                     capacity_factor_tech=solar_cf,
                                     existing=existing,
                                     emissions=emissions,
                                     cost_fixed=solar_fixed,
                                     cost_invest=solar_capital
                                     )
        solar_farm_techs.append(solarfarm)

    return solar_farm_techs


def wind_farm_constructor(datafiles,
                           N_seasons,
                           N_hours,
                           region='IL',
                           input_comm=ethos,
                           output_comm=electricity,
                           efficiency=1.0,
                           tech_lifetime=25,
                           loan_lifetime=10,
                           existing=existing_wind,
                           emissions={co2eq:1.1e-5},
                           cost_fixed=wind_fixed,
                           cost_invest=wind_capital):

    wind_farm_techs = []
    method = choose_distribution_method(N_seasons, N_hours)
    # divide the existing capacity by the number of regions
    N_regions = len(datafiles)
    existing = {k: v / N_regions for k, v in existing.items()}

    for n, file in enumerate(datafiles):
        wind_cf = method(file, N_seasons, N_hours, kind='cf')
        windfarm = Technology(tech_name = f'WIND_FARM_{n+1}',
                               units = 'MWe',
                               tech_sector='electricity',
                               category='renewable',
                               tech_label='p',
                               description=file,
                               capacity_to_activity=8.76,
                               curtailed_tech=True)
        windfarm.add_regional_data(region=region,
                                     input_comm=input_comm,
                                     output_comm=output_comm,
                                     efficiency=efficiency,
                                     tech_lifetime=tech_lifetime,
                                     loan_lifetime=loan_lifetime,
                                     capacity_factor_tech=wind_cf,
                                     existing=existing,
                                     emissions=emissions,
                                     cost_fixed=wind_fixed,
                                     cost_invest=wind_capital
                                     )
        wind_farm_techs.append(windfarm)

    return wind_farm_techs






if __name__ == "__main__":

    print(existing_solar)

    reduced_existing = {k: v / 2 for k, v in existing_solar.items()}

    print(reduced_existing)
