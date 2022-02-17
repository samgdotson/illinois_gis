# So the database can be saved in the location from which
# the command is called.
import os
import glob
curr_dir = os.path.dirname(__file__)

# Simulation metadata goes here
scenario_name = 'LC'
emissions_scenario = 'CC30'
version = '01'
folder = 'least_cost'
start_year = 2025  # the first year optimized by the model
end_year = 2050  # the last year optimized by the model
N_years = 6  # the number of years optimized by the model
N_seasons = 12  # the number of "seasons" in the model
N_hours = 24  # the number of hours in a day
N_regions=20
reserve_margin = {'IL':0.15}  # fraction of excess capacity to ensure reliability
discount_rate = 0.05 # The discount rate applied globally.
database_filename = f'{folder}/illinois_{scenario_name}_{N_regions}_{N_seasons}.sqlite'  # where the database will be written



"""
This section performs the regionalization using the SKATER algorithm with
the PySAL library.
"""
import pandas as pd
from spopt.region import RegionKMeansHeuristic
# from spopt.region.skater import Skater
import geopandas as gp
import libpysal
import numpy as np
# from sklearn.metrics import pairwise as skm

print('Calculating Illinois regions with PySAL... \n')

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
# open the file
illinois = gp.read_file('../il_shapefiles/illinois_average_resource.shp')
# set up weights
w = libpysal.weights.Queen.from_dataframe(illinois)
# wind model
attrs_name = ['avg_wind', 'pop_dens']
model_wind = RegionKMeansHeuristic(illinois[attrs_name].values, N_regions, w)
model_wind.solve()
illinois['reg_k_mean_wind'] = model_wind.labels_

# solar model
attrs_name = ['avg_ghi', 'pop_dens']
model_solar = RegionKMeansHeuristic(illinois[attrs_name].values, N_regions, w)
model_solar.solve()
illinois['reg_k_mean_solar'] = model_solar.labels_
county_list_wnd = []
county_list_sol = []
for r in range(N_regions):
    county_w = np.random.choice(illinois[illinois['reg_k_mean_wind']==r].COUNTY_NAM)
    county_s = np.random.choice(illinois[illinois['reg_k_mean_solar']==r].COUNTY_NAM)

    solar_files = glob.glob(f'../psm_transformed/{county_s}_**.csv')[0]
    wind_files = glob.glob(f'../wtk_transformed/{county_w}_**.csv')[0]

    print(solar_files)
    print(wind_files)

    county_list_wnd.append(wind_files)
    county_list_sol.append(solar_files)
    print(county_w, county_s)


#=================================================================================
#=================================================================================
# ELECTRICITY DEMAND DATA
#=================================================================================
#=================================================================================

from pygenesys.commodity.demand import ELC_DEMAND

# Add demand forecast

ELC_DEMAND.add_demand(region='IL',
                      init_demand=1.87e5,
                      start_year=start_year,
                      end_year=end_year,
                      N_years=N_years,
                      growth_rate=0.01,
                      growth_method='linear')

miso_path = '/home/sdotson/research/2021-dotson-ms/data/miso_hourly_demand.csv'

ELC_DEMAND.set_distribution(region='IL',
                            data=miso_path,
                            n_seasons=N_seasons,
                            n_hours=N_hours,
                            normalize=True)


# Import transmission technologies, set regions, import input commodities
from pygenesys.technology.transmission import TRANSMISSION
from pygenesys.commodity.resource import (electricity,
                                          ethos)
TRANSMISSION.add_regional_data(region='IL',
                               input_comm=electricity,
                               output_comm=ELC_DEMAND,
                               efficiency=1.0,
                               tech_lifetime=1000,)

# Import technologies that generate electricity
from pygenesys.technology.electric import NUCLEAR_CONV
from pygenesys.technology.electric import COAL_CONV, NATGAS_CONV, BIOMASS
from pygenesys.technology.electric import COAL_ADV, NATGAS_ADV, NUCLEAR_ADV
from pygenesys.technology.storage import LI_BATTERY

# Import emissions
from pygenesys.commodity.emissions import co2eq, CO2


# Import capacity factor data
from pygenesys.utils.tsprocess import choose_distribution_method

# Calculate the capacity factor distributions
method = choose_distribution_method(N_seasons, N_hours)
years = np.linspace(start_year, end_year, N_years).astype('int')


# Add regional data
import pandas as pd

#=================================================================================
#=================================================================================
# COST DATA
#=================================================================================
#=================================================================================

data_path = '/home/sdotson/research/2021-dotson-ms/data/fixed_cost_projections_bfill.csv'
fixed_df = pd.read_csv(data_path, parse_dates=True, index_col='year')
data_path = '/home/sdotson/research/2021-dotson-ms/data/capital_cost_projections_bfill.csv'
capital_df = pd.read_csv(data_path, parse_dates=True, index_col='year')
nrel_years = np.array(fixed_df.index.year).astype('int')

#=================================================================================
#=================================================================================
# CREATE SOLAR AND WIND TECHNOLOGIES
#=================================================================================
#=================================================================================

# import technology data from EIA
from pygenesys.data.eia_data import get_eia_generators, get_existing_capacity
curr_data = get_eia_generators()

from regional_techs import solar_farm_constructor, wind_farm_constructor

print('Creating Solar Technologies... \n')

solar_techs = solar_farm_constructor(county_list_sol,
                                     N_seasons,
                                     N_hours)
print('Creating Wind Technologies... \n')
wind_techs = wind_farm_constructor(county_list_wnd,
                                    N_seasons,
                                    N_hours)

tech_list = solar_techs+wind_techs

#=================================================================================
#=================================================================================
# CREATE OTHER TECHNOLOGIES
#=================================================================================
#=================================================================================

NUCLEAR_CONV.add_regional_data(region='IL',
                               input_comm=ethos,
                               output_comm=electricity,
                               efficiency=1.0,
                               tech_lifetime=60,
                               loan_lifetime=1,
                               capacity_factor_tech=0.93,
                               emissions={co2eq:1.2e-5},
                               existing=get_existing_capacity(curr_data,
                                                              'IL',
                                                              'Nuclear'),
                               cost_fixed=0.17773741,
                               cost_invest=0.05,
                               cost_variable=0.005811,
                               max_capacity = {
                                               2025:12.42e3,
                                               2030:12.42e3,
                                               2035:12.42e3,
                                               2040:12.42e3,
                                               2045:12.42e3,
                                               2050:12.42e3,},
                                               # 2050:0.0,}, # zero nuclear scenario
                               )

# Multiply capital cost by 2 to simulate cost overruns.
# nuclear_capital = np.array(capital_df['Nuclear']).astype('float')*2
nuclear_capital = np.array(capital_df['Nuclear']).astype('float')
nuclear_capital = dict(zip(nrel_years, nuclear_capital))
nuclear_fixed = np.array(fixed_df['Nuclear']).astype('float')
nuclear_fixed = dict(zip(nrel_years, nuclear_fixed))
NUCLEAR_ADV.add_regional_data(region='IL',
                               input_comm=ethos,
                               output_comm=electricity,
                               efficiency=1.0,
                               tech_lifetime=60,
                               loan_lifetime=10,
                               capacity_factor_tech=0.93,
                               emissions={co2eq:1.2e-5},
                               ramp_up=0.25,
                               ramp_down=0.25,
                               cost_fixed=nuclear_fixed,
                               cost_invest=nuclear_capital,
                               cost_variable=0.009158,
                               # max_capacity = {2050:0.0} # zero nuclear scenario
                               )

ngcc_existing = get_existing_capacity(curr_data,
                                      'IL',
                                      'Natural Gas Fired Combined Cycle')
ngct_existing = get_existing_capacity(curr_data,
                                      'IL',
                                      'Natural Gas Fired Combustion Turbine')
import collections, functools, operator
ng_existing = dict(functools.reduce(operator.add,
                   map(collections.Counter, [ngcc_existing, ngct_existing])))
NATGAS_CONV.add_regional_data(region='IL',
                             input_comm=ethos,
                             output_comm=electricity,
                             efficiency=1.0,
                             tech_lifetime=40,
                             loan_lifetime=25,
                             capacity_factor_tech=0.55,
                             emissions={co2eq:4.9e-4, CO2:1.81e-4},
                             existing=ng_existing,
                             ramp_up=1.0,
                             ramp_down=1.0,
                             cost_fixed=0.0111934,
                             cost_invest=0.95958,
                             cost_variable=0.022387
                             )
ng_capital = np.array(capital_df['NaturalGas-CCS']).astype('float')
ng_capital = dict(zip(nrel_years, ng_capital))
ng_fixed = np.array(fixed_df['NaturalGas-CCS']).astype('float')
ng_fixed = dict(zip(nrel_years, ng_fixed))
NATGAS_ADV.add_regional_data(region='IL',
                             input_comm=ethos,
                             output_comm=electricity,
                             efficiency=1.0,
                             tech_lifetime=40,
                             loan_lifetime=25,
                             capacity_factor_tech=0.55,
                             emissions={co2eq:1.7e-4, CO2:1.81e-5},
                             ramp_up=1.0,
                             ramp_down=1.0,
                             cost_fixed=ng_fixed,
                             cost_invest=ng_capital,
                             cost_variable=0.027475
                             )
COAL_CONV.add_regional_data(region='IL',
                             input_comm=ethos,
                             output_comm=electricity,
                             efficiency=1.0,
                             tech_lifetime=60,
                             loan_lifetime=25,
                             capacity_factor_tech=0.54,
                             emissions={co2eq:8.2e-4, CO2:3.2595e-4},
                             existing=get_existing_capacity(curr_data,
                                                            'IL',
                                                            'Conventional Steam Coal'),
                             ramp_up=0.5,
                             ramp_down=0.5,
                             cost_fixed=0.0407033,
                             cost_invest=1.000,
                             cost_variable=0.021369
                             )
coal_capital = np.array(capital_df['Coal-CCS']).astype('float')
coal_capital = dict(zip(nrel_years, coal_capital))
coal_fixed = np.array(fixed_df['Coal-CCS']).astype('float')
coal_fixed = dict(zip(nrel_years, coal_fixed))
COAL_ADV.add_regional_data(region='IL',
                             input_comm=ethos,
                             output_comm=electricity,
                             efficiency=1.0,
                             tech_lifetime=60,
                             loan_lifetime=25,
                             capacity_factor_tech=0.54,
                             emissions={co2eq:2.2e-4, CO2:3.2595e-5},
                             ramp_up=0.5,
                             ramp_down=0.5,
                             cost_fixed=coal_fixed,
                             cost_invest=coal_capital,
                             cost_variable=0.0366329
                             )

biomass_capital = np.array(capital_df['Biomass']).astype('float')
biomass_capital = dict(zip(nrel_years, biomass_capital))
BIOMASS.add_regional_data(region='IL',
                          input_comm=ethos,
                          output_comm=electricity,
                          efficiency=1.0,
                          tech_lifetime=60,
                          loan_lifetime=25,
                          capacity_factor_tech=0.61,
                          emissions={co2eq:2.3e-4},
                          cost_fixed=0.123,
                          cost_invest=biomass_capital,
                          cost_variable=0.047,
                          # max_capacity = {2025:4.0e3,
                          #                 2030:4.0e3,
                          #                 2035:4.0e3,
                          #                 2040:4.0e3,
                          #                 2045:4.0e3,
                          #                 2050:4.0e3,},
                          )

libatt_capital = np.array(capital_df['Battery']).astype('float')
libatt_capital = dict(zip(nrel_years, libatt_capital))
libatt_fixed = np.array(fixed_df['Battery']).astype('float')
libatt_fixed = dict(zip(nrel_years, libatt_fixed))
LI_BATTERY.add_regional_data(region='IL',
                             input_comm=electricity,
                             output_comm=electricity,
                             efficiency=0.85,
                             capacity_factor_tech=0.2,
                             tech_lifetime=15,
                             loan_lifetime=5,
                             existing=get_existing_capacity(curr_data,
                                                            'IL',
                                                            'Batteries'),
                             emissions={co2eq:2.32e-5},
                             cost_invest=libatt_capital,
                             cost_fixed=libatt_fixed,
                             storage_duration=4)

# 2050 carbon limits

if emissions_scenario == "CC50":
    print('Applying constraints carbon neutral by 2050')
    CO2.add_regional_limit(region='IL',
                           limits={2025:52.34375,
                                   2030:41.875,
                                   2035:31.40625,
                                   2040:20.9375,
                                   2045:10.46875,
                                   2050:0.0})

# 2030 carbon limits
elif emissions_scenario == "CC30":
    print('Applying constraints carbon neutral by 2030')
    CO2.add_regional_limit(region='IL',
                           limits={2025:27.917,
                                   2030:0.0,
                                   2035:0.0,
                                   2040:0.0,
                                   2045:0.0,
                                   2050:0.0,})

else:
    print('No carbon limits -- Business as usual')

demands_list = [ELC_DEMAND]
resources_list = [electricity, ethos]
emissions_list = [co2eq, CO2]




if __name__=='__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [12, 9]
    print(illinois.columns)
    fig , axes = plt.subplots(1,2)
    illinois.plot(ax=axes[0],
                  column='skater_wind',
                  categorical=True,
                  cmap='tab20b',
                  figsize=(12,8),
                  edgecolor='w',
                  legend=True)
    illinois.plot(ax=axes[1],
                  column='skater_solar',
                  categorical=True,
                  cmap='tab20c',
                  figsize=(12,8),
                  edgecolor='w',
                  legend=True)

    axes[0].set_axis_off()
    axes[1].set_axis_off()
    axes[0].set_title(f'Wind Regions: {N_regions} Regions')
    axes[1].set_title(f'Solar Regions: {N_regions} Regions')
    plt.show()
