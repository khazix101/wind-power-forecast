import xarray as xr
import pandas as pd
import numpy as np

nc_file = "data\wind_nc\data_stream-oper_stepType-instant_1.nc"
ds = xr.open_dataset(nc_file)
print("1")
print(ds)