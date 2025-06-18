import cdsapi

c = cdsapi.Client()

c.retrieve(
    'cams-global-reanalysis-eac4',
    {
        'date': '2022-06-01/2022-06-03',
        'type': 'analysis',
        'variable': ['nitrogen_dioxide'],
        'format': 'netcdf',
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'pressure_level': ['500'],  # opsiyonel katman
    },
    'data/NO2_June2022.nc')
