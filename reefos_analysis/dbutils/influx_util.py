import pandas as pd
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

influx_write_api = None
influx_query_api = None


def setup_influx(influxdb_url, influxdb_token, influxdb_org):
    # Set up Influx
    global influx_write_api
    if influx_write_api is None:
        client = influxdb_client.InfluxDBClient(url=influxdb_url, token=influxdb_token, org=influxdb_org)
        influx_write_api = client.write_api(write_options=SYNCHRONOUS)
    return influx_write_api


def setup_influx_query(influxdb_url, influxdb_token, influxdb_org):
    # Set up Influx query
    global influx_query_api
    if influx_query_api is None:
        client = influxdb_client.InfluxDBClient(url=influxdb_url, token=influxdb_token, org=influxdb_org)
        influx_query_api = client.query_api()
    return influx_query_api


# ###################################################################################################################
# Influx Query Functions

measurement_info = {
    "Temperature": {'measurement': 'sensors',
                    'fields': ['externalTemperature'],
                    'filters': [],
                    'tags': [],
                    'groupagg': None,
                    'units': 'degC'},
    "Fish Community Fractions": {'measurement': 'fish_community_fractions',
                                 'fields': ['fish', 'parrotfish', 'butterflyfish', 'brown_tang', 'surgeonfish'],
                                 'filters': [],
                                 'tags': [],
                                 'groupagg': None,
                                 'units': 'fraction'},
    "Fish Diversity": {'measurement': 'class_detection',
                       'fields': ['count'],
                       'filters': [('_field', "count"), ('version', 4)],
                       'tags': [],
                       'groupagg': ('class', '1d', 'mean'),
                       'units': 'count'},
    "Bioacoustic Indices": {'measurement': 'bioacoustics',
                            'fields': ['ACI', 'ADI', 'BI', 'Hf'],
                            'filters': [],
                            'tags': ['version'],
                            'groupagg': None,
                            'units': None},
    }


def _run_query(env, query, fields):
    # run the query
    query_api = setup_influx_query(env.influxdb_url, env.influxdb_token, env.influxdb_org)
    result = query_api.query(org=env.influxdb_org, query=query)
    results = [rec.values for table in result for rec in table.records]
    df = pd.DataFrame(results).drop(columns=['result', 'table'])
    if '_field' in df:
        df = df[df._field.isin(fields)].sort_values(['_time', '_field'])
    else:
        df = df.rename(columns={'_value': fields[0]})
    return df


def _get_data(env, bucket, start_time, end_time, measurement, filters, fields, tags, groupagg):
    # set up the query
    keep_cols = ['_time', '_value', '_field', 'class', ] + tags
    keep_cols = [f'"{col}"' for col in keep_cols]

    if end_time is None:
        range_str = f'range(start: {start_time})'
    else:
        range_str = f'range(start: {start_time}, stop: {end_time})'
    query_parts = [
        f'from(bucket: "{bucket}")',
        range_str,
        f'filter(fn: (r) => r._measurement == "{measurement}")'
    ]
    filter_parts = [f'filter(fn: (r) => r.{filter[0]} == "{filter[1]}")' for filter in filters]
    keep_part = [f'keep(columns:[{", ".join(keep_cols)}])']
    if groupagg is not None:
        groupagg_parts = [
            f'group(columns: ["{groupagg[0]}"], mode: "by")',
            f'window(every: {groupagg[1]})',
            f'{groupagg[2]}()'
            ]
    else:
        groupagg_parts = []

    query = '|> '.join(query_parts + filter_parts + keep_part + groupagg_parts)
    return _run_query(env, query, fields)


def _get_last(env, bucket, measurement, filters, fields, tags):
    # set up the query
    keep_cols = ['_time', '_value', '_field', 'class', ] + tags
    keep_cols = [f'"{col}"' for col in keep_cols]

    query_parts = [
        f'from(bucket: "{bucket}")',
        'range(start: -100d)',
        f'filter(fn: (r) => r._measurement == "{measurement}")'
    ]
    filter_parts = [f'filter(fn: (r) => r.{filter[0]} == "{filter[1]}")' for filter in filters] + ['last()']
    keep_part = [f'keep(columns:[{", ".join(keep_cols)}])']

    query = '|> '.join(query_parts + filter_parts + keep_part)
    return _run_query(env, query, fields)


def get_timeseries_data(env, device_name, measurement, start_date='-1d', end_date=None):
    m_info = measurement_info[measurement]
    ts_df = _get_data(env, device_name, start_date, end_date,
                      m_info['measurement'],
                      m_info['filters'],
                      m_info['fields'],
                      m_info['tags'],
                      m_info['groupagg'],
                      )
    return ts_df


def get_last_timeseries_data(env, device_name, measurement):
    m_info = measurement_info[measurement]
    ts_df = _get_last(env, device_name,
                      m_info['measurement'],
                      m_info['filters'],
                      m_info['fields'],
                      m_info['tags'],
                      )
    return ts_df
