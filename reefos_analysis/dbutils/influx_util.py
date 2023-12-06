import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

influx_write_api = None


def setup_influx(influxdb_url, influxdb_token, influxdb_org):
    # Set up Influx
    global influx_write_api
    if influx_write_api is None:
        write_client = influxdb_client.InfluxDBClient(url=influxdb_url, token=influxdb_token, org=influxdb_org)
        influx_write_api = write_client.write_api(write_options=SYNCHRONOUS)
    return influx_write_api
