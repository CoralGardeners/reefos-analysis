from google.cloud import storage
from reefos_analysis import detection_io as dio
import reefos_analysis.dbutils.firestore_util as fsu

import datetime as dt

_gcs_client = None
_fs_gcs_client = None


def get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client.from_service_account_json('reefos-4b72e6f5ff78.json')
    return _gcs_client


def get_fs_gcs_client():
    global _fs_gcs_client
    if _fs_gcs_client is None:
        _fs_gcs_client = storage.Client.from_service_account_json(fsu.creds)
    return _fs_gcs_client


def get_gcs_blob_list(bucket_name, name_prefix, start_offset=None, end_offset=None,
                      suffix='.wav', client=None, max_results=1000):
    if client is None:
        client = get_gcs_client()
    # get blobs
    blobs = client.list_blobs(bucket_name, prefix=name_prefix,
                              fields='items(name), nextPageToken',
                              max_results=max_results,
                              start_offset=start_offset, end_offset=end_offset)
    blob_list = list(blobs)
    # filter to only include suffix
    if suffix is not None:
        blob_list = [blob for blob in blob_list if blob.name.endswith(suffix)]
    return blob_list


def download_gcs_file(blob, dst_path=None, client=None):
    if client is None:
        client = get_gcs_client()
    if dst_path is None:
        return blob.download_as_bytes(client)
    # Generate the destination file path
    file_path = dst_path / blob.name.split('/')[-1]
    # Skip downloading if the file already exists locally
    if not file_path.exists():
        with open(file_path, 'wb') as fp:
            client.download_blob_to_file(blob, fp)
            client.close()
            global _gcs_client
            _gcs_client = None


def download_blobs(blob_list, file_path, clean_destination=True):
    # clean the destination - remove files not in download list
    if clean_destination:
        blob_names = [blob.name.split('/')[-1] for blob in blob_list]
        for f in file_path.glob("*"):
            if f.is_file() and f not in blob_names:
                f.unlink()
    # download them
    nfiles = len(blob_list)
    for idx, blob in enumerate(blob_list):
        print(f'Downloading {idx + 1} of {nfiles}')
        download_gcs_file(blob, file_path)


# def get_blobs_of_dates(start_date, end_date=None, blobs=None):
#    blobs = blobs or get_gcs_blob_list()
#    date_blobs = []
#    for blob in blobs:
#        fn = blob.name.split('/')[-1]
#        dt = dio.get_filename_time(fn)
#        if end_date is None:
#            if dt.date() == start_date:
#                date_blobs.append(blob)
#        else:
#            if dt.date() >= start_date and dt.date() <= end_date:
#                date_blobs.append(blob)
#    return date_blobs
