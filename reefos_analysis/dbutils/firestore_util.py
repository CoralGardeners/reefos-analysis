import firebase_admin
from firebase_admin import firestore, credentials, initialize_app, storage

collections = {'branches': 'Branches',
               'genera': 'CoralGenera',
               'fragments': 'Fragments',
               'logs': 'MonitoringLogs',
               'mothercolonies': 'MotherColonies',
               'nurseries': 'Nurseries',
               'users': 'Users',
               'outplants': 'OutplantSites',
               'outplantcorals': 'OutplantCells',
               'controls': 'ControlSites',
               }


creds = 'restoration-ios-firebase-adminsdk-wg0a4-a59664d92f.json'
app = None
db = None


def init_firebase_db():
    global app, db
    # initialize talking to firebase
    if app is None:
        print("Initialize Firestore")
        cred = credentials.Certificate(creds)
        app = firebase_admin.initialize_app(cred, {'storageBucket': 'restoration-ios.appspot.com'})
    db = firestore.client()
    return app, db


def cleanup_firestore():
    global app, db
    # cleanup - remove the app
    if app is not None:
        print("Cleanup Firestore")
        firebase_admin.delete_app(app)
        app = None
        db = None


def get_collection(db, coll):
    # get the collection
    db_coll = db.collection(coll)
    return db_coll


def init_firestore(org=None, collection='branches'):
    init_firebase_db()
    if org is not None:
        coll = get_collection(db, 'Orgs')
        return coll.document(org).collection(collections.get(collection, collection))
    return get_collection(db, collections.get(collection, collection))


def get_reference(path):
    if db is None:
        init_firebase_db()
    path = path.split('/')
    return db.document(*path)


def _get_blob(org, branch, fname):
    init_firebase_db()
    if fname[0] == '/':
        fname = fname[1:]
    source_blob = f"{org}/{branch}/{fname}"
    bucket = storage.bucket()
    return bucket.blob(source_blob)


def download_blob(org, branch, fname, dest_file):
    blob = _get_blob(org, branch, fname)
    blob.download_to_filename(dest_file)


def get_blob_url(org, branch, fname):
    blob = _get_blob(org, branch, fname)
    return blob.public_url
