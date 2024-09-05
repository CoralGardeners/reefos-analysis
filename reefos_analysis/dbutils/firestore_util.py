import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

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
    print("Initialize Firestore")
    if app is None:
        cred = credentials.Certificate(creds)
        app = firebase_admin.initialize_app(cred)
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
