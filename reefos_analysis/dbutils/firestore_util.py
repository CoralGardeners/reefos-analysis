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
               'outplantcorals': 'OutplantCells'}


creds = 'restoration-ios-firebase-adminsdk-wg0a4-a59664d92f.json'
app = None


def get_firebase_db():
    global app
    # initialize talking to firebase
    if app is None:
        cred = credentials.Certificate(creds)
        app = firebase_admin.initialize_app(cred)
    db = firestore.client()
    return app, db


def cleanup_firestore():
    global app
    # cleanup - remove the app
    if app is not None:
        firebase_admin.delete_app(app)
        app = None


def get_collection(db, coll):
    # get the collection
    db_coll = db.collection(coll)
    return db_coll


def init_firestore(collection='branches'):
    app, db = get_firebase_db()
    return get_collection(db, collections[collection])
