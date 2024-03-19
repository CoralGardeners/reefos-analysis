# get outplant data from firestore.
# For each outplant, get all corals. Compute stats: number of corals, corals per bommie/structure,
# number of species, species per bommie/structure

# from google.cloud.firestore_v1.base_query import FieldFilter

from cachetools.func import ttl_cache
import reefos_analysis.dbutils.firestore_util as fsu
import pandas as pd
# %%


def clear_cache():
    cached = [get_branch_fragments, get_nursery_info, get_branch_fragments,
              get_nursery_info, get_nursery_fragments, get_branch_collection, get_outplant_info]
    for cached_fn in cached:
        with cached_fn.cache_lock:
            cached_fn.cache.clear()


def update_outplant_stats(branches, write_stats=True):
    def get_outplant_stats(op_info, op_id):
        # compute:
        #    total number of frags planted
        #    average frags per cell/bommie
        #    total number of spp planted
        #    average number of spp/cell/bommie
        cell_frag_counts = []
        cell_frag_corals = []
        for cell, frags in op_info['cell_fragments'].items():
            cell_frag_counts.append(len(frags))
            cell_frag_corals.append(
                pd.DataFrame([frag.get('coral') for frag in frags]).drop_duplicates())
        spp_by_cell = [len(df) for df in cell_frag_corals]
        # compute outplant stats
        stats = {
            'total_corals': sum(cell_frag_counts),
            'mean_corals_per_cell': round(sum(cell_frag_counts) / len(cell_frag_counts), 2),
            'total_coral_species': len(pd.concat(cell_frag_corals).drop_duplicates()),
            'spp_per_cell': round(sum(spp_by_cell) / len(spp_by_cell), 2)
            }
        return stats

    # get outplant data for each branch
    for branch_doc in branches.list_documents():
        # get collections in branch and data about the branch (name and location)
        branch_collections = {coll.id: coll for coll in branch_doc.collections()}
        # get fragments for the branch
        if fsu.collections['fragments'] not in branch_collections:
            continue
        fragment_collection = branch_collections[fsu.collections['fragments']]

        # process each outplant in branch
        if fsu.collections['outplants'] in branch_collections:
            # get outplant info
            print("Getting outplant info")
            outplant_collection = branch_collections[fsu.collections['outplants']]
            for outplant in outplant_collection.list_documents():
                op_id = outplant.id
                op_info = outplant.get().to_dict()
                op_coll = {coll.id: coll for coll in outplant.collections()}
                cells = [doc.id for doc in op_coll['OutplantCells'].list_documents()]
                op_info['cells'] = cells
                # query fragment collection for fragments (corals) planted on each cell/bommie in each outplant
                cell_fragments = {}
                for idx, cell_id in enumerate(op_info['cells']):
                    print(f"Getting fragments in cell {idx + 1} of {len(op_info['cells'])}")
                    cell_fragments[cell_id] = fragment_collection.where("outplantInfo.outplantCellID",
                                                                        "==",
                                                                        cell_id).get()
                op_info['cell_fragments'] = cell_fragments
                # compute stats from outplant data
                stats = get_outplant_stats(op_info, op_id)
                # save stats to firestore
                if write_stats:
                    outplant.set({"stats": stats}, merge=True)
                else:   # for debugging
                    print(stats)


def get_branch_collection_count(bd_ref, collection):
    # get count of items in a collection for a branch
    coll = bd_ref.collection(fsu.collections[collection])
    return coll.count().get()[0][0].value


@ttl_cache(maxsize=20, ttl=43200)
def get_branch_fragments(branch_doc_path):
    keep = {'previousLocations', 'colonyID', 'created', 'lost'}
    # get fragments for the branch
    bd_ref = fsu.get_reference(branch_doc_path)
    fragment_collection = bd_ref.collection(fsu.collections['fragments'])

    # get fragments (corals) in the branch
    fragments = fragment_collection.get()
    frag_data = []
    frag_health_data = []
    for frag in fragments:
        frag_id = frag.id
        frag = frag.to_dict()
        # make sure location has a good value
        frag['location'] = frag.get('location', {}) or {}
        # copy and flatten frag data
        data = {k: v for k, v in frag.items() if k in keep}
        data = (data | frag.get('location', {}) | frag.get('coral', {})
                | frag.get('monitoring', {}) | frag.get('outplantInfo', {}))
        data['id'] = frag_id
        frag_data.append(data)
        for hd in frag['healths']:
            hd['id'] = frag_id
            hd['nurseryID'] = data.get('nurseryID')
            hd['colonyID'] = data.get('colonyID')
            hd['structureID'] = data.get('structureID')
            frag_health_data.append(hd)
    return frag_data, frag_health_data


@ttl_cache(maxsize=20, ttl=43200)
def get_nursery_info(nursery_doc_path):
    # get fragments for the branch
    nd_ref = fsu.get_reference(nursery_doc_path)

    nu_info = nd_ref.get().to_dict()
    nu_info['id'] = nd_ref.id
    return nu_info


@ttl_cache(maxsize=20, ttl=43200)
def get_nursery_fragments(branch_doc_path, nursery_doc_path):
    # get fragments for the branch
    bd_ref = fsu.get_reference(branch_doc_path)
    nd_ref = fsu.get_reference(nursery_doc_path)
    fragment_collection = bd_ref.collection(fsu.collections['fragments'])

    nu_info = nd_ref.get().to_dict()
    nu_info['id'] = nd_ref.id
    # query fragment collection for fragments (corals) in the nursery
    nursery_fragments = fragment_collection.where("location.nurseryID",
                                                  "==",
                                                  nd_ref.id).get()
    nu_info['fragments'] = {frag.id: frag for frag in nursery_fragments}
    return nu_info


def get_nursery_stats(nu_info, include_structures=False):
    # compute:
    #    total number of frags planted
    #    total number of spp planted
    #    number of structures (tables, trees) in the nursery
    frag_counts = len(nu_info['fragments'])
    spp_df = pd.DataFrame([frag.get('coral') for frag in nu_info['fragments'].values()]).drop_duplicates()
    # compute nursery stats
    stats = {
        'total_corals': frag_counts,
        'total_coral_species': len(spp_df),
        }
    if include_structures:
        loc_df = pd.DataFrame([frag.get('location') for frag in nu_info['fragments'].values()]).drop_duplicates()
        stats['total_nursery_structures'] = len(loc_df.structure.unique())
    return stats


def update_nursery_stats(branches, write_stats=True):
    # get outplant data for each branch
    for bd_ref in branches.list_documents():
        fragment_collection = bd_ref.collection(fsu.collections['fragments'])
        nursery_collection = bd_ref.collection(fsu.collections['nurseries'])

        # process each nursery in the branch
        print("Getting nursery info")
        for nursery in nursery_collection.list_documents():
            nu_id = nursery.id
            nu_info = nursery.get().to_dict()
            # query fragment collection for fragments (corals) in the nursery
            nursery_fragments = fragment_collection.where("location.nurseryID",
                                                          "==",
                                                          nu_id).get()
            nu_info['fragments'] = {frag.id: frag for frag in nursery_fragments}
            # compute stats from outplant data
            stats = get_nursery_stats(nu_info)
            # save stats to firestore
            if write_stats:
                nursery.set({"stats": stats}, merge=True)
            else:   # for debugging
                print(f"{nu_info['name']}: {stats}")


@ttl_cache(maxsize=20, ttl=43200)
def get_branch_collection(branch_doc_path, collection_type):
    bd_ref = fsu.get_reference(branch_doc_path)
    coll = bd_ref.collection(fsu.collections[collection_type])
    return coll if coll.count().get()[0][0].value else None


@ttl_cache(maxsize=20, ttl=43200)
def get_outplant_info(outplant_doc_path):
    op_ref = fsu.get_reference(outplant_doc_path)
    op_info = op_ref.get().to_dict()
    op_info['id'] = op_ref.id
    if 'stats' not in op_info:
        op_info['stats'] = {'total_corals': 0}
    return op_info


def get_outplant_cells_fragments(branch_doc_path, outplant_doc_path):
    bd_ref = fsu.get_reference(branch_doc_path)
    op_ref = fsu.get_reference(outplant_doc_path)
    # get fragments
    fragment_collection = bd_ref.collection(fsu.collections['fragments'])
    # get outplant info
    op_info = op_ref.get().to_dict()
    op_info['id'] = op_ref.id
    # get cells in this outplant
    cells = op_ref.collection('OutplantCells')
    cell_map = {cell.id: cell for cell in cells.get()}
    # get fragments with matching cell_ids, query only takes up to 30 cellids
    cell_ids = list(cell_map.keys())
    outplant_fragments = []
    idx = 0
    max_incr = 30
    cnt = len(cell_ids)
    while idx < cnt:
        # query fragment collection for outplanted corals
        outplant_fragments.extend(fragment_collection.where("outplantInfo.outplantCellID",
                                                            "in",
                                                            cell_ids[idx:idx + 30]).get())
        idx += max_incr
    # add fragments and cells to outplant info
    op_info['fragments'] = {frag.id: frag for frag in outplant_fragments}
    op_info['cells'] = cell_map
    return op_info


def get_outplant_stats(op_info):
    # compute:
    #    total number of frags planted
    #    total number of spp planted
    cell_counts = len(op_info['cells'])
    frag_counts = len(op_info['fragments'])
    spp_df = pd.DataFrame([frag.get('coral') for frag in op_info['fragments'].values()]).drop_duplicates()
    # compute outplant stats
    stats = {
        'number_cells': cell_counts,
        'total_corals': frag_counts,
        'total_coral_species': len(spp_df)
        }
    return stats
