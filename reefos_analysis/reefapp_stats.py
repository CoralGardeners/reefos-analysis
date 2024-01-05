# get outplant data from firestore.
# For each outplant, get all corals. Compute stats: number of corals, corals per bommie/structure,
# number of species, species per bommie/structure

# from google.cloud.firestore_v1.base_query import FieldFilter

import reefos_analysis.dbutils.firestore_util as fsu
import pandas as pd
# %%


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
    branch_collections = {coll.id: coll for coll in bd_ref.collections()}
    coll = fsu.collections[collection]
    if coll in branch_collections:
        _collection = branch_collections[coll]
        return _collection.count().get()[0][0].value
    return 0


def get_nursery_fragments(branch_doc_path, nursery_doc_path):
    # get fragments for the branch
    bd_ref = fsu.get_reference(branch_doc_path)
    nd_ref = fsu.get_reference(nursery_doc_path)
    branch_collections = {coll.id: coll for coll in bd_ref.collections()}
    fragment_collection = branch_collections[fsu.collections['fragments']]

    nu_info = nd_ref.get().to_dict()
    nu_info['id'] = nd_ref.id
    # query fragment collection for fragments (corals) in the nursery
    nursery_fragments = fragment_collection.where("location.nurseryID",
                                                  "==",
                                                  nd_ref.id).get()
    nu_info['fragments'] = {frag.id: frag for frag in nursery_fragments}
    return nu_info


def get_nursery_stats(nu_info):
    # compute:
    #    total number of frags planted
    #    total number of spp planted
    frag_counts = len(nu_info['fragments'])
    spp_df = pd.DataFrame([frag.get('coral') for frag in nu_info['fragments'].values()]).drop_duplicates()
    # compute nursery stats
    stats = {
        'total_corals': frag_counts,
        'total_coral_species': len(spp_df)
        }
    return stats


def update_nursery_stats(branches, write_stats=True):
    # get outplant data for each branch
    for branch_doc in branches.list_documents():
        # get collections in branch and data about the branch (name and location)
        branch_collections = {coll.id: coll for coll in branch_doc.collections()}
        # get fragments for the branch
        fragment_collection = branch_collections[fsu.collections['fragments']]

        # process each nursery in the branch
        if fsu.collections['nurseries'] in branch_collections:
            # get nursery info
            print("Getting nursery info")
            nursery_collection = branch_collections[fsu.collections['nurseries']]
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


def get_branch_collection(branch_doc_path, collection_type):
    bd_ref = fsu.get_reference(branch_doc_path)
    # get collections in branch and data about the branch (name and location)
    branch_collections = {coll.id: coll for coll in bd_ref.collections()}
    # get collections in the branch
    if fsu.collections[collection_type] in branch_collections:
        # get nursery info
        collection = branch_collections[fsu.collections[collection_type]]
    else:
        collection = None
    return collection


def get_outplant_cells_fragments(branch_doc_path, outplant_doc_path):
    bd_ref = fsu.get_reference(branch_doc_path)
    op_ref = fsu.get_reference(outplant_doc_path)
    # get fragments
    branch_collections = {coll.id: coll for coll in bd_ref.collections()}
    fragment_collection = branch_collections[fsu.collections['fragments']]
    # get outpland info
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
