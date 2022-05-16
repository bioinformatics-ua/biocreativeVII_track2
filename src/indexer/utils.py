

def simplify_format(collection):
    mapped_documents = dict()
    

    for _id, document in collection:
        mesh_tuple_list = list()
        for passage in document:
            for entity in passage.nes:
                mesh_tuple_list.append((entity.identifiers, entity.span))

        mapped_documents[_id] = mesh_tuple_list

    return mapped_documents