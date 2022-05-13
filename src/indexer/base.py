from core import IModule
from indexer.utils import simplify_format
from annotator.utils import write_collections_to_file
from annotator.corpora import BaseCorpus
from collections import defaultdict
import json

class Indexer(IModule):
    
    def __init__(self, 
                 write_output,
                 write_path,
                 method = 1,
                 min_occur_captions = 0.22,
                 min_occur_abstract = 0.1,
                 min_occur_title = 0.02,
                 min_occur_concl = 0.1,):
        super().__init__()
        self.method = method
        self.min_occur_captions = min_occur_captions
        self.min_occur_abstract = min_occur_abstract
        self.min_occur_title = min_occur_title
        self.min_occur_concl = min_occur_concl
        
        self.write_output = write_output
        self.write_path = write_path
    

    def process(self, corpus, mesh):
        return self.getScores(corpus, mesh, self.getDicts(corpus))
    
    def transform(self, inputs):
        
        ## convert inputs to joao format
        
        collections = defaultdict(dict)
        for group, collection in inputs:
            
            mesh = simplify_format(collection)
            
            # debug remove this!!
            #with open(f"{self.write_path}/only_mesh_{collection.corpus}_{collection.group}", "w") as f:
            #    json.dump(mesh,f)
            
            indexed_collection = self.process(collection, mesh)
        
            collections[collection.corpus][collection.group] = indexed_collection
            
        if self.write_output:
            write_collections_to_file(collections, name = self.write_path)
        
        return BaseCorpus.from_dict(collections)[0]
    
    

    def getDicts(self, corpus):
        dicts = {}
        dicts["sections"] = {} #docID:[section:startSpan]
        dicts["captions"] = {} #docID:[captions:span tuple]
        dicts["abstracts"] = {}#docID:[captions:span tuple]
        dicts["conclusions"] = {}#docID:[captions:span tuple]

        for id, document in corpus:
            dicts["sections"][id] = {}
            dicts["captions"][id] = {}
            dicts["abstracts"][id] = {}
            dicts["conclusions"][id] = {}

            for passage in document:
                if "fig_" in passage.typ or "table_" in passage.typ:
                    dicts["captions"][id][passage.text] = passage.span

                elif "title" in passage.typ or "front" in passage.typ:
                    dicts["sections"][id][passage.text] = passage.span

                if "abstract" in passage.typ:
                    dicts["abstracts"][id][passage.text] = passage.span

                if "CONCL" in passage.section_type:
                    dicts["conclusions"][id][passage.text] = passage.span

        return dicts

    def getScores(self, corpus, meshes, dicts):
        sections = dicts["sections"]
        captions = dicts["captions"]
        abstracts = dicts["abstracts"]
        conclusions = dicts["conclusions"]

        countMeshes = {}
        countAllMeshesByDoc = {}
        for id in meshes:
            countMeshes[id] = {}
            countAllMeshesByDoc[id] = 0
            for x in meshes[id]:
                mesh = x[0][0]
                if mesh == "-":
                    continue
                if mesh not in countMeshes[id]:
                    countMeshes[id][mesh] = 0
                countMeshes[id][mesh] += 1
                countAllMeshesByDoc[id] += 1

        indexedMeshsByDoc = {}
        for id in meshes:
            meshesToAdd = set()
            for x in meshes[id]:
                mesh = x[0][0]
                span = x[1]
                spanStart = span[0]
                spanEnd = span[1]
                if mesh == "-":
                    continue

                if self.method == 1:
                    #Check if mesh is a section title
                    for sec in sections[id]:
                        if spanStart > sections[id][sec][0] and spanEnd < sections[id][sec][1]:
                            #mesh in title
                            if countMeshes[id][mesh]/countAllMeshesByDoc[id] > self.min_occur_title:
                                meshesToAdd.add(mesh)

                    #Check if mesh is in a caption
                    for sec in captions[id]:
                        if spanStart > captions[id][sec][0] and spanEnd < captions[id][sec][1]:
                            #mesh in caption
                            if countMeshes[id][mesh]/countAllMeshesByDoc[id] > self.min_occur_captions:
                                meshesToAdd.add(mesh)

                    #Check if is in abstract
                    for sec in abstracts[id]:
                        if spanStart > abstracts[id][sec][0] and spanEnd < abstracts[id][sec][1]:
                            if countMeshes[id][mesh]/countAllMeshesByDoc[id] > self.min_occur_abstract:
                                meshesToAdd.add(mesh)

                    #Check if is in abstract
                    for sec in conclusions[id]:
                        if spanStart > conclusions[id][sec][0] and spanEnd < conclusions[id][sec][1]:
                            if countMeshes[id][mesh]/countAllMeshesByDoc[id] > self.min_occur_concl:
                                meshesToAdd.add(mesh)

                elif self.method == 2:
                    for sec in sections[id]:
                        if spanStart > sections[id][sec][0] and spanEnd < sections[id][sec][1]:
                            #mesh in title
                            pass#meshesToAdd.add(mesh)
                    
                    #Check if is in abstract
                    for sec in abstracts[id]:
                        if spanStart > abstracts[id][sec][0] and spanEnd < abstracts[id][sec][1]:
                            meshesToAdd.add(mesh)
                            print(mesh)
                            
                    '''
                    #Check if mesh is in a caption
                    for sec in captions[id]:
                        if spanStart > captions[id][sec][0] and spanEnd < captions[id][sec][1]:
                            #mesh in caption
                            if countMeshes[id][mesh]/countAllMeshesByDoc[id] > MIN_OCCUR_CAPTIONS:
                                if mesh in abstractMeshes:
                                    meshesToAdd.add(mesh)
                    '''
                else:
                    print("NO METHOD DEFINED!!!")

            indexedMeshsByDoc[id] = list(meshesToAdd)
            corpus[id].set_mesh_indexing_identifiers(list(meshesToAdd))

        return corpus
