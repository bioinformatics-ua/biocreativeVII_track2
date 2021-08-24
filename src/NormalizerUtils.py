import os
import json
import tempfile
import subprocess

def dictionaryLoader(meshDictionaryTypes):

    meshDict = dict()
    if "MeSH_D01_04" in meshDictionaryTypes:
        with open("../dataset/MeSH_DICTIONARIES/filteredMeSH2021_D01-D02-D03-D04.json","r") as file:
            meshJSON = json.load(file)

        for entry in meshJSON:
            meshDict[entry["DescriptorName"].lower()] = entry["DescriptorUI"]
            for concept in entry["Concepts"]:
                if "ConceptCASN1Name" in concept.keys():
                    meshDict[concept["ConceptCASN1Name"].lower()] = entry["DescriptorUI"]
                for entryTerm in concept["EntryTerms"]:
                    meshDict[entryTerm.lower()] = entry["DescriptorUI"]

    if "MeSH_Dxx" in meshDictionaryTypes:
        with open("../dataset/MeSH_DICTIONARIES/full_Dsection_MeSH_dictionary.json","r") as file:
            meshJSON = json.load(file)

        for entry in meshJSON:
            meshDict[entry["DescriptorName"].lower()] = entry["DescriptorUI"]
            for concept in entry["Concepts"]:
                if "ConceptCASN1Name" in concept.keys():
                    meshDict[concept["ConceptCASN1Name"].lower()] = entry["DescriptorUI"]
                for entryTerm in concept["EntryTerms"]:
                    meshDict[entryTerm.lower()] = entry["DescriptorUI"]

    if "SCR" in meshDictionaryTypes:
        with open("../dataset/MeSH_DICTIONARIES/filteredSCR2021.json","r") as file:
            meshJSON = json.load(file)

        for entry in meshJSON:
            meshDict[entry["DescriptorName"].lower()] = entry["DescriptorUI"]
            for concept in entry["Concepts"]:
                if "ConceptCASN1Name" in concept.keys():
                    meshDict[concept["ConceptCASN1Name"].lower()] = entry["DescriptorUI"]
                if "ConceptName" in concept.keys():
                    meshDict[concept["ConceptName"].lower()] = entry["DescriptorUI"]
            try:
                for mapping in concept["HeadingMappings"]:
                    meshDict[mapping["HeadingMappedName"].lower()] = meshDict["HeadingMappedUI"].strip("*")
            except KeyError:
                pass

    return meshDict



def mapWithoutAb3P(corpus):

    mappedDocuments = dict()
    mapped=0
    unmapped=0

    for id, document in corpus:
        meshTupleList = set()
        for passage in document:
            for entity in passage.nes:
                # print(entity.text, entity.identifiers)
                #if entity.text in meshDict.keys():
                if entity.text in meshDict.keys():
                    meshCode = "MESH:" + meshDict[entity.text]
                    meshTupleList.append(([meshCode], entity.span))
                    entity.set_identifiers([meshCode])
                    mapped+=1
                else:
                    meshTupleList.append((["-"], entity.span))
                    entity.set_identifiers(["-"])
                    unmapped+=1
            # print(entity.text)
        # print(entity.identifiers)
        # print(entity.text)

        mappedDocuments[id] = meshTupleList
    print("Mapped entries: {}".format(mapped))
    print("Unmapped entries: {}".format(unmapped))

    return corpus, mappedDocuments



def mapWithAb3P(corpus, meshDictionary, ab3pDictLevel):

    mappedDocuments = dict()
    mapped=0
    unmapped=0

    # COM Ab3P a expandir abreviações, dicionário de abreviações ao nível do documento
    if ab3pDictLevel == "Document":
        for id, document in corpus:
            meshTupleList = list()
            fd, filePath = tempfile.mkstemp()
            try:
                with os.fdopen(fd, 'w') as tmpFile:
                    tmpFile.write(document.text())
                    abbreviationMap = dict()
                    process = subprocess.Popen(["/backup/tools/NCBI/Ab3P/identify_abbr", filePath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    stdout, stderr = process.communicate()
                    ab3pOutput = stdout.split("\n")
                    if len(ab3pOutput)>2:
                        ab3pOutput = ab3pOutput[1:-1]
                        for abbreviation in ab3pOutput:
                            try:
                                abbreviation, longForm, confidenceScore = abbreviation.strip().split("|")
                                abbreviationMap[abbreviation.lower()] = longForm.lower()
                            except ValueError:
                                pass

                    for passage in document:
                        for entity in passage.nes:
                            # if entity.text in meshDict.keys():
                            if entity.text.lower() in meshDictionary.keys():
                                meshCode = "MESH:" + meshDictionary[entity.text.lower()]
                                meshTupleList.append(([meshCode], entity.span))
                                entity.set_identifiers([meshCode])
                                mapped+=1
                            elif entity.text.lower() in abbreviationMap.keys():
                                text = abbreviationMap[entity.text.lower()]
                                if text in meshDictionary.keys():
                                    meshCode = "MESH:" + meshDictionary[text]
                                    meshTupleList.append(([meshCode], entity.span))
                                    entity.set_identifiers([meshCode])
                                    mapped+=1
                                else:
                                    meshTupleList.append((["-"], entity.span))
                                    entity.set_identifiers(["-"])
                                    unmapped+=1
                            else:
                                meshTupleList.append((["-"], entity.span))
                                entity.set_identifiers(["-"])
                                unmapped+=1
            finally:
                os.remove(filePath)

            mappedDocuments[id] = meshTupleList
        print("Mapped entries: {}".format(mapped))
        print("Unmapped entries: {}".format(unmapped))

        return corpus, mappedDocuments



    # COM Ab3P a expandir abreviações, dicionário de abreviações ao nível do corpus
    elif ab3pDictLevel == "Corpus":
        abbreviationMap = dict()
        for id, document in corpus:
            fd, filePath = tempfile.mkstemp()
            try:
                with os.fdopen(fd, 'w') as tmpFile:
                    tmpFile.write(document.text())
                    process = subprocess.Popen(["/backup/tools/NCBI/Ab3P/identify_abbr", filePath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    stdout, stderr = process.communicate()
                    ab3pOutput = stdout.split("\n")
                    if len(ab3pOutput)>2:
                        ab3pOutput = ab3pOutput[1:-1]
                        for abbreviation in ab3pOutput:
                            try:
                                abbreviation, longForm, confidenceScore = abbreviation.strip().split("|")
                                abbreviationMap[abbreviation.lower()] = longForm.lower()
                            except ValueError:
                                pass
            finally:
                os.remove(filePath)

        for id, document in corpus:
            meshTupleList = list()
            for passage in document:
                for entity in passage.nes:
                    # if entity.text in meshDict.keys():
                    if entity.text.lower() in meshDictionary.keys():
                        meshCode = "MESH:" + meshDictionary[entity.text.lower()]
                        meshTupleList.append(([meshCode], entity.span))
                        entity.set_identifiers([meshCode])
                        mapped+=1
                    elif entity.text.lower() in abbreviationMap.keys():
                        text = abbreviationMap[entity.text.lower()]
                        if text in meshDictionary.keys():
                            meshCode = "MESH:" + meshDictionary[text]
                            meshTupleList.append(([meshCode], entity.span))
                            entity.set_identifiers([meshCode])
                            mapped+=1
                        else:
                            meshTupleList.append((["-"], entity.span))
                            entity.set_identifiers(["-"])
                            unmapped+=1
                    else:
                        meshTupleList.append((["-"], entity.span))
                        entity.set_identifiers(["-"])
                        unmapped+=1

            mappedDocuments[id] = meshTupleList
        print("Mapped entries: {}".format(mapped))
        print("Unmapped entries: {}".format(unmapped))

        return corpus, mappedDocuments
