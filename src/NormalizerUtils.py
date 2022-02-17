import os
import json
import tempfile
import subprocess

ab3P_path = "/backup/tools/NCBI/Ab3P/identify_abbr"

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



def mapWithoutAb3P(corpus, meshDictionary):

    mappedDocuments = dict()
    mapped=0
    unmapped=0

    for id, document in corpus:
        meshTupleList = list()
        for passage in document:
            for entity in passage.nes:
                # print(entity.text, entity.identifiers)
                #if entity.text in meshDictionary.keys():
                if entity.text in meshDictionary.keys():
                    if isinstance(meshDictionary[entity.text], list):
                        meshCode = meshDictionary[entity.text]
                        meshTupleList.append((meshCode, entity.span))
                        entity.set_identifiers(meshCode)
                        mapped+=1
                    else:
                        meshCode = "MESH:" + meshDictionary[entity.text]
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


def mapWithoutAb3P_AugmentedDictionary(corpus, meshDictionary, test=False):

    mappedDocuments = dict()
    mapped=0
    unmapped=0

    for id, document in corpus:
        meshTupleList = list()
        for passage in document:
            for entity in passage.nes:
                # print(entity.text, entity.identifiers)
                #if entity.text in meshDict.keys():
                if entity.text in meshDictionary.keys():
                    if isinstance(meshDictionary[entity.text], list):
                        meshCode = meshDictionary[entity.text]
                        meshTupleList.append((meshCode, entity.span))
                        entity.set_identifiers(meshCode)
                        mapped+=1
                    else:
                        meshCode = "MESH:" + meshDictionary[entity.text]
                        meshTupleList.append(([meshCode], entity.span))
                        entity.set_identifiers([meshCode])
                        mapped+=1
                else:
                    if not test:
                        meshDictionary[entity.text.lower()] = entity.identifiers
                        meshTupleList.append((entity.identifiers, entity.span))
                        entity.set_identifiers(entity.identifiers)
                        mapped+=1
                    else:
                        meshTupleList.append((["-"], entity.span))
                        entity.set_identifiers(["-"])
                        unmapped+=1

        mappedDocuments[id] = meshTupleList
    print("Mapped entries: {}".format(mapped))
    print("Unmapped entries: {}".format(unmapped))

    return corpus, mappedDocuments, meshDictionary





def mapWithAb3P(corpus, meshDictionary, ab3pDictLevel, abbreviationMap=None):

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
                    process = subprocess.Popen([ab3P_path, filePath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
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
                                if isinstance(meshDictionary[entity.text.lower()], list):
                                    meshCode = meshDictionary[entity.text.lower()]
                                    meshTupleList.append((meshCode, entity.span))
                                    entity.set_identifiers(meshCode)
                                    mapped+=1
                                else:
                                    meshCode = "MESH:" + meshDictionary[entity.text.lower()]
                                    meshTupleList.append(([meshCode], entity.span))
                                    entity.set_identifiers([meshCode])
                                    mapped+=1
                            elif entity.text.lower() in abbreviationMap.keys():
                                text = abbreviationMap[entity.text.lower()]
                                if text in meshDictionary.keys():
                                    if isinstance(meshDictionary[text], list):
                                        meshCode = meshDictionary[text]
                                        meshTupleList.append((meshCode, entity.span))
                                        entity.set_identifiers(meshCode)
                                        mapped+=1
                                    else:
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

        return corpus, mappedDocuments, abbreviationMap



    # COM Ab3P a expandir abreviações, dicionário de abreviações ao nível do corpus
    elif ab3pDictLevel == "Corpus":
        if abbreviationMap is None:
            abbreviationMap = dict()
        for id, document in corpus:
            fd, filePath = tempfile.mkstemp()
            try:
                with os.fdopen(fd, 'w') as tmpFile:
                    tmpFile.write(document.text())
                    process = subprocess.Popen([ab3P_path, filePath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
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
                        if isinstance(meshDictionary[entity.text.lower()], list):
                            meshCode = meshDictionary[entity.text.lower()]
                            meshTupleList.append((meshCode, entity.span))
                            entity.set_identifiers(meshCode)
                            mapped+=1
                        else:
                            meshCode = "MESH:" + meshDictionary[entity.text.lower()]
                            meshTupleList.append(([meshCode], entity.span))
                            entity.set_identifiers([meshCode])
                            mapped+=1
                    elif entity.text.lower() in abbreviationMap.keys():
                        text = abbreviationMap[entity.text.lower()]
                        if text in meshDictionary.keys():
                            if isinstance(meshDictionary[text], list):
                                meshCode = meshDictionary[text]
                                meshTupleList.append((meshCode, entity.span))
                                entity.set_identifiers(meshCode)
                                mapped+=1
                            else:
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

        return corpus, mappedDocuments, abbreviationMap



def mapWithAb3P_AugmentedDictionary(corpus, meshDictionary, ab3pDictLevel, abbreviationMap=None, test=False):

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
                    process = subprocess.Popen([ab3P_path, filePath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
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
                                if isinstance(meshDictionary[entity.text.lower()], list):
                                    meshCode = meshDictionary[entity.text.lower()]
                                    meshTupleList.append((meshCode, entity.span))
                                    entity.set_identifiers(meshCode)
                                    mapped+=1
                                else:
                                    meshCode = "MESH:" + meshDictionary[entity.text.lower()]
                                    meshTupleList.append(([meshCode], entity.span))
                                    entity.set_identifiers([meshCode])
                                    mapped+=1
                            elif entity.text.lower() in abbreviationMap.keys():
                                text = abbreviationMap[entity.text.lower()]
                                if text in meshDictionary.keys():
                                    if isinstance(meshDictionary[text], list):
                                        meshCode = meshDictionary[text]
                                        meshTupleList.append((meshCode, entity.span))
                                        entity.set_identifiers(meshCode)
                                        mapped+=1
                                    else:
                                        meshCode = "MESH:" + meshDictionary[text]
                                        meshTupleList.append(([meshCode], entity.span))
                                        entity.set_identifiers([meshCode])
                                        mapped+=1
                                else:
                                    if not test:
                                        meshDictionary[entity.text.lower()] = entity.identifiers
                                        meshTupleList.append((entity.identifiers, entity.span))
                                        entity.set_identifiers(entity.identifiers)
                                        mapped+=1
                                    else:
                                        meshTupleList.append((["-"], entity.span))
                                        entity.set_identifiers(["-"])
                                        unmapped+=1
                            else:
                                if not test:
                                    meshDictionary[entity.text.lower()] = entity.identifiers
                                    meshTupleList.append((entity.identifiers, entity.span))
                                    entity.set_identifiers(entity.identifiers)
                                    mapped+=1
                                else:
                                    meshTupleList.append((["-"], entity.span))
                                    entity.set_identifiers(["-"])
                                    unmapped+=1
            finally:
                os.remove(filePath)

            mappedDocuments[id] = meshTupleList
        print("Mapped entries: {}".format(mapped))
        print("Unmapped entries: {}".format(unmapped))

        return corpus, mappedDocuments, meshDictionary, abbreviationMap



    # COM Ab3P a expandir abreviações, dicionário de abreviações ao nível do corpus
    elif ab3pDictLevel == "Corpus":
        if abbreviationMap is None:
            abbreviationMap = dict()
        for id, document in corpus:
            fd, filePath = tempfile.mkstemp()
            try:
                with os.fdopen(fd, 'w') as tmpFile:
                    tmpFile.write(document.text())
                    process = subprocess.Popen([ab3P_path, filePath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
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
                        if isinstance(meshDictionary[entity.text.lower()], list):
                            meshCode = meshDictionary[entity.text.lower()]
                            meshTupleList.append((meshCode, entity.span))
                            entity.set_identifiers(meshCode)
                            mapped+=1
                        else:
                            meshCode = "MESH:" + meshDictionary[entity.text.lower()]
                            meshTupleList.append(([meshCode], entity.span))
                            entity.set_identifiers([meshCode])
                            mapped+=1
                    elif entity.text.lower() in abbreviationMap.keys():
                        text = abbreviationMap[entity.text.lower()]
                        if text in meshDictionary.keys():
                            if isinstance(meshDictionary[text], list):
                                meshCode = meshDictionary[text]
                                meshTupleList.append((meshCode, entity.span))
                                entity.set_identifiers(meshCode)
                                mapped+=1
                            else:
                                meshCode = "MESH:" + meshDictionary[text]
                                meshTupleList.append(([meshCode], entity.span))
                                entity.set_identifiers([meshCode])
                                mapped+=1
                        else:
                            if not test:
                                meshDictionary[entity.text.lower()] = entity.identifiers
                                meshTupleList.append((entity.identifiers, entity.span))
                                entity.set_identifiers(entity.identifiers)
                                mapped+=1
                            else:
                                meshTupleList.append((["-"], entity.span))
                                entity.set_identifiers(["-"])
                                unmapped+=1
                    else:
                        if not test:
                            meshDictionary[entity.text.lower()] = entity.identifiers
                            meshTupleList.append((entity.identifiers, entity.span))
                            entity.set_identifiers(entity.identifiers)
                            mapped+=1
                        else:
                            meshTupleList.append((["-"], entity.span))
                            entity.set_identifiers(["-"])
                            unmapped+=1

            mappedDocuments[id] = meshTupleList
        print("Mapped entries: {}".format(mapped))
        print("Unmapped entries: {}".format(unmapped))

        return corpus, mappedDocuments, meshDictionary, abbreviationMap
