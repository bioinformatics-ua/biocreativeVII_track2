"""
Author: Joao F Silva
Date: 04/09/2021

This script filters MeSH codes of interest from a MeSH dictionary. MeSH codes are considered valid if they belong to a tree structure
specified in the "validTreeCodes" variable. Practically, a MeSH code is valid if it has a TreeNumber beginning with one of the codes in the list "validTreeCodes".
To tune the list of valid MeSH codes, simply adjust the "validTreeCodes" with the codes of interest.
For a list of possible Tree codes please check https://www.nlm.nih.gov/mesh/2021/download/2021NewMeshHeadingsByCategory.pdf

@params:
    input_mesh_xml_file   - Directory for the xml file containing the full MeSH dictionary
    input_mesh_json_file  - Directory for the json file containing the full MeSH dictionary
    output_mesh_json_file - Destination directory for the json file containing the filtered MeSH dictionary
    save_xml_to_json      - Flag used to define if a .json file is to be saved with the xml content from input_mesh_xml_file converted to json
    valid_mesh_tree_codes - List of MeSH initial tree codes used to filter the dictionary. Eg. D01 D02 D03

The output of this script is a .json file with a list of dicts, where each dict contains a MeSH term and its corresponding information

[
    {
        "DescriptorUI":,
        "DescriptorName":,
        "TreeNumberList": ,
        "Concepts": [{"ConceptName":, "ConceptCASN1Name", "ConceptScopeNote", "EntryTerms":[...]}]
    },
    {
        "DescriptorUI":,
        "DescriptorName":,
        "TreeNumberList": ,
        "Concepts": [{"ConceptName":, "ConceptCASN1Name", "ConceptScopeNote", "EntryTerms":[...],}]
    },
    ...
]

Mandatory fields:
    DescriptorUI
    DescriptorName
    TreeNumberList
    ConceptName
    EntryTerms

NOTE: Some concepts do not have "ConceptCASN1Name" or "ConceptScopeNote", thus these are not mandatory. """


import json
import argparse
import xmltodict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_mesh_xml_file",
                        type=str,
                        required=False,
                        help="Directory for the original xml file with the complete MeSH dictionary.")
    parser.add_argument("--input_mesh_json_file",
                        type=str,
                        required=True,
                        help="Directory for the json file with the complete MeSH dictionary.")
    parser.add_argument("--save_xml_to_json",
                        default=False,
                        action='store_true',
                        help="Whether to save a .json file with the content from the converted xml input file.")
    parser.add_argument("--output_mesh_json_file",
                        type=str,
                        required=True,
                        help="Directory for the json file with the filtered MeSH dictionary.") 
    parser.add_argument('--valid_mesh_tree_codes',
                        default=None,
                        nargs="*",
                        type=str,
                        help='Codes from MeSH TreeNumbers considered valid, to be used in the filtering of MeSH codes. Eg: D01 D02 D03 D04')
    args = parser.parse_args()

    if args.input_mesh_xml_file:
        with open(args.input_mesh_xml_file) as xmlFile:
            data = xmltodict.parse(xmlFile.read())
        data = json.dumps(data, indent=4, sort_keys=True)
        if args.save_xml_to_json:
            with open(args.input_mesh_json_file,"w") as jsonFile:
                jsonFile.write(data)

    with open(args.input_mesh_json_file,"r") as jsonFile:
        data=json.load(jsonFile)

    validTreeCodes = args.valid_mesh_tree_codes

    RecordsList = []

    for descriptor in data["DescriptorRecordSet"]["DescriptorRecord"]:
        DescriptorDict = {}
        DescriptorDict["DescriptorUI"]   = descriptor["DescriptorUI"]
        DescriptorDict["DescriptorName"] = descriptor["DescriptorName"]["String"]

        try:
            DescriptorDict["TreeNumberList"] = descriptor["TreeNumberList"]["TreeNumber"]
        except KeyError:
            DescriptorDict["TreeNumberList"] = ""

        validTreeCode = False
        if isinstance(DescriptorDict["TreeNumberList"], list):
            for treeCode in DescriptorDict["TreeNumberList"]:
                if treeCode.startswith(tuple(validTreeCodes)):
                    validTreeCode = True
        else:
            if DescriptorDict["TreeNumberList"].startswith(tuple(validTreeCodes)):
                validTreeCode = True

        if validTreeCode:
            conceptList=[]
            if isinstance(descriptor["ConceptList"]["Concept"], list):
                for concept in descriptor["ConceptList"]["Concept"]:
                    conceptDict = {}
                    entryTermsList = []
                    conceptDict["ConceptName"] = concept["ConceptName"]["String"]
                    if "CASN1Name" in concept.keys():
                        conceptDict["ConceptCASN1Name"] = concept["CASN1Name"]
                    if "ScopeNote" in concept.keys():
                        conceptDict["ConceptScopeNote"] = concept["ScopeNote"]

                    if isinstance(concept["TermList"]["Term"], list):
                        for term in concept["TermList"]["Term"]:
                            entryTermsList.append(term["String"])
                    else:
                        entryTermsList.append(concept["TermList"]["Term"]["String"])  
                    conceptDict["EntryTerms"] = entryTermsList
                    conceptList.append(conceptDict)
            else:
                conceptDict = {}
                entryTermsList = []
                concept = descriptor["ConceptList"]["Concept"]
                conceptDict["ConceptName"] = concept["ConceptName"]["String"]
                if "CASN1Name" in concept.keys():
                    conceptDict["ConceptCASN1Name"] = concept["CASN1Name"]
                if "ScopeNote" in concept.keys():
                    conceptDict["ConceptScopeNote"] = concept["ScopeNote"]

                if isinstance(concept["TermList"]["Term"], list):
                    for term in concept["TermList"]["Term"]:
                        entryTermsList.append(term["String"])
                else: 
                    entryTermsList.append(concept["TermList"]["Term"]["String"])
                conceptDict["EntryTerms"] = entryTermsList
                conceptList.append(conceptDict)

            DescriptorDict["Concepts"] = conceptList
            RecordsList.append(DescriptorDict)


    with open(args.output_mesh_json_file, "w") as jsonFile:
        json.dump(RecordsList, jsonFile, indent=4, sort_keys=True)

    
        
if __name__ == "__main__":
    main()
