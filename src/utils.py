import json
import requests
import os

def download_from_PMC(pmcid):
    
    URL = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    
    response = requests.get(URL)
    
    data = json.loads(response.text)
    
    path = os.path.join("datasets","PMC",f"BioC_{pmcid}.json")
    
    with open(path, "w") as f:
        json.dump(data, f)
    
    return path
    
    

class Utils():
    def readFiles():
        return None,None
    
    def readAnnotations():
        return []
    
    def buildIndentificationSubmission(indentifiedChemicals):
        print("Write file")
    
    def buildIndexingSubmission(indexedChemicals):
        print("Write file")
        
        
        
