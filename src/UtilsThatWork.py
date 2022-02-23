from elements import merge_collections
from corpora import NLMChemCorpus, NLMChemTestCorpus #, CDRCorpus, CHEMDNERCorpus

class Utils():
	def readFiles():
		return None,None

	def readAnnotations():
		return []

	def readMesh(meshFileLocation):
		import json
		f = open(meshFileLocation,)
		data = json.load(f)
		f.close()
		return data
		
	def buildIndentificationSubmission(indentifiedChemicals):
		print("Write file")
		
	def buildIndexingSubmission(indexedChemicals):
		print("Write file")

	def merge(dst):
		trainNLMCorpus 	= NLMChemCorpus()
		trainCollection = trainNLMCorpus["train"]
		devCollection   = trainNLMCorpus["dev"]
		trainCorpus 	= merge_collections(trainCollection, devCollection)
		with open(dst, "w") as file:
			newJson = trainCorpus.pretty_json()
			_ = file.write(newJson)