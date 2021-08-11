from corpora import NLMChemCorpus#, CDRCorpus, CHEMDNERCorpus
import json
import subprocess


import os
import tempfile


class Normalizer():
	def normalize(annotations, goldStandard, test=False):

		corpus = NLMChemCorpus()
		if test:
			corpus = corpus["test"]
		else:
			corpus = corpus["train"]
			# corpus = corpus["dev"]

		with open("/backup/data/MeSH2021/filteredMeSH2021_D01-D02-D03-D04.json","r") as file:
			meshJSON = json.load(file)

# PARA UM DICIONÁRIO BASEADO APENAS NO DESCRIPTOR NAME E ID
		# meshDict = dict()
		# for entry in meshJSON:
		# 	meshDict[entry["DescriptorName"]] = entry["DescriptorUI"]

# PARA UM DICIONÁRIO EXPANDIDO COM ENTRYTERMS
		meshDict = dict()
		for entry in meshJSON:
			meshDict[entry["DescriptorName"].lower()] = entry["DescriptorUI"]
			for concept in entry["Concepts"]:
				if "ConceptCASN1Name" in concept.keys():
					meshDict[concept["ConceptCASN1Name"].lower()] = entry["DescriptorUI"]
				for entryTerm in concept["EntryTerms"]:
					meshDict[entryTerm.lower()] = entry["DescriptorUI"]

		mapped=0
		unmapped=0

# SEM Ab3P a expandir abreviações
		# print(corpus)
		# for id, document in corpus:
		# 	for entity in document.entities():
		# 		print(document.text())
		# 		if entity.text in meshDict.keys():
		# 		# if entity.text.lower() in meshDict.keys():
		# 			mapped+=1
		# 		else:
		# 			unmapped+=1
		# 			# print(entity.text)
		# 		# print(entity.identifiers)
		# 		# print(entity.text)
		# print(mapped)
		# print(unmapped)


# # COM Ab3P a expandir abreviações, dicionário de abreviações ao nível do documento
# 		print(corpus)
# 		for id, document in corpus:
# 			fd, filePath = tempfile.mkstemp()
# 			try:
# 				with os.fdopen(fd, 'w') as tmpFile:
# 					tmpFile.write(document.text())
# 					abbreviationMap = dict()
# 					process = subprocess.Popen(["/backup/tools/NCBI/Ab3P/identify_abbr", filePath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
# 					stdout, stderr = process.communicate()
# 					ab3pOutput = stdout.split("\n")
# 					if len(ab3pOutput)>2:
# 						ab3pOutput = ab3pOutput[1:-1]
# 						for abbreviation in ab3pOutput:
# 							try:
# 								abbreviation, longForm, confidenceScore = abbreviation.strip().split("|")
# 								abbreviationMap[abbreviation.lower()] = longForm.lower()
# 							except ValueError:
# 								pass
#
# 					for entity in document.entities():
# 						# if entity.text in meshDict.keys():
# 						if entity.text.lower() in meshDict.keys():
# 							mapped+=1
# 						elif entity.text.lower() in abbreviationMap.keys():
# 							entity.text = abbreviationMap[entity.text.lower()]
# 							if entity.text in meshDict.keys():
# 								mapped+=1
# 							else:
# 								unmapped+=1
# 						else:
# 							unmapped+=1
# 			finally:
# 					os.remove(filePath)
#
# 		print(mapped)
# 		print(unmapped)

# COM Ab3P a expandir abreviações, dicionário de abreviações ao nível do corpus
		print(corpus)
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
			for entity in document.entities():
				# if entity.text in meshDict.keys():
				if entity.text.lower() in meshDict.keys():
					mapped+=1
				elif entity.text.lower() in abbreviationMap.keys():
					entity.text = abbreviationMap[entity.text.lower()]
					if entity.text in meshDict.keys():
						mapped+=1
					else:
						unmapped+=1
				else:
					unmapped+=1

		print(mapped)
		print(unmapped)









		return []