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
		print(corpus)

		with open("/backup/data/MeSH2021/filteredMeSH2021_D01-D02-D03-D04.json","r") as file:
			meshJSON = json.load(file)

		# # PARA UM DICIONÁRIO BASEADO APENAS NO DESCRIPTOR NAME E ID
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


		ab3pAbbreviationExpansion = True
		ab3pExpansionDictLevels = ["Document", "Corpus"]
		ab3pDictLevel = ab3pExpansionDictLevels[1]

		mapped=0
		unmapped=0
		mappedDocuments = dict()

		# SEM Ab3P a expandir abreviações
		if not ab3pAbbreviationExpansion:
			for id, document in corpus:
				meshSet = set()
				for passage in document.pol:
					for entity in passage.es:
						print(entity.text, entity.identifiers)
						#if entity.text in meshDict.keys():
						if entity.text in meshDict.keys():
							meshCode = "MESH:" + meshDict[entity.text]
							meshSet.add(meshCode)
							entity.identifiers = {meshCode}
							entity.identifiers_str = ','.join(sorted(entity.identifiers))
							mapped+=1
						else:
							meshSet.add("-")
							entity.identifiers = {"-"}
							entity.identifiers_str = ','.join(sorted(entity.identifiers))
							unmapped+=1
						# print(entity.text)
					# print(entity.identifiers)
					# print(entity.text)

				mappedDocuments[id] = list(meshSet)
			print(mapped)
			print(unmapped)

		elif ab3pAbbreviationExpansion:
			if ab3pDictLevel == "Document":
				# COM Ab3P a expandir abreviações, dicionário de abreviações ao nível do documento
				for id, document in corpus:
					meshSet = set()
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

							for passage in document.pol:
								for entity in passage.es:
									# if entity.text in meshDict.keys():
									if entity.text.lower() in meshDict.keys():
										meshCode = "MESH:" + meshDict[entity.text.lower()]
										meshSet.add(meshCode)
										entity.identifiers = {meshCode}
										entity.identifiers_str = ','.join(sorted(entity.identifiers))
										mapped+=1
									elif entity.text.lower() in abbreviationMap.keys():
										text = abbreviationMap[entity.text.lower()]
										if text in meshDict.keys():
											meshCode = "MESH:" + meshDict[text]
											meshSet.add(meshCode)
											entity.identifiers = {meshCode}
											entity.identifiers_str = ','.join(sorted(entity.identifiers))
											mapped+=1
										else:
											meshSet.add("-")
											entity.identifiers = {"-"}
											entity.identifiers_str = ','.join(sorted(entity.identifiers))
											unmapped+=1
									else:
										meshSet.add("-")
										entity.identifiers = {"-"}
										entity.identifiers_str = ','.join(sorted(entity.identifiers))
										unmapped+=1
					finally:
							os.remove(filePath)

					mappedDocuments[id] = list(meshSet)
				print(mapped)
				print(unmapped)

			elif ab3pDictLevel == "Corpus":
				# COM Ab3P a expandir abreviações, dicionário de abreviações ao nível do corpus
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
					meshSet = set()
					for passage in document.pol:
						for entity in passage.es:
							# if entity.text in meshDict.keys():
							if entity.text.lower() in meshDict.keys():
								meshCode = "MESH:" + meshDict[entity.text.lower()]
								meshSet.add(meshCode)
								entity.identifiers = {meshCode}
								entity.identifiers_str = ','.join(sorted(entity.identifiers))
								mapped+=1
							elif entity.text.lower() in abbreviationMap.keys():
								text = abbreviationMap[entity.text.lower()]
								if text in meshDict.keys():
									meshCode = "MESH:" + meshDict[text]
									meshSet.add(meshCode)
									entity.identifiers = {meshCode}
									entity.identifiers_str = ','.join(sorted(entity.identifiers))
									mapped+=1
								else:
									meshSet.add("-")
									entity.identifiers = {"-"}
									entity.identifiers_str = ','.join(sorted(entity.identifiers))
									unmapped+=1
							else:
								meshSet.add("-")
								entity.identifiers = {"-"}
								entity.identifiers_str = ','.join(sorted(entity.identifiers))
								unmapped+=1

					mappedDocuments[id] = list(meshSet)
				print(mapped)
				print(unmapped)



		# print(mappedDocuments)
		#
		# with open("teste.json", "w") as file:
		# 	json.dump(mappedDocuments, file, indent=4)

		# for id, document in corpus:
		# 	for entity in document.entities():
		# 		print(entity.text, entity.identifiers)

		with open("nlm_chem_train_bioc.json", "w") as file:
			# newJson = corpus.json()
			# json.dump(newJson, file, indent=4)
			newJson = corpus.pretty_json()
			_ = file.write(newJson)





		return []

#python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-train.BioC.json --prediction_path ./nlm_chem_train_bioc.json --evaluation_type identifier --evaluation_method strict --annotation_type Chemical