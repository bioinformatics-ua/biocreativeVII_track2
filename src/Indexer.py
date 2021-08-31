from corpora import NLMChemCorpus, NLMChemTestCorpus#, CDRCorpus, CHEMDNERCorpus

PERCENTAGE = 1
MIN_OCCUR_CAPTIONS = 0.2 * PERCENTAGE
MIN_OCCUR_ABSTRACT = 0.07 * PERCENTAGE
MIN_OCCUR_TITLE = 0.1 * PERCENTAGE

#evaluator
#python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-train.BioC.json --prediction_path ./nlm_index_chem_train_bioc.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical

class Indexer():
	def index(mesh, test=False):
		if True: #test:
			corpus = NLMChemTestCorpus()
			corpus = corpus["test"]
		#else:
		#	corpus = NLMChemCorpus()
		#	corpus = corpus["train"]
		#	print(corpus)
		#	mesh = Indexer.readGoldStandard(corpus)

		corpusWithScores = Indexer.getScores(corpus, mesh, Indexer.getDicts(corpus))

		with open("Track2-Team-110-Subtask1-Indexing-Run-3.json", "w") as file:
			newJson = corpusWithScores.pretty_json()
			_ = file.write(newJson)

	def getScores(corpus, meshes, dicts):
		sections = dicts["sections"]
		captions = dicts["captions"]
		abstracts = dicts["abstracts"]

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

		for id in meshes:
			meshesToAdd = set()
			for x in meshes[id]:
				mesh = x[0][0]
				span = x[1]
				spanStart = span[0]
				spanEnd = span[1]
				if mesh == "-":
					continue

				#Check if mesh is a section title
				for sec in sections[id]:
					if spanStart > sections[id][sec][0] and spanEnd < sections[id][sec][1]:
						#mesh in title
						if countMeshes[id][mesh]/countAllMeshesByDoc[id] > MIN_OCCUR_TITLE:
							meshesToAdd.add(mesh)

				#Check if mesh is in a caption
				for sec in captions[id]:
					if spanStart > captions[id][sec][0] and spanEnd < captions[id][sec][1]:
						#mesh in caption
						if countMeshes[id][mesh]/countAllMeshesByDoc[id] > MIN_OCCUR_CAPTIONS:
							meshesToAdd.add(mesh)

				#Check if is in abstract
				for sec in abstracts[id]:
					if spanStart > abstracts[id][sec][0] and spanEnd < abstracts[id][sec][1]:
						if countMeshes[id][mesh]/countAllMeshesByDoc[id] > MIN_OCCUR_ABSTRACT:
							meshesToAdd.add(mesh)


			corpus[id].set_mesh_indexing_identifiers(list(meshesToAdd))
		return corpus

	def getDicts(corpus):
		dicts = {}
		dicts["sections"] = {} #docID:[section:startSpan]
		dicts["captions"] = {} #docID:[captions:span tuple]
		dicts["abstracts"] = {}#docID:[captions:span tuple]

		for id, document in corpus:
			dicts["sections"][id] = {}
			dicts["captions"][id] = {}
			dicts["abstracts"][id] = {}

			for passage in document:
				if "fig_" in passage.typ or "table_" in passage.typ:
					dicts["captions"][id][passage.text] = passage.span
				elif "title" in passage.typ:
					dicts["sections"][id][passage.text] = passage.span

				if "abstract" in passage.typ:
					dicts["abstracts"][id][passage.text] = passage.span

		return dicts

	def readGoldStandard(corpus):
		meshes = {}
		for id, document in corpus:
			meshes[id] = []
			for passage in document:
				for ide in passage.nes:
					for mesh in ide.identifiers:
						meshes[id].append([[mesh],ide.span])
		return meshes



