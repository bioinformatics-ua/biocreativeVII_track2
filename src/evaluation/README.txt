evaluate.py

This is the official evaluation script for the BioCreative 7 NLM Chem track

The script requires the following switches:

--reference_path / -r: This is the path to the file or set of files with the annotations that should be consiered 'correct.'
--prediction_path / -p: This is the path to the file or set of files with the annotations that will be evalauted
--evaluation_type / -t: This is the type of evaluation to perform, either 'span' or 'identifier.' Span evaluation compares the set of annotated spans (locations), ignoring the annotated identifiers. This is the typical evaluation for named entity recognition. Identifier evaluation compares the set of identifiers annotated, ignoring the span. This is the typical evaluation for normalization (entity linking) and for indexing.
--evaluation_method/-m: This specifies whether to perform a 'strict' or 'approx' evaluation. For span evaluation, strict requires that the start and end offsets match exactly, while approx only requires that they overlap. For identifier evaluation, strict compares the set of identifiers directly, while approx first augments the sets of identifiers with (a subset of) parent identifiers as described for LCAF evaluation [1, 2].
--annotation_type / -a: This is the type of the annotation, such as "Chemical"
--parents_filename: This is the name of the parents file to be used for approx identifier evaluation
--logging_level / -l: This is the logging level (optional). Options are {critical, error, warning, info, debug}, with the 'info' the default 
--no_document_verification: Turns off verification that the text of the reference and prediction documents match.

Examples:

To perform a strict span evaluation for Chemicals:

python evaluate.py --reference_path REFERENCE_PATH --prediction_path PREDICTION_PATH --evaluation_type span --evaluation_method strict --annotation_type Chemical

To perform an approx span evaluation for Chemicals:

python evaluate.py --reference_path REFERENCE_PATH --prediction_path PREDICTION_PATH --evaluation_type span --evaluation_method approx --annotation_type Chemical

To perform a strict identifier evaluation for Chemicals:

python evaluate.py --reference_path REFERENCE_PATH --prediction_path PREDICTION_PATH --evaluation_type identifier --evaluation_method strict --annotation_type Chemical

To perform a strict identifier evaluation for indexing: 

python evaluate.py --reference_path REFERENCE_PATH --prediction_path PREDICTION_PATH --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical

To perform an approx identifier evaluation for indexing: 

python evaluate.py --reference_path REFERENCE_PATH --prediction_path PREDICTION_PATH --evaluation_type identifier --evaluation_method approx --annotation_type MeSH_Indexing_Chemical

NOTE: the approx identifier evaluation is not completely deterministic; a variation of <1% run to run is expected. It may be run repeatedly to obtain an average.

CHANGELOG

v1:
	* Initial version
v2: 
	* Corrected a bug in approximate span matching that did not differentiate spans by document, causing the reported performance to be incorrectly high. 
	* Modfied span evaluation to filter zero-length annotations from either the reference or predicted sets. They are now ignored.
v3:
	* Corrected a bug in lca.py using the wrong filename for the parents file
	* Corrected a bug introduced in v2 that incorrectly filtered zero-length annotations when performing an identifier evaluation. Zero-length annotations are still filtered when performing a span evaluation, but a warning is now produced.
	* Added a check that the span of an annotation does not extend beyond the current passage. If so, it is ignored and a warning is produced.

REFERENCES
1. Tsatsaronis, G., Balikas, G., Malakasiotis, P. et al. An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition. BMC Bioinformatics 16, 138 (2015). https://doi.org/10.1186/s12859-015-0564-6
2. Kosmopoulos, Aris, et al. "Evaluation measures for hierarchical classification: a unified view and novel approaches." Data Mining and Knowledge Discovery 29.3 (2015): 820-865. https://arxiv.org/pdf/1306.6802