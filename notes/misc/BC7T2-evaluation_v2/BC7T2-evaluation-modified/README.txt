2021-07-13, 18:02

These scripts were slightly modified by Rui Antunes to work properly.
To use these, copy them to the original directory.

Original evaluation script (v2) can be found at:
https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-evaluation_v2.zip

What changed:

- evaluate.py:

  * The evaluation was not working for the annotations of type
    `MeSH_Indexing_Chemical` because annotations with an empty span were
    being discarded. Now these are considered.

  * Improved prints.

- lca.py:

  * In the method `load_parents()` the variable name `parents_filename`
    was changed to `filename`.
