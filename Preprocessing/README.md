In the project three sources of data were used, that had to be preprocssed differently:
- ProThermDB
- Data published by Leuenberger et al.
- Meltome Atlas

Running order:
  1. The preprocessing script (either `.../Preprocessing.py` or `.../PreProcessing.ipynb`).
  2. ProThermDB has to be clustered. Can be ignored for the other datasets (See thesis appendix for command).
  3. Generate the MSAs by running `.../MSA/allMSAgen.py` (Recommended to do in nohup overnight).
  4. Generated embeddings and condensed row attentions `MSA Transformer/MSA_to_embed.py`
