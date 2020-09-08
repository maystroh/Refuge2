# Refuge2
Solution to be proposed for REFUGE 2 challenge





Segmentation.ipynb


We use this file do two time training, one for disc, another for cup. After prediction, we merge disc and cup.

we use binary focal dice loss.




multiSeg.ipynb

This file we try to segment disc and cup at the same time, input is two masks, one only disc and another is cup.

We try to calculate binary crossentry and CDR as loss, but not yet finish.
