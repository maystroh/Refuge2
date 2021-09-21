# Refuge2
Solution to be proposed for REFUGE 2 challenge





Segmentation.ipynb


We use this file do two time training, one for disc, another for cup. After prediction, we merge disc and cup.

we use binary focal dice loss.

Training input:

                ./Training/image for disc

                ./Training/image for cup
                
                ./Training/mask_disc
                
                ./Training/mask_cup
                
Validation input: 

                ./Validation/image for disc

                ./Validation/image for cup
                
                ./Validation/mask_disc
                
                ./Validation/mask_cup
                

multiSeg.ipynb

# Copyright
Copyright © 2018-2019 [LaTIM U1101 Inserm](http://latim.univ-brest.fr/)

This file we try to segment disc and cup at the same time, input is two masks, one only disc and another is cup.

We try to calculate binary crossentry and CDR as loss, but not yet finish.


Training input: 

                ./Training/image for disc

                ./Training/mask_disc
                
                ./Training/mask_cup
                
Validation input: 

                ./Validation/image for disc

                ./Validation/mask_disc
                
                ./Validation/mask_cup
