# Stance-Conditioned-Modeling-for-Rumor-Verification
This repository contains code for our paper Stance-Conditioned Modeling for Rumor Verification (RV) (accepted at eKNOW 2015). The project used three publicly available rumor verification datasets. SemEval-2017, RumorEval-2019, and PHEME. 
--BiLSTM_LLM.py 
  is the driver code for SemEval-2017 and  RumorEval-2019 tasks
--BiLSTM_LLM_pheme.py
  is the driver code for PHEME.
  
--Post_Representation.py 
    does the post aggregation feature. 
--ReadInputStancDistr.py
    reads SemEval-2017 and  RumorEval-2019 data. 
--ReadInputPheme.py 
    reads PHEME data.  
--Stanc_Train.py and Stanc_Predictor.py 
    predict stance labels for PHEME dataset. 
--Text_preprocessing.py and Utilities.py 
    provide utility functions. 
--Test_BiLSTM_LLM.py 
    tests the model saved by the driver code.
