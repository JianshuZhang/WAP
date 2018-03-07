# Description
* Train WAP-VGG without using weightnoise and save the best model in terms of WER

      $ bash train.sh
	
* Anneal the best model by using weightnoise and save the new best model

      $ bash train_weightnoise.sh
	
* Reload the new best model and generate the testing latex strings

      $ bash test.sh
