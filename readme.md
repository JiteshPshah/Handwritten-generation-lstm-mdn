
## Handwritten generation

The implementation was done using tensorflow-gpu library with tensorflow version 1.8 and python 2.7.

in case of python package dependency, please follow requirement.txt 


### Handwritten generation from text

* to generate handwritting from given string text from trained model, use following command:

`python handwritten_generation_from_text-test.py "model checkpoint directory" "text" "random seed" "file name to save strokes and text(in .pkl extension)"`


* for training of handwritten generation from text use following command:

`python handwritten_generation_from_text-train.py "model checkpoint directory to save checkpoints" "data_file_path" `

*model_checkpoint_dir=chkpts_handwritten_generation-from_text*

*data_path=data/training_data.pkl*

### Handwritten generation randomly

* to randomly generate handwriting from trained model, use following command:

`python handwritten_generation_randomly-test.py "model checkpoint directory" "timesteps" "random seed" "file name to save strokes(.npy)"`

* to training model for genrating random handwritting , use followong command:

`python handwritten_generation_randomly-train.py "model checkpoint directory to save checkpoints" "data_file_path"`

*model_checkpoint_dir=checkpoint_dir_random_handwritten_generation*

*data_path=data/training_data.pkl*

* to see demo and handwrtting plotting refer *demo.ipynb* file
