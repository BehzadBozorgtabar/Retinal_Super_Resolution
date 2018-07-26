config.py    - Configuration file with all the data paths and training/testing settings
layer_def.py - definitions of all the layers
SuperRes.py  - SRGAN Model implementation
main_SR.py   - main file to run training/testing of the model

For training:
Set training data path to "IMAGES" in config.py  and execute "python main_SR.py"
Checkpoints will be saved to the path given in "CHECKPOINT" in config.py file.

For testing:
Change the "test path" in line 66 of main_SR.py to the
required testing data path and execute "python main_SR.py".

If there are matching checkpoint files to the "NUM_TRAIN_EPOCHS" inside the
"CHECKPOINT" path, the model will use them and run the predictions on the 
test data.
Otherwise, it will train the model first and then do the predictions.
