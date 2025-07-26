# INLP Assignment 3

### Download the data

Data can be downloaded from this [link.](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/sriharshitha_mareddy_students_iiit_ac_in/Enn1M7FNhUtKubWP0sjSz6oBIR_qOzVQ5RXb5gin2RKDtA?e=ifJLDb)

### Pre train

1. Modify the hyperparameters in the `ELMO.py` file.
2. Run the following command in your terminal:
    ```
    python3 ELMO.py 
    ```
3. pretrained elmo model will be stored in `bilstm.pt` and vocab will be stored in `word2idx.pt` and `idx2word.pt`.


## Classification Model

Pretrained models and embeddings can be downloaded from this [link.](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/sriharshitha_mareddy_students_iiit_ac_in/EiYluaeWO4ZCoJJ_I0uR860BORnVg4NVPHGOXbaK_pFj7A?e=a2HxjP)

### Using pretrained elmo model 
1. Download the pretrained model `bilstm.pt` and vocab `word2idx.pt`, `idx2word.pt` from the provided link.
2. Run the following command in your terminal:
    ```
    python3 classification.py <option>
    ```
    - Replace `<option>` with `-t` for trainable lambdas or `-f` for frozen lambdas or `-l` for learnable function.
3. The classifier model will be saved in `classifier_<option>.pt`
4. The results for losses and accuracies will be printed here.

## Load the classifier model

for best results load the model `classifier_l.pt` (i.e., the one considering the hyperparameter of learnable function)

