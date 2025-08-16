**Comprehensive assessment of BERT-based methods for predicting antimicrobial peptides**
```
│  README.md
│
├─dataset # Evaluate experimental datasets
│      independent dataset_AMPs.fasta
│      independent dataset_nonAMPs.fasta
│      ├─ADAPTABLE database
│           adaptable_amps.fa
│           adaptable_nonamps.fa
│      ├─APD database
│           apd_amps.fa
│           apd_nonamps.fa
│      ├─CAMP database
│           camp_amps.fa
│           camp_ nonamps.fa
│      ├─dbAMP database
│           dbamp_amps.fa
│           dbamp_ nonamps.fa
│      ├─DRAMP database
│           dramp_amps.fa
│           dramp_ nonamps.fa
│      ├─YADAMP database
│           yadamp_amps.fa
│           yadamp_nonamps.fa
│
└─Utils # Utility scripts
        test_in_AMP-BERT.ipynb # Testing on AMP-BERT
        test_in_Bert-Protein.ipynb # Testing on Bert-Protein
        test_in_cAMPs_pred.py # Testing on cAMPs_pred
        test_in_LM_pred.ipynb # Testing on LM_pred
```
# Methods for assessments
The environments used in this study are available on \dependencies.<br>
| |Pretraining|Parameter|Classification|Repository|
|------------------|----------------------|--------------------|-----------------|------------------|
|**Bert-Protein**|UniProt|12 layers<br>12 heads|FFN|https://github.com/BioSequenceAnalysis/Bert-Protein|
|**AMP-BERT**|BFD|30 layers<br>16 heads|FCN|https://github.com/GIST-CSBL/AMP-BERT|
|**LM_pred**|BFD100 <br>UniRef100|30 layers<br>16 heads|CNN|https://github.com/williamdee1/LMPred_AMP_Prediction|
|**cAMPs_pred**|BookCorpus<br>Wikipedia|12 layers<br>12 heads|FFN|https://github.com/mayuefine/c_AMPs-prediction|
# Dataset
* **ADAPTABLE**<br>

    *  AMP sequence data were downloaded from http://gec.u-picardie.fr/adaptable<br>
    
* **AMPfun**<br>

    *  AMP sequence data were downloaded from http://fdblab.csie.ncu.edu.tw/AMPfun/index.html<br>
     
* **APD**<br>

    *  AMP sequence data were downloaded from http://aps.unmc.edu/AP/<br>
     
* **CAMP**<br>

    *  AMP sequence data were downloaded from http://www.bicnirrh.res.in/antimicrobial<br>
     
* **dbAMP**<br>

    *  AMP sequence data were downloaded from http://csb.cse.yzu.edu.tw/dbAMP/<br>
     
* **DRAMP**<br>

    *  AMP sequence data were downloaded from http://dramp.cpu-bioinfor.org/<br>
     
* **YADAMP**<br>

    *  AMP sequence data were downloaded from http://www.yadamp.unisa.it<br>
     
Peptide sequences were downloaded from UniProt http://www.uniprot.org<br>

# Experimental instructions
## The architecture of evaluation experiments
In the study, we employed the strategy of independent test, multiple database validation and 5-fold cross-validation to evaluate the predictive performance of these methods. 
![image](https://github.com/WanlingGao/AMPpred-BERT-assessment/blob/main/img/The%20architecture%20of%20evaluation%20experiments.PNG)
## Performance evaluation on the independent dataset and validation datasets
We collected an independent test dataset based on multiple different AMP databases, and then compared and analyzed the prediction performance of different prediction tools on the independent test dataset. In order to compare the robustness and generalization ability of the model, we further tested the prediction performance of different tools on multiple validation datasets.
### Collection of the independent test dataset 
Positive samples for independent test set were collected from different comprehensive AMP databases, including APD, CAMP, dbAMP, DRAMP, YADAMP, ADAPTABLE and AMPfun. Negative samples were collected from UniProt.
![image](https://github.com/WanlingGao/AMPpred-BERT-assessment/blob/main/img/Collection%20process%20for%20the%20independent%20test%20dataset.PNG)

### Testing on Bert-Protein
```python
# ljy_predict_AMP.py
# The test data set is tested on the trained model

if __name__ == '__main__':
    main(data_name=r"test.csv",
         out_file="result.txt",
         model_path="model_train/1kmer_model/model.ckpt",
         step=1,
         config_file="./bert_config_1.json",
         vocab_file="./vocab/vocab_1kmer.txt")
```

### Testing on AMP-BERT

```python
# test_ with_amps.ipynb
# load appropriate tokenizer and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained("Train/")

# Input test dataset
with open('test.csv') as l:
# Output prediction result
  with open('pred.txt','w') as r:
# Output predicted probability results
     with open('prob.txt','w') as f:
```

### Testing on LM_pred
```python
# Testing_Models.ipynb
# Input: test dataset
# Output: prediction result with label
#         prediction result with probability

BERT_model_INDEP = keras.models.load_model('Train model/BERT-BFD_best_model.epoch03-loss0.19.hdf5')
X_test = load_INDEP_X_data('BERT_BFD')
BERT_mod_pred = BERT_model_INDEP.predict(X_test, batch_size=8)
file = open('prob.txt', 'a')

for i in range(len(BERT_mod_pred)):
    mid = str(BERT_mod_pred[i]).replace('[', '').replace(']', '')
    mid = mid.replace("'", '').replace(',', '') + '\n'
    file.write(mid)
file.close()

BERT_mod_pred_labels = convert_preds(BERT_mod_pred)
file = open('pred.txt', 'a')
for i in range(len(BERT_mod_pred_labels)):
    mid = str(BERT_mod_pred_labels[i]).replace('[', '').replace(']', '')
    mid = mid.replace("'", '').replace(',', '') + '\n'
    file.write(mid)
file.close()

BERT_metrics = display_conf_matrix(y_test_INDEP, BERT_mod_pred_labels, BERT_mod_pred, 'BERT Model', 'BERT-BFD_Model_CM.png')
```

### Testing on cAMPs_pred
```python
environ["CUDA_VISIBLE_DEVICES"] = "0"
from bert_sklearn import load_model
import pandas as pd
x_test=[]
with open(r'test.csv')  as l:
    lines=l.readlines()
    for line in lines:
        line=line.rsplit()
        x_test.append(line)
model = load_model("bert.bin")
pre=model.predict(x_test)
prod=model.predict_proba(x_test)[:,1]
pre=pd.DataFrame(pre)
prod=pd.DataFrame(prod)
prod.to_csv('prod.txt')
pre.to_csv('pred.txt')
```

## Performance evaluation on the retraining dataset
We retrained representative BERT-based models for AMP prediction on the comprehensive dataset and assessed their performance using the five-fold cross-validation test. 
### Retraining and testing on Bert-Protein
* **Date example**:<br>
```
    Train  1  M I S D S G ...
```
* **run_fine_tune.sh**

```python
# Set the Gpus that can be used
export CUDA_VISIBLE_DEVICES=0
python ljy_run_classifier.py \
--do_eval True \
--do_save_model True \
--data_name  AMPScan\
--batch_size 16 \
--num_train_epochs 1 \
--warmup_proportion 0.1 \
--learning_rate 2e-5 \
--using_tpu False \
--seq_length 128 \
--data_root ./dataset/1kmer_tfrecord/AMPScan/ \
--vocab_file ./vocab/vocab_1kmer.txt \
--init_checkpoint ./model/1kmer_model/model.ckpt \
--bert_config ./bert_config_1.json \
--save_path ./model_train/1kmer_model/model.ckpt
```

### Retraining and testing on AMP-BERT

* **Data example**<br>
```
AMP--1,FQPYDHPAEVSY,12,TRUE
```

* **fine-tune_with_amps.ipynb**
```python
# Fine tuning
# Training set
df = pd.read_csv('train.csv', index_col = 0)
df = df.sample(frac=1, random_state = 0)
print(df.head(7))
train_dataset = amp_data(df)

# Validation set
df_val = pd.read_csv('val.csv', index_col = 0)
df_val= df_val.sample(frac=1, random_state = 0)
val_dataset = amp_data(df_val)

# Save model
trainer.train()
trainer.save_model('Train/')
```
### Retraining and testing on LM_pred

* **Data example**<br>
```
25,TFFRLFNRGGGWGSFFKKAAHVGKL,AMP--955
```
* **Model Training**
```python
BERT_filepath = 'Keras_Models/BERT_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
BERT_Plots_Path = 'Training_Plots/INDEP/BERT_Best_Model_Plot.png'
train_model(X_train, y_train_res, X_val, y_val_res, BERT_filepath, BERT_Plots_Path, 30, 8, False, 320, 11, 'RandomNormal', 8, 0.0001, 'SGD')
```
### Retraining and testing on cAMPs_pred

* **Installation bert_sklearn**<br>

   Download and copy bert_sklearn to your python3 site-packages folder<br>
```python
   cd bert-sklearn
   pip install .
```
* **Data example**<br>
```
  >AMP-467
  KNLRRIIRKIAHIIKKYG
```
* **train_in_cAMPs_pred.py**
```python
from bert_sklearn import BertClassifier
model = BertClassifier()

model.train_batch_size=16
model.eval_batch_size=16
model.learning_rate=2e-5
model.epochs=10

model.fit(x_train, y_train)
print(model.score(x_test,y_test))
model.save('bert.bin')
```

# Contact
Please feel free to contact us if you need any help: zhenyuyue@ahau.edu.cn
