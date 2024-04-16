# README

Thanks for your attention. The following instructions can help you reproduce the experiments.

## Platform

Our experiments are conducted on a platform with single GPU NVIDIA 3090 RTX 24GB.


## Running

```
cd code
bash train.sh
```

The detailed configurations can be found in the ```train.sh```. As the Bert model is too large, you can download the Bert model from [Hugging Face(```bert-tiny```)](https://huggingface.co/prajjwal1/bert-tiny).

## Files Definition

- ```data``` : contains three public datasets: BANNKING77, HWU64, Liu57 and Clinc150

- ```code``` : contains python files of our framework

    - ```data_loader``` : used to sample each episode's data
    - ```encoder``` : model file
    - ```losses_new.py``` : contains loss function
    - ```parser_util.py``` : parse parameters
    - ```train.py``` : train the model
    - ```train.sh``` : parameters used to train models
