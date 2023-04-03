# learn_transformers

https://huggingface.co/spaces/hot6/vit_base-dog_breeds

---

# transformers

## model, extractor

- .from_pretrained

  - extractor에서 return_tensors='pt'를 인자로 넘겨야 텐서 출력
  - model에서 num_label, id2label, label2id인자로 넘겨야 나중에 model.config. 에서 사용가능

## datasets

- load_dataset("imagefolder", data_dir="data/", split="train") split의 경우 원래 데이터구조가 어떤지에 따라 사용

  ->

  - Dataset
  - DatasetDict

- shared

- train_test_split

- with_transform(fn)

### labels

- dataset['train'].features['label']
  - names

## TrainingArguments

## Trainer

- TrainerCallbacks
- .train()

## Auto

Trainer를 통해 save_model()하고

AutoFeatureExtractor, AutoModelForImageClassification에서 from_pretrained로

huggingface나 local경로를 전달하면

아마 config.json, preprocessor_config.json, pytorch_mode.bin등을 통해 객체(model, extractor) 생성

- predict

  model(extractor(INPUT)) -> classification의 경우 출력층은 softmax해줘야함

# gradio

```python
 import gradio as gr

 image = gr.inputs.Image(shape=(W,H)) # inputs
 label = gr.outputs.Label() # outputs
 examples = ['ex1.jpeg','ex2.jpeg'] # examples

 intf = gr.Interface(fn=predict, inputs=image, outputs=label, examples=examples)
 intf.launch(inline=False)
```

이때 fn이 리턴하는 형태는 dict( 예측 : 확률 )

```python
TOP_NUM = 5

def predict(image):  # classify_dog_breed(image):
    inputs = extractor(images=image, return_tensors="pt")['pixel_values']
    outputs = model(inputs)
    probs, preds = torch.topk(F.softmax(outputs['logits'].data, dim=-1), TOP_NUM)
    return dict((model.config.id2label[pred].split('-')[-1], prob) for pred, prob in zip(preds.tolist()[0], probs.tolist()[0]))
```
