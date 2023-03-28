# learn_gradio

https://huggingface.co/spaces/hot6/test_vit

## 메모

- transformers

  - ViTForImageClassification, ViTImageProcessor
    - google/vit-base-patch16-224

  ```python
    from transformers import ViTImageProcessor, ViTForImageClassification

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
  ```

- Dataset과 DataLoader

  - from torch.utils.data import Dataset, DataLoader
  - Dataset은 하나의 데이터에 대해 처리
  - DataLoader는 배치크기를 받고 각 데이터에 대해 Dataset 작업을 함

  ```python
   train_dataset = ImageDataset(train_paths, train_targets, shape=(224, 224))
   train_loader = create_data_loader(train_dataset, batch_size=64)
  ```

- pytorch gpu 이용

  - device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  - .to(device)

- gradio

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
# dict()
dict((i,i+2) for i in range(3)) # {0: 2, 1: 3, 2: 4}
```

## 참고

- [keras Dataset](https://www.kaggle.com/code/hengzheng/dog-breeds-classifier)

- [vit model](https://huggingface.co/google/vit-base-patch16-224)

- [Gradio](https://www.tanishq.ai/blog/gradio_hf_spaces_tutorial)

