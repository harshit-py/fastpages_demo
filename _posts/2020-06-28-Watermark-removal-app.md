---
toc: true
layout: post
description: A complete tutorial of building an ML application that helps you remove watermarks.
categories: [markdown]
title: Going from idea to a Data driven Application
---

# Intro

I had recently completed a couple of online courses on Deep Learning including Practical Deep Learning for Coders by FastAI V3 and [TensorFlow in Practice Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/EHSHBQWMBH8R) from DeepLearning.ai and wanted to try out a project with a real world application and not classifiying digits or cats and dogs. So, one of the things that piqued my interest was using U-Net with pre-trained encoder to basically remove artifacts from photos. This was discussed in Lesson 7 part 1 of FAST AI course. The idea was to use a Non GAN approach to this problem. The idea could be used to either remove physical photo artifacts from digital scans or removing watermark from photos.
So in this article I'll take you through the complete ML pipeline; starting from idea, collecting data, training model and serving it. We will train an ML model to remove watermark from images. The complete pipeline is in Python with ML models trained in Fast AI (built on PyTorch) and served using Flask.
Also, throughout the article I'll add a challenges tab to list out challenges in each step.

# Setting up project
If you'd like to follow along, I recommend using [Cookiecutter for data science](https://drivendata.github.io/cookiecutter-data-science/). This project structure helps you in maintaining all the data versions, temporary outputs, analyses notebooks etc. in a concise manner and could also help you in version controlling.
In your venv or conda environment do

```bash
pip install cookiecutter
```

and then 
```bash
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```
This will set up a directory structure that will help you in almost all Data Science/Machine Learning projects. For more info, visit their documentation. Our data structure looks like this (after removing unecessary folders)

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
│
├── data
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   ├── web deploy	   <- Flask app detailed below
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```


# Getting data

Preparing training data for any ML based product is actually the hardest part in ML pipelines. If it's a classification problem getting labelled data is one of the foremost challenges. Some workarounds in these cases are using pre-trained models and use transfer learning. Luckily, in our case getting/preparing data is actually an easy process. For our model, the input is going to be an image with a watermark overlayed and target is going to be the same image without any watermark. This can be simply acheived by adding watermarks on any publicly available image dataset. However, you'll need to pay attention to the kind of dataset you choose. If you intend to build a watermark removal for documents, the training dataset needs to be made up of scanned documents and you can find a lot many publicaly available datasets. Here we're going to use [Oxford IIIT Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) that has a lot many use cases including foreground-background segmentation truth, boundingboxes for pet detection etc. But we'll primarily use this to create our own dataset of watermarked images.

## Watermarking Images

Selecting a watermark is as crucial as selecting the dataset. You can chose to add random texts in different colours all over your image or you can chose to add alpha channel watermarks that keep alpha trasparency in the final watermarked images. It depends on the final application but keep in mind the final trained model will only be able to remove artifacts that it has seen during training or that closely resembles watermarks from the training data.
To give an example of how to add an alpha channel watermark to an image, select an image with an alpha channel transparency (PNG), like [this](https://favpng.com/png_view/symbol-copyright-symbol-registered-trademark-symbol-png/9ttjUKMB), and add it over on alpha channel on your RGB image (which by default doesn't have any alpha channel).

Import image and watermark
```python
import PIL
wm = PIL.Image.open('path/to/watermark')
img = PIL.Image.open('path/to/original/image')
```

create a mask for  watermark image
```python
mask = wm.convert('L').point(lambda x: min(x, 25))
wm.putalpha(mask)
```
Resize the watermark according to the original input and loop over the original dimensions, skipping over the size of watermark image, and overlay the watermark image over the original image and save it to disk

```python
wm.thumbnail((new_mark_width, new_mark_width / self.aspect_ratio), PIL.Image.ANTIALIAS)
for a in range(0, img.size[0], wm.size[0]):
	for b in range(0, img.size[1], wm.size[1]):
		img.paste(wm, (a, b), wm)
		img.thumbnail((8000,  8000), PIL.Image.ANTIALIAS)
img.save('path/to/destination')
```

The output and input looks like this:

![original]({{ site.baseurl }}/images/original_input.jpeg "Input Image")

![watermark added]({{ site.baseurl }}/images/original_watermarked.jpg "Input Image Watermarked")

Now the Oxford IIIT Pets Dataset has a total of 7349 images and looping over all the images on disk to add watermark is going to take a lot of time and memory. But luckily for us Fast.ai has a utility available that takes help of Process based parallelism to take advantage of multiple processors in CPUs, by side-stepping the infamous Python GIL. `parallel` in fast.ai core makes the full use of multiple processors to effectively speed up the task. 
Here I wrap up the watermark logic into a callable object which is then called with the FastAI's Image ItemList of Oxford Images to iteratively, load each image, add watermark and save watermarked image to disk.

```python
class AddWatermark:
	def  __init__(self, path_lr, path_hr):
		self.path_lr = path_lr
		self.path_hr = path_hr
		self.wm = PIL.Image.open('path/to/watermark')
		mask = self.wm.convert('L').point(lambda x:  min(x,  25))
		self.wm.putalpha(mask)
		self.mark_width,  self.mark_height = self.wm.size
		self.aspect_ratio = self.mark_width / self.mark_height
		
	def  __call__(self, fn, i):
		dest = self.path_lr/fn.relative_to(self.path_hr)
		dest.parent.mkdir(parents=True, exist_ok=True)
		img = PIL.Image.open(fn)
		targ_sz = resize_to(img,  128, use_min=True)
		img = img.resize(targ_sz,resample=PIL.Image.BILINEAR).convert('RGB')
		main_width, main_height = img.size
		new_mark_width = main_width * 0.25
		self.wm.thumbnail((new_mark_width, new_mark_width / self.aspect_ratio), PIL.Image.ANTIALIAS)
		tmp_img = PIL.Image.new('RGB', img.size)
		for a in  range(0, tmp_img.size[0],  self.wm.size[0]):
			for b in  range(0, tmp_img.size[1],  self.wm.size[1]):
				img.paste(self.wm,  (a, b),  self.wm)
				img.thumbnail((8000,  8000), PIL.Image.ANTIALIAS)
		img.save(dest)
```

Now call this object with `parallel` from FastAI
```python
parallel(AddWatermark('path/to/input', 'path/to/output'), il.items)
```
This function then parallelizes the process and quickly adds watermark to all the images in dataset.
Now the input for our model becomes the output image from the previous function (watermark added images) and the target image is the original un-watermarked image.

{% include alert.html text="A lot of images will not be 3 channel RGB images, some would be either only two channels or corrupt; try to keep Image open and resize/convert call in a try-except block." %}



# Creating the Model
We are going to use U-Net with pre-trained Resnet34 as the encoder using `unet_learner` from FastAI. The idea of this tutorial is to create a working ML application from ideation to production and therefore we'll skip over a lot of details of implementing a [U-Net](https://arxiv.org/abs/1505.04597) (I'll take it up in next article, where we'll  implement and train a U-Net from scratch in PyTorch). We'll implement a flavour of U-Net in [FastAI](https://docs.fast.ai/vision.models.unet.html) which allows us to create a U-Net from any backbone trained on ImageNet. We'll use FastAI's DataBlock API to create DataLoaders for the FastAI model, specifically `ImageList.from_folder` which is then fed to `unet_learner` method which gives back a `Learner` object. 

```python
arch = models.resnet34 # encoder
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)
```

Define the Image Transform Pipeline, `zoom` and Normalize with `imagenet_stats`. Keep in mind to keep `tfm_y` as `True` or you'll have a mismatch in input and output data fed to the model, with transforms only applied to input data.

```python
def  get_data(bs,size):
	data = (src.label_from_func(lambda x: path_hr/x.name)
	.transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
	.databunch(bs=bs).normalize(imagenet_stats, do_y=True))
	data.c = 3
	return data

data_gen = get_data(32,128)
```

Finally, initialize the `Learner` model with default parameters

```python
learn_gen = unet_learner(data_gen, arch, wd=1e-3, blur=True, norm_type=NormType.Weight,self_attention=True, y_range=(-3.,3.), loss_func=MSELossFlat())
```

Note that `MSELossFlat` is similar to PyTorch's `nn.MSELoss` but flattens input and target. So, that gives us `Learner` object to start model training.


{% include info.html text="I have created the model with input size fixed at (128, 128), if you have better resources available try an even bigger size 256 or even 512." %}

## Training the model

I trained the model on Google Colab using GPU; I recommend that, the only downside is you might have to juggle between your local machine and Colab to transfer saved model weights etc., unless ofcourse you have NVIDIA Titan or similar ;) 

Training models in FastAI is fairly simple and follows the steps:
1. Find the optimal LR using `learn.lr_find` which is one of the better ways, if not the best way, of finding LR using [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186). Plot it using 'learn.recorder.plot()` to decide on an LR (1e-3 works most of the time).
2. Use `learn.fit_one_cycle()`, to run the model with frozen pre-trained weights for few epochs. This helps only the later layers of the model to have their parameters updated and let's you validate if the model is actually learning.
3. Unfreeze all the layers, and use `learn.fit_one_cycle()` again, this time with reduced LR.

FastAI basically wraps a lot of best practices for training the model under-the-hood, and let's you train the model with a very reasonable `val loss` for a total of about 5 epochs (2 epochs with frozen weights for pre-trained model and 3 epochs unfrozen). For me, the `train loss` and `val loss` was down to 0.045 and 0.043 respectively. One of the reason, the loss was low with very few epochs is probably due to the fact that the copyright watermark I've added does not add a lot of noise. Again, there's a lot of work behind the scenes that FastAI does but our objective for this article is to go from idea to production.

![Model train]({{ site.baseurl }}/images/train.png)

Once you have the model ready save it to disk, copy it to your local machine in case you trained it on colab. 
```python
learn_gen.export('wm_model')
```


{% include info.html text="Note that `learn_gen.save` won't work in case you want to run inference on your local machine but use `learn_gen.export` instead." %}


## Checking model performance

Once you are done training, it's time to check how the model is performing on new images. We'll first load the learner object from `fastai` and put the model in `eval` mode:

```python
learner = load_learner(path='path', file='wm_remove.pkl')
learner.model.eval()
```
Load the image, preprocess it as per model's input and score it through the model

```python
img = PIL.Image.open('path/to/image.jpg')
img = np.asarray(img.resize((128,128), resample=PIL.Image.BILINEAR).convert('RGB'), dtype=np.float32).transpose((2,0,1))/255 # pytorch dim
img = torch.from_numpy(img)
prediction = learner.predict(Image(img))
```
Note that the learner object, when predicting a single image, expects the images to be of type `fastai.vision.image.Image`, which in turn accepts a NumPy array. 

Here are the input and output images

![watermark added]({{ site.baseurl }}/images/watermark_added.jpg "Input Image to Model")
![Watermark removed from model]({{ site.baseurl }}/images/output.jpg "Output Image")

The model actually performs well and is able to remove alpha channel watermark with a good precision. The image is downsized and is 128 by 128 pixels owing to limited resources of my machine while training the model.

## Creating a web app

Deploying ML is one of the most talked about and yet the least explored part of the overall pipeline. One which is either dependent on separate deployment team after you've container-ised your app, or as a batch job which gets called by a cron job at a specified time. Either way, this takes away a lot of ownership and creates a void in the feedback loop of an ML monitoring systems, which is, at least as important as training lifecycle, if not more. Data driven apps require a robust feedback loop to quickly identify falling model performace and implement quick iterations to re-train models in production.
This part is one of the most exciting part, getting your trained model to produce results and expose it as an API. Exciting because it lets you use the model as a service, and, of course, you can design a useful front-end to make it look inviting. I'll be using Flask to create a simple interface to interact with your app. 

The interface will be created keeping in mind the requirements of our app. Also, I have minimal exposure to the front end and please excuse any noob error you find. 
At the very least, our app should have, at the front-end:
A. Upload feature to upload image.
B. An option to add watermark after uploading, if the user chooses to do so.
C. Download the watermark removed images, after running through the model.

And at the working end:
A. Simple data validations, to check input image conformance to the model.
B. Input image transformations.
C. Logging model metrics, which can be used to detect any fall in model performance.

Let's start with defining the project structure under `web deploy` folder

```
├── __init__.py
├──	api.py
├── static
│   ├── temp1.jpeg
│   └── temp2.jpeg
│   └── watermark.png
├── templates
│   ├── add_watermark.html
│   └── index.html
│   └── output.html
```

Where `api.py` is for defining the Flask App and other functions at the working end of our app. `static` will hold watermark needed to be added and other intermediate output from the model. `templates` will hold templates to render. In `api.py` script, let's define method for loading the trained model

```python
def load_model():
    # Define model
    global model
    try:
        model = load_learner(path='../../models/', file='wm_remove.pkl')
        model.model.eval()
    except:
        print('Error Loading model, check model')
```

The method is self-explanatory, except for `model` variable definition as global, which is required to make it available globally. Let's then define a `prepare_image` which will be called after the user has uploaded an image, to prepare image for running inference with the model. This basically does the same steps as we used while checking the model performance

```python
def prepare_image(image_path):
    imsize = (128, 128)
    img = PIL.Image.open(image_path)
    img = np.asarray(img.resize(imsize, resample=PIL.Image.BILINEAR).convert('RGB'),
                                 dtype=np.float32).transpose((2,0,1))/255 # pytorch dim
    img = torch.from_numpy(img)
    return Image(img)
```
 Coming to the front-end part, we'll define a simple index.html with an upload option and a couple other options of adding watermark and removing watermark.

```html
<html>
<head>
<title>Image Watermark removal model as a Flask API</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
    <h1>Image Watermark removal: UNet with ResNet34 Encoder Model</h1>
    <form action="/prediction" method="post" enctype="multipart/form-data">
      <input type="file" name="image" value="Upload">
      <input type="submit" value="Predict">
    </form>
    <form action="/addwatermark" method="post" enctype="multipart/form-data">
      <input type="file" name="image" value="Upload">
      <input type="submit" value="Add Watermark">
    </form>
</body>
```
The rest of the templates, `add watermark.html` and `output.html` have similar structure, except it shows the additional output in each case.
 In the `add_watermark` method we'll add watermark and render template which shows original and watermark added image, recall the watermark logic we defined before:

```python
@app.route("/addwatermark", methods=["POST"])
def add_watermark():
    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # Read the image in PIL format
            f = flask.request.files["image"]
            f.save(os.path.join('static/input.jpeg'))  # save file to disk

            # add watermark logic
            main = PIL.Image.open(f.filename)
            mark = PIL.Image.open('static/pngkit_copyright-symbol-png_185408.png')
            mask = mark.convert('L').point(lambda x: min(x, 25))
            mark.putalpha(mask)
            mark_width, mark_height = mark.size
            main_width, main_height = main.size
            aspect_ratio = mark_width / mark_height
            new_mark_width = main_width * 0.25
            mark.thumbnail((new_mark_width, new_mark_width / aspect_ratio), PIL.Image.ANTIALIAS)

            tmp_img = PIL.Image.new('RGB', main.size)

            for i in range(0, tmp_img.size[0], mark.size[0]):
                for j in range(0, tmp_img.size[1], mark.size[1]):
                    main.paste(mark, (i, j), mark)
                    main.thumbnail((8000, 8000), PIL.Image.ANTIALIAS)
            main.save('static/watermark_added.jpg', quality=100) # save watermarked file


            # Return the prediction to HTML Template
            return flask.render_template("add_watermark.html")
```
When you click `predict`, it calls `html_predict` which saves the generated output and then the watermark template renders it.

```python
@app.route("/prediction", methods=["POST"])
def html_predict():

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == "POST":
        print(flask.request.files)
        if flask.request.files.get("image"):
            # Read the image in PIL format
            f = flask.request.files["image"]

            # Preprocess the image and prepare it for classification.
            image = prepare_image(f.filename)
            prediction = learner.predict(image)
            # Classify the input image and then initialize the list of predictions to return to the client.
            im = PIL.Image.fromarray((prediction[2].numpy().transpose((1,2,0))*255).astype(np.uint8))
            im.save('static/output.jpg')
            return flask.render_template("output.html")
```
I also defined a `/predict` route for raw JSON responses.
Finally add the `app.run()` and optionally add `debug=True` to debug errors which you might run into while running for the first time.

```python
if __name__ == "__main__":
    print("Loading PyTorch model and starting server.")
    print("Please wait until server has fully started...")
    load_model()
    app.run()
```
Run the app from command line `python api.py` and you are good to go. By default the app opens up on `localhost:5000`.

# Concluding Note
In this article we breifly glanced through all the requisites for a data driven app from ideation to production. I skipped over a lot many details, U-Net architecture, model training iterations, serving challenges, model monitoring/logging etc. but gave you an idea of what it takes to do it. Use this idea in your own app and share it with everyone. Also, feel free to point out any errors you find in the implementation.
