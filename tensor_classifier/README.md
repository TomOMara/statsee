# Overview
- - -

## Gathering Training Data
- - -

`cd` into `tensorflow-for-poets-2` where the body of the training will take place, adapted from [here](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc#0)

For each class of images you want to predict, we need to have true cases of each class.

For instance, as we are differentiating in this example between line and bar graphs, we need

    1.  a set of line graph images
    2.  a set of bar graph images.


To quickly get these images, grab [Fatkun image downloader](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf) from the chrome store which lets you bulk download images.


Point the default google directory inside the `/tf_files/`, and hit save images after deselecting the anomolies.


This will give us Neural Net classes for our neural net to learn, so now we must retrain it using our new classes.


## Retraining the Neural Net
- - -

Firstly set some parameters for the training, pasting time into console:

```
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"
```

Now we can retrain the Neural Net:

```
python -m scripts.retrain   --bottleneck_dir=tf_files/bottlenecks
                            --how_many_training_steps=500
                            --model_dir=tf_files/models/
                            --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}"
                            --output_graph=tf_files/retrained_graph.pb
                            --output_labels=tf_files/retrained_labels.txt
                            --architecture="${ARCHITECTURE}"
                            --image_dir=tf_files/line_graph_images/`
```

This takes around 30 seconds using the mobilenet architechture. For a real realease we should use the full architechture described in the tutorial linked above.


Now our Neural Net is trained, on to the next step!

## Making new Predictions
- - -

```
python -m scripts.label_image     --graph=tf_files/retrained_graph.pb      --image=path_to_image/image.png
```

And if everything worked you should see the following:


```
Evaluation time (1-image): 0.188s

line graph images 0.99486107
other graphs 0.0051389434
```

Ta Da!


