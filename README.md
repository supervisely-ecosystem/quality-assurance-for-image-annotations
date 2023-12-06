<div align="center" markdown>
<img src="poster placeholder"/>

# On-the-fly Quality Assurance for Image Annotations

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/PLACEHOLDER-FOR-APP-PATH)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/PLACEHOLDER-FOR-APP-PATH)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/PLACEHOLDER-FOR-APP-PATH.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/PLACEHOLDER-FOR-APP-PATH.png)](https://supervise.ly)

</div>

## Overview

The app is claimed to unleash the full power of the [DatasetNinja.com](https://datasetninja.com) platform, allowing on-the-fly update of project statistics. It is equipped with efficient features that enable rapid and reliable statistical calculations for images with updated or added annotations. Consider it like a library cabinet filled with boxes, each containing individual elements. When you need to recalculate statistics, it's as straightforward as computing the elements within a specific box. This approach allows for a focused update, seamlessly integrating the new or modified data chunk with the existing information.

## How To Run

For the project you wish to obtain the statistics, do the following steps:

**Step 1:** Click on the three-dot menu in the project card.<br><br>

**Step 2:** Choose `Reports & Stats -> Quality Assurance`.<br><br>

**Step 3:** Wait while full stastics will be initially calculated. If it was already calculated, wait for the statstics update. (Promise - it won't take long 😊)

## Statistics Description

**Class Balance:** Compare key properies of every class in the dataset.

Columns:

* `Class` - Class name.
* `Images` - Number of images with at least one object of corresponding class.
* `Objects` - Number of objects of corresponding class in the project.
* `Count on image` - Average number of objects of corresponding class on the image. Images without such objects are not taking into account.
* `Area on image` - Average image area of corresponding class. Images without such objects are not taking into account.

**Class Cooccurence:** This statistics shows you the images for every pair of classes: how many images have objects of both classes at the same time.

* `Class name 1` - The column with occurences of the first class with the other classes.
* `etc.`

**Images:** Explore every single image in the dataset with respect to the number of annotations of each class it has.

Columns:

* `Image` - File name of the image.
* `Height` - Height of the image in pixels .
* `Width` - Width of the image in pixels.
* `Unlabeled` - Relative size (%) of the unlabeled area.
* `Class name 1` - Average image area of corresponding class.
* `etc.`

**Object Distribution:** Interactive heatmap chart for every class with object distribution shows how many images are in the dataset with a certain number of objects of a specific class.

Columns:

* `Numbers of objects on image` - Corresponding number of objects on the image.

**Classes Treemap:** Visualize average area of every class on the image in relation to the other classes.

**Spatial Heatmap:** (*Not yet implemented*) Show the spatial distributions of all objects for every class. These visualizations provide insights into the most probable and rare object locations on the image. It helps analyze objects' placements in a dataset.

**Objects:** Explore every single object in the dataset.

Columns:

* `Object ID` - Unique identifier of the object (i.e. the instance of the class).
* `Class` - Name of the class and its shape.
* `Image name` - Name of the image.
* `Image size` - Height and Width of the image in pixels.
* `Height px` - Height of the object in pixels.
* `Height %` - Height (%) of the object relative to the height of the image.
* `Width px` - Width of the object in pixels.
* `Width %` - Width (%) of the object relative to the height of the image.
* `Area %` - Relative object area (%).
