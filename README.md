<div align="center" markdown>

# On-the-fly Quality Assurance for Image Annotations

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/quality-assurance-for-image-annotations)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/quality-assurance-for-image-annotations)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/quality-assurance-for-image-annotations.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/quality-assurance-for-image-annotations.png)](https://supervisely.com)

</div>

## Overview

The app is claimed to unleash the full power of the [DatasetNinja.com](https://datasetninja.com) platform, allowing on-the-fly updates of project statistics. It is equipped with efficient features that enable rapid and reliable statistical calculations for images with updated or added annotations. Consider it like a library cabinet filled with boxes, each containing individual elements. When you need to recalculate statistics, it's as straightforward as computing the elements within a specific box. This approach allows for a focused update, seamlessly integrating the new or modified data chunk with the existing information without recalculating data.

## How To Run

The application is designed to run as a service, which allows it to receive requests from the `Statistics` tab in the project's settings after its initial start. Ensuring the application operates in the background is important to handle the statistics.

To run the app, do the following steps:

**Step 1:** Open the homepage of `On-the-Fly Quality Assurance` app in the supervisely ecosystem.<br><br>

**Step 2:** Run the application (_Note to admins_: It should be run as `admin` user, `Restart Policy` -> `On Error`).<br><br>

**Step 3:** After the successful start, choose the selected project and trigger the calculation of statistics by opening the `Statistics` app. (Promise - it won't take long 😊)

_Note to admins:_ To run the application across different users, it needs to be shared. To enable the feature, refer to the `Apps` sessions tab.

![image](https://github.com/supervisely-ecosystem/quality-assurance-for-image-annotations/assets/78355358/8c9bb566-3445-4fff-9a31-bfee2c4c60b9)

The calculated statistics will appear in the `stats` folder in team files.

## Chunks file structure

Every generated chunk has the universal description:

```
chunk_<chunk-index>_<dataset-id>_<project-id>_<chunk-size-images-per-batch>_<most-recent_updated_image>.npy
```

Be careful: the chunks' number of indexes is **derived from the number of images in a dataset**

## Statistics Description

**Class Balance:** Compare key properies of every class in the dataset.

Columns:

-   `Class` - Class name.
-   `Images` - Number of images with at least one object of corresponding class.
-   `Objects` - Number of objects of corresponding class in the project.
-   `Count on image` - Average number of objects of corresponding class on the image. Images without such objects are not taking into account.
-   `Area on image` - Average image area of corresponding class. Images without such objects are not taking into account.

**Class Cooccurence:** This statistics shows you the images for every pair of classes: how many images have objects of both classes at the same time.

-   `Class name 1` - The column with occurences of the first class with the other classes.
-   `etc.`

**Images:** Explore every single image in the dataset with respect to the number of annotations of each class it has.

Columns:

-   `Image` - File name of the image.
-   `Height` - Height of the image in pixels .
-   `Width` - Width of the image in pixels.
-   `Unlabeled` - Relative size (%) of the unlabeled area.
-   `Class name 1` - Average image area of corresponding class.
-   `etc.`

**Object Distribution:** Interactive heatmap chart for every class with object distribution shows how many images are in the dataset with a certain number of objects of a specific class.

Columns:

-   `Numbers of objects on image` - Corresponding number of objects on the image.

**Classes Treemap:** Visualize average area of every class on the image in relation to the other classes.

**Spatial Heatmap:** (_Not yet implemented_) Show the spatial distributions of all objects for every class. These visualizations provide insights into the most probable and rare object locations on the image. It helps analyze objects' placements in a dataset.

**Objects:** Explore every single object in the dataset.

Columns:

-   `Object ID` - Unique identifier of the object (i.e. the instance of the class).
-   `Class` - Name of the class and its shape.
-   `Image name` - Name of the image.
-   `Image size` - Height and Width of the image in pixels.
-   `Height px` - Height of the object in pixels.
-   `Height %` - Height (%) of the object relative to the height of the image.
-   `Width px` - Width of the object in pixels.
-   `Width %` - Width (%) of the object relative to the height of the image.
-   `Area %` - Relative object area (%).
