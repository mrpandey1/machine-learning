# Fruits-360: A dataset of images containing fruits and vegetables

## Version: 2019.09.21.0 ##

A high-quality, dataset of images containing fruits and vegetables. The following fruits and vegetables are included: 
Apples (different varieties: Crimson Snow, Golden, Golden-Red, Granny Smith, Pink Lady, Red, Red Delicious), Apricot, Avocado, Avocado ripe, Banana (Yellow, Red, Lady Finger), Beetroot Red, Blueberry, Cactus fruit, Cantaloupe (2 varieties), Carambula, Cauliflower, Cherry (different varieties, Rainier), Cherry Wax (Yellow, Red, Black), Chestnut, Clementine, Cocos, Dates, Eggplant, Ginger Root, Granadilla, Grape (Blue, Pink, White (different varieties)), Grapefruit (Pink, White), Guava, Hazelnut, Huckleberry, Kiwi, Kaki, Kohlrabi, Kumsquats, Lemon (normal, Meyer), Lime, Lychee, Mandarine, Mango (Green, Red), Mangostan, Maracuja, Melon Piel de Sapo, Mulberry, Nectarine (Regular, Flat), Nut (Forest, Pecan), Onion (Red, White), Orange, Papaya, Passion fruit, Peach (different varieties), Pepino, Pear (different varieties, Abate, Forelle, Kaiser, Monster, Red, Williams), Pepper (Red, Green, Yellow), Physalis (normal, with Husk), Pineapple (normal, Mini), Pitahaya Red, Plum (different varieties), Pomegranate, Pomelo Sweetie, Potato (Red, Sweet, White), Quince, Rambutan, Raspberry, Redcurrant, Salak, Strawberry (normal, Wedge), Tamarillo, Tangelo, Tomato (different varieties, Maroon, Cherry Red, Yellow), Walnut.

## Dataset properties ##

Total number of images: 82213.

Training set size: 61488 images (one fruit or vegetable per image).

Test set size: 20622 images (one fruit or vegetable per image).

Multi-fruits set size: 103 images (more than one fruit (or fruit class) per image)

Number of classes: 120 (fruits and vegetables).

Image size: 100x100 pixels.

Filename format: image_index_100.jpg (e.g. 32_100.jpg) or r_image_index_100.jpg (e.g. r_32_100.jpg) or r2_image_index_100.jpg or r3_image_index_100.jpg. "r" stands for rotated fruit. "r2" means that the fruit was rotated around the 3rd axis. "100" comes from image size (100x100 pixels).

Different varieties of the same fruit (apple for instance) are stored as belonging to different classes.

## Repository structure ##

Folders [Training](Training) and [Test](Test) contain all images used for training and testing.

Folder [test-multiple_fruits](test-multiple_fruits) contains images with multiple fruits. Some of them are partially covered by other fruits. Also, they were captured in different lighting conditions compared to the fruits from Training and Test folder. This is an excelent test for real-world detection.

## How to cite ##

Horea Muresan, [Mihai Oltean](https://mihaioltean.github.io), [Fruit recognition from images using deep learning](https://www.researchgate.net/publication/321475443_Fruit_recognition_from_images_using_deep_learning), Acta Univ. Sapientiae, Informatica Vol. 10, Issue 1, pp. 26-42, 2018.

## How we created the dataset ##

Fruits and vegetables were planted in the shaft of a low speed motor (3 rpm) and a short movie of 20 seconds was recorded. 

A Logitech C920 camera was used for filming the fruits. This is one of the best webcams available.

Behind the fruits we placed a white sheet of paper as background. 

However due to the variations in the lighting conditions, the background was not uniform and we wrote a dedicated algorithm which extract the fruit from the background. This algorithm is of flood fill type: 
we start from each edge of the image and we mark all pixels there, then we mark all pixels found in the neighborhood of the already marked pixels for which the distance between colors is less than a prescribed value. We repeat the previous step until no more pixels can be marked.

All marked pixels are considered as being background (which is then filled with white) and the rest of pixels are considered as belonging to the object.

The maximum value for the distance between 2 neighbor pixels is a parameter of the algorithm and is set (by trial and error) for each movie.

Pictures from the test-multiple_fruits folder were made with a Nexus 5X phone.

## History ##

Fruits were filmed at the dates given below (YYYY.MM.DD):

2017.02.25 - Apple (golden).

2017.02.28 - Apple (red-yellow, red, golden2), Kiwi, Pear, Grapefruit, Lemon, Orange, Strawberry.

2017.03.05 - Apple (golden3, Braeburn, Granny Smith, red2).

2017.03.07 - Apple (red3).

2017.05.10 - Plum, Peach, Peach flat, Apricot, Nectarine, Pomegranate.

2017.05.27 - Avocado, Papaya, Grape, Cherrie.

2017.12.25 - Carambula, Cactus fruit, Granadilla, Kaki, Kumsquats, Passion fruit, Avocado ripe, Quince.

2017.12.28 - Clementine, Cocos, Mango, Lime, Lychee.

2017.12.31 - Apple Red Delicious, Pear Monster, Grape White.

2018.01.14 - Banana, Grapefruit Pink, Mandarine, Pineapple, Tangelo.

2018.01.19 - Huckleberry, Raspberry.

2018.01.26 - Dates, Maracuja, Plum 2, Salak, Tamarillo.

2018.02.05 - Guava, Grape White 2, Lemon Meyer

2018.02.07 - Banana Red, Pepino, Pitahaya Red.

2018.02.08 - Pear Abate, Pear Williams.

2018.05.22 - Lemon rotated, Pomegranate rotated

2018.05.24 - Cherry Rainier, Cherry 2, Strawberry Wedge.

2018.05.26 - Cantaloupe (2 varieties).

2018.05.31 - Melon Piel de Sapo.

2018.06.05 - Pineapple Mini, Physalis, Physalis with Husk, Rambutan.

2018.06.08 - Mulberry, Redcurrant.

2018.06.16 - Cherry Red, Hazelnut, Walnut, Tomato.

2018.06.17 - Cherry Wax (Yellow, Red, Black).

2018.08.19 - Apple Red Yellow 2, Grape Blue, Grape White 3, Grape White 4, Peach 2, Plum 3, Tomato Maroon, Tomato 1-4.

2018.12.20 - Nut Pecan, Pear Kaiser, Tomato Yellow.

2018.12.21 - Banana Lady Finger, Chesnut, Mangostan.

2018.12.22 - Pomelo Sweetie.

2019.04.21 - Apple Crimson Snow, Apple Pink Lady, Blueberry, Kohlrabi, Mango Red, Pear Red, Pepper (Red, Yellow, Green).

2019.06.18 - Beetroot Red, Ginger Root, Nectarine Flat, Nut Forest, Onion Red, Onion Red Peeled, Onion White, Potato Red, Potato Red Washed, Potato Sweet, Potato White.

2019.07.07 - Cauliflower, Eggplant, Pear Forelle.

## License ##

MIT License

Copyright (c) 2017-2019 [Mihai Oltean](https://mihaioltean.github.io), Horea Muresan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
