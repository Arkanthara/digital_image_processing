r"""°°°
<img style="float: left;" src="images/unige_csd.png" alt="drawing" width="200"/>

# <center>Imagerie Numérique 2023 Automne</center>

<center>October 26, 2023</center>
°°°"""
# |%%--%%| <1W5a5BHCnD|ysIiWWDRcb>
r"""°°°
# <center><ins>TP Class N°3 - Image acquisition and sensing</ins></center>
°°°"""
# |%%--%%| <ysIiWWDRcb|L11SERXPLo>
r"""°°°
#### *Instructions :*  

* This TP should be completed and uploaded on Moodle before **Thursday 9 November 2023, 23h59**.
* The name of the file you upload should be **TP3_name_surname.ipynb**.
* If you need to include attached files to you TP, please archive them together in a folder named **TP3_name_surname.zip**.
°°°"""
# |%%--%%| <L11SERXPLo|VkOakuICGo>
r"""°°°
# Exercise 1
°°°"""
# |%%--%%| <VkOakuICGo|vAp1n3HWY8>
r"""°°°
Have a look at the following image :

![photography main parameters](images/photography_main_parameters.jpg)
°°°"""
# |%%--%%| <vAp1n3HWY8|TRioLvUzEh>
r"""°°°
Using these illustrations, explain the concepts of Aperture, Shutter speed and ISO. How are they built in a modern Digital Single-Lens Reflex ?
°°°"""
# |%%--%%| <TRioLvUzEh|Ou5S6BzqSs>
r"""°°°
# Exercise 2
°°°"""
# |%%--%%| <Ou5S6BzqSs|qUV01RgoPP>
r"""°°°
(a) In a 100 × 100 RGB image each pixel is represented by 256 levels of intensity. How many bytes are needed to store these image without any compression?
°°°"""
# |%%--%%| <qUV01RgoPP|xCuPD6r2Oj>
r"""°°°
For coding 1 pixel, we need 8 bits: $2^8 = 256$, and a byte is equal to 8 bits. So to store the image RGB $100\times100$ without any compression,
$$\frac{100 \times 100 \times 3 \times 8}{8} = 30000\ bytes$$
are needed.
°°°"""
# |%%--%%| <xCuPD6r2Oj|mWAK8s3RfQ>
r"""°°°
(b) In a 100 × 100 gray-scale image each pixel is represented by 4 levels of intensity. How many bytes are needed to store these image without any compression?
°°°"""
# |%%--%%| <mWAK8s3RfQ|IULJhGE29g>
r"""°°°
Each pixel is represented by 4 levels of intensity. So we need 2 bits to encode this information: $log_2(4) = 2$. So to store the gray-scale image $100 \times 100$ without any compression, we need:
$$\frac{100 \times 100 \times 2}{8} = 2500\ bytes$$ 
°°°"""
# |%%--%%| <IULJhGE29g|8KOx488yPH>

print(100*100*2/8)

# |%%--%%| <8KOx488yPH|jBdbu7OB94>
r"""°°°
(c) Generate a $100 \times 100$ RGB image constituted of uniform random noise (use *numpy.random.uniform()*). Save it as a png file using *plt.imsave()*. Comment on the size of the file.

**Hint :** In order to understand what is going on, you might want to load the image again in Python using *plt.imread()*
°°°"""
# |%%--%%| <jBdbu7OB94|6bBbjs51ai>

import matplotlib.pyplot as plt
import numpy as np

def generate_image() -> np.ndarray:
    image = np.random.uniform(size=(100, 100, 3))
    image = np.float32(image)
    return image

def print_informations_image(image: np.ndarray):
    print("Image data type: " + str(image.dtype))
    print("Image shape: " + str(image.shape))

def print_image(image: np.ndarray, title: str, cmap='viridis'):
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()

def MSE(image1: np.ndarray, image2: np.ndarray):
    return np.linalg.norm(image1 - image2)
    
def save_image(image: np.ndarray, name: str):
    plt.imsave('./images_saved/' + name, image)

def load_image(name: str, path='./images_saved/'):
    return plt.imread(path + name)

random_image = generate_image()
print_informations_image(random_image)
print_image(random_image, "Image random generated")
save_image(random_image, 'saved.png')
random_image_saved = load_image('saved.png')
print_informations_image(random_image_saved)


# |%%--%%| <6bBbjs51ai|hvtSyyDVWW>
r"""°°°
(d) Generate a $100 \times 100$ grayscale gradient image (see TP1 ex 2). Save it again as a png file. Comment.
°°°"""
# |%%--%%| <hvtSyyDVWW|fiHvt5l7oS>

def generate_gradient_image(heigh: int, weight: int) -> np.ndarray:
    gradient = np.linspace(0, 255, weight, dtype=np.uint8)
    gradient = np.tile(gradient, (heigh, 1))
    return gradient

gradient = generate_gradient_image(100, 100)
print_informations_image(gradient)
print_image(gradient, "Gradient image", 'gray')
save_image(gradient, 'gradient.png')
gradient_saved = load_image('gradient.png')
print_informations_image(gradient_saved)


# |%%--%%| <fiHvt5l7oS|yXwsPPAR1B>
r"""°°°
# Exercise 3
°°°"""
# |%%--%%| <yXwsPPAR1B|ZzIeTEICuA>
r"""°°°
(a) Explain the difference between sampling and quantization.
sampling: pixels de l'image: rendre la grille des pixels
quantization: distrétiser les valeurs dans la grille
°°°"""
# |%%--%%| <ZzIeTEICuA|bU8R0gHecW>
r"""°°°
For a continuous function given, sampling cuts our functions in regurar intervales, whereas quantization make an approximation of our function... For instance, quantization create a grid on our function thanks to the sampling, and then it put all the elements of our function on the grid...
°°°"""
# |%%--%%| <bU8R0gHecW|g4jmz7TC2u>

def example():
    x = np.linspace(0, 5, 100)
    y = np.sin(x)
    x_grid = [i for i in range(6)]
    y_grid = np.sin(x_grid)
    plt.figure()
    # plt.subplot(1, 3, 1)
    plt.plot(x, y)
    plt.title('Function sinus')
    plt.show()
    # plt.subplot(1, 3, 2)
    plt.plot(x, y)
    plt.plot(x_grid, y_grid, 'ro')
    plt.title('Sampling')
    plt.grid(axis = 'x')
    plt.show()
    # plt.subplot(1, 3, 3)
    plt.plot(x, y)
    plt.plot(x_grid, y_grid, 'r.')
    plt.plot([1, 1], [y_grid[1], 0.75], 'r-')
    plt.plot([2, 2], [y_grid[2], 1], 'r-')
    plt.plot([3, 3], [y_grid[3], 0.25], 'r-')
    plt.plot([4, 4], [y_grid[4], - 0.75], 'r-')
    plt.plot([5, 5], [y_grid[5], -1], 'r-')
    y_quantized = [0, 0.75, 1, 0.25, -0.75, -1]
    plt.plot(x_grid, y_quantized, 'gx')
    plt.title("Quantization")
    plt.grid()
    plt.show()
    plt.figure()
    plt.plot(x, y)
    plt.plot(x_grid, y_quantized)
    plt.grid()
    plt.title("Result after sampling 1 and quatization 1/4")
    plt.show()

example()

# |%%--%%| <g4jmz7TC2u|bd4gHmtScr>
r"""°°°
(b) You are given a continuous signal  *$f(x) = sin(x)+\frac{1}{10} cos(10x)$* over the interval *$0 \leq  x \leq 8 \pi$* and *$-1.1 \leq y \leq 1.1$*.

Using *np.linspace()* and *plt.plot()*, visualize this continuous signal on the given interval with a high number of samples.
°°°"""
# |%%--%%| <bd4gHmtScr|wrFXWFGyfL>

def visualize_signal():
    x = np.linspace(0, 8*np.pi, 10000)
    y = np.sin(x) + np.cos(x)/10
    plt.figure()
    plt.plot(x, y)
    plt.title("Signal given")
    plt.show()

visualize_signal()

# |%%--%%| <wrFXWFGyfL|uL1l5UfQGW>
r"""°°°
(c) Choose various values of sampling and quantization for this signal and plot the results on a grid of subplots, varying both parameters. Comment on the quality of the approximation.

**Hint :** Use *np.linspace()* and *np.digitize()* to generate the correct sampling and quantizations, try different values of samples and bins.
°°°"""
# |%%--%%| <uL1l5UfQGW|C7xlZdTmHZ>

def sampling_quantize(x_min, x_max, sampling):
    x = np.linspace(x_min, x_max, sampling)
    y = np.sin(x) + np.cos(x)/10
    y = np.digitize(y, np.linspace(-1.1, 1.1, 20))
    plt.figure()
    plt.plot(x, y)
    plt.show()

sampling_quantize(0, 8*np.pi, 500)
    

# |%%--%%| <C7xlZdTmHZ|nfMgnpi5cV>
r"""°°°
> Answer here
°°°"""
# |%%--%%| <nfMgnpi5cV|HxVVvt1EBJ>
r"""°°°
# Exercise 4
°°°"""
# |%%--%%| <HxVVvt1EBJ|oAMEwy1SXo>
r"""°°°
(a) Generate a gradient image like the one represented in Figure 1. Encode the image with $k=7, 5, 3, 2, 1$ bits (Theme 3, page 109). Display and explain the results.

<figure>
<center>
<img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgMAAACGCAIAAADPWx5CAAAAA3NCSVQICAjb4U/gAAAAGXRFWHRTb2Z0d2FyZQBnbm9tZS1zY3JlZW5zaG907wO/PgAAAg1JREFUeJzt17ERAyAQBDHwuCT6rwB6cuAiPlgpONLPdtj33gVA2Gf6AACGKQFA3ff/nHPWWntva2s7foC1g/veW/4EACgBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1CkBQJ0SANQpAUCdEgDUKQFAnRIA1O177/QNAEzyJwCoUwKAuh+FIAvVlYufEwAAAABJRU5ErkJggg==' width="150" height="150"/>
<figcaption>Figure 1: Gradient image</figcaption></center>
</figure>
°°°"""
# |%%--%%| <oAMEwy1SXo|Nx3aRPrrdU>

gradient_image = generate_gradient_image(100, 100)
print_image(gradient_image, 'Gradient image', 'gray')

# |%%--%%| <Nx3aRPrrdU|UluF8A95Qr>
r"""°°°
> Answer here
°°°"""
# |%%--%%| <UluF8A95Qr|RzrynXPiJA>
r"""°°°
(b) Do the same for the grayscale image *lena.png*.  Display the obtained results.
°°°"""
# |%%--%%| <RzrynXPiJA|GrMybqoqf3>



# |%%--%%| <GrMybqoqf3|KUw9jaZQf4>
r"""°°°
> Answer here
°°°"""
# |%%--%%| <KUw9jaZQf4|artRV4mfkp>
r"""°°°
# Exercise 5
°°°"""
# |%%--%%| <artRV4mfkp|uG6ox0IRA5>
r"""°°°
(a) Write the function that measures PSNR value between two images (see Theme 2, Lecture notes).
°°°"""
# |%%--%%| <uG6ox0IRA5|e0DfUblAfa>

def PSNR(image_1: np.ndarray, image_2: np.ndarray) -> float:
    return 10*np.log10(np.max(image_1)/MSE(image_1, image_2))

# |%%--%%| <e0DfUblAfa|OGViSS6Thw>
r"""°°°
(b) Read the image *lena.png* and convert it to grayscale with dynamic range in $[0,1]$. Create 10 noisy lena images by adding a zero-mean white Gaussian noise with standard deviation $\sigma = 0.1$.
°°°"""
# |%%--%%| <OGViSS6Thw|bVDDhl0k7O>



# |%%--%%| <bVDDhl0k7O|8Wj8fZX8tw>
r"""°°°
(c) Report the average PSNR value between the original and noisy images.
> **Hint** Measure the PSNR between the original and each noisy image, then compute the mean of the results.
°°°"""
# |%%--%%| <8Wj8fZX8tw|UYgcZTX2ww>



# |%%--%%| <UYgcZTX2ww|vEDQiRZkbC>
r"""°°°
(d) Perform image denoising by using the so named _frame averaging_ approach.
> **Hint** Perform a pixel-wise summation of all noisy images. Divide the obtained sum image by the number of images in the summation.

Ex: on prend plusieurs photos d'un même objet, on fait la somme de ces 3 photos et du bruit de ces 3 photos, et on divise par 3
°°°"""
# |%%--%%| <vEDQiRZkbC|Va0ZM88rEB>



# |%%--%%| <Va0ZM88rEB|btu7CS2oLz>
r"""°°°
(e) Measure the PSNR between the original and the denoised image. Comment the obtained result in the light of the previous computations. Explain when (under which condition) _frame averaging_ is successful and when it does not work.
°°°"""
# |%%--%%| <btu7CS2oLz|77THcG79Ep>



# |%%--%%| <77THcG79Ep|Q8YYOX0sOu>
r"""°°°
> -- your answer --
°°°"""
# |%%--%%| <Q8YYOX0sOu|5X2qBXmPbs>
r"""°°°
# Exercise 6
°°°"""
# |%%--%%| <5X2qBXmPbs|CdiXdxfzM4>
r"""°°°
You are given a pair of two images (reference and noisy) from the [RENOIR dataset](http://ani.stat.fsu.edu/~abarbu/Renoir.html).

<p align="center">
<img src="images/reference.bmp" alt="reference" width="300"/>
<img src="images/noisy.bmp" alt="noisy" width="300"/>
</p>
°°°"""
# |%%--%%| <CdiXdxfzM4|p65H0ekSyY>
r"""°°°
(a) Visualize each color channels for both images (a grayscale display of each channel). Are all channels equally affected by the noise? Justify your answer based on the _PSNR_ or _MSE_.
°°°"""
# |%%--%%| <p65H0ekSyY|P0xnCDSGtU>



# |%%--%%| <P0xnCDSGtU|IquRaVazGH>
r"""°°°
> --your answer --
°°°"""
# |%%--%%| <IquRaVazGH|2eGLMerlxh>
r"""°°°
(b) Try to decrease the noise by downsampling the image 2 times and then upsampling it back to its original size. Apply this method to the RGB noisy image. Measure the PSNR between the reference and the obtained denoised images.
> **Hint** To measure the PSNR between RGB images, compute the PSNR for each color channel and then take the average value.

Utiliser imresize ou resize ou quelque chose du genre...
°°°"""
# |%%--%%| <2eGLMerlxh|RkF1wiDEfy>



# |%%--%%| <RkF1wiDEfy|RkgD0SEfnU>
r"""°°°
(c) Convert both images to grayscale and redo part (b). Explain why the PSNR is higher for the denoised grayscale image.
> **Hint**: The reason is linked to exercise 5. Explain why.
°°°"""
# |%%--%%| <RkgD0SEfnU|jC94BTRfW5>



# |%%--%%| <jC94BTRfW5|F7cO5krs7f>
r"""°°°
> --your answer --
°°°"""
# |%%--%%| <F7cO5krs7f|t5YuvcknUV>
r"""°°°
(d) What other methods could you suggest to improve the noisy image quality?
°°°"""
# |%%--%%| <t5YuvcknUV|mkIVcs8Cw8>
r"""°°°
> --your answer --
°°°"""
# |%%--%%| <mkIVcs8Cw8|z2VjnsVAn4>
r"""°°°
___
°°°"""