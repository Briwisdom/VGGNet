# VGGNet
My master research is sea ice classification based on deep learning with SAR imagery. This is one experiment demo of my master research.
my patch size choose 48*48 pixel. To vggnet16, the feature map  will be 2*2 after the five pooling layer, so the fifth converlution block and pooling layer are commented out in my code file.

Experiment result:
vggnet16 is just a demo of my experiment, the sea ice type is 7(they are: open water, new ice, grey ice, grey white ice, thin first year ice, medium first year ice, thick first year ice). The network does not converge and the accuracy is 0.1605.

run:
if you want run this code, you can copy my data format and run it directly. 
For example: python vggnet.py
