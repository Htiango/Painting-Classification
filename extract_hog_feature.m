function x = extract_hog_feature(img)
width = 640;
height = 640;
resized_img = imresize(img,[width,height]);
x_temp = double(extractHOGFeatures(resized_img, 'CellSize', [64,64]));
x = x_temp';
end