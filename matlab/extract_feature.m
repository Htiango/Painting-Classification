function x = extract_feature(img)
width = 640;
height = 640;
resized_img = imresize(img,[width,height]);
x1 = extract_gaussian_filter2(resized_img, width, height); % 64 by 1
x2 = extract_hist(resized_img, 1, width, height); % red
x3 = extract_hist(resized_img, 2, width, height); % green
x4 = extract_hist(resized_img, 3, width, height); % blue
h = fspecial('laplacian',0.2);
x5 = extract_with_kernel(resized_img, fspecial('laplacian',0.2), width, height);
horizontal = fspecial('prewitt');
vertical = horizontal';
x6 = extract_with_kernel(resized_img, horizontal, width, height);
x7 = extract_with_kernel(resized_img, vertical, width, height);
x8 = extract_avg_color(resized_img); % 192 by 1
x = [x1;x2;x3;x4;x5;x6;x7;x8];
end