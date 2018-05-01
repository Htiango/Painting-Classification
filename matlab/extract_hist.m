function x = extract_hist(resized_img, i, width, height)
h = imhist(resized_img(:,:,i), 64);
x = h / norm(h,1);
end