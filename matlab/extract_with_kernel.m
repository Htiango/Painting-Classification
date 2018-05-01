function x = extract_with_kernel(resized_img, kernel, width, height)
processed_img = imfilter(resized_img, kernel);
x = sel_64_fixed(processed_img, width, height);
end