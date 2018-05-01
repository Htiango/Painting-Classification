function x = extract_gaussian_filter2(img, width, height)
processed_img =  imgaussfilt(img, 2);
x = sel_64_fixed(processed_img, width, height);
end