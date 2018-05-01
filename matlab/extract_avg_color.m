function x = extract_avg_color(resized_img)
avg = imresize(resized_img,[8,8]);
avg = im2double(avg);
x = [reshape(avg(:,:,1),64,1);reshape(avg(:,:,2),64,1);reshape(avg(:,:,3),64,1)];
end