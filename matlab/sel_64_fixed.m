function x = sel_64_fixed(processed_img, width, height)
off_x = width / 8 / 2;
off_y = width / 8 / 2;
unit_x = width / 8;
unit_y = width / 8;
x = [];
k = 0;
for i = off_x: unit_x: width
    for j = off_y: unit_y:height
        k = k + 1;
        x(k) = processed_img(j,i);
    end
end
x = x';
end