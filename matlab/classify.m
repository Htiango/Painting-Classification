% use library from https://www.mathworks.com/matlabcentral/fileexchange/55863-logistic-regression-for-classification
function classify(styles_name)
file_set_1 = dir(['../data/paintings/' , styles_name{1}]);
file_set_2 = dir(['../data/paintings/' , styles_name{2}]);
train_size = 500;
N1 = train_size;
N2 = train_size;
dim = 2916 + 640;
tag = zeros(1, N1 + N2);
for i = 1:1: train_size
    tag(1,i) = 1;
end
size(file_set_2)
k = 0;
X = zeros(dim,N1 - 2 + N2 - 2);
i = 0;
while(i < N1)
    i = i + 1;
    if file_set_1(i).isdir
        fprintf('dir image %d\n', i)
        N1 = N1 + 1;
        continue
    end
    pic_name = [ '../data/paintings/', styles_name{1},'/', file_set_1(i).name];
    try
        img = imread(pic_name); 
    catch
        fprintf('error image %d\n', i)
        N1 = N1 + 1;
        continue
    end
    try
    features = [extract_hog_feature(img); extract_feature(img)];
    k = k + 1;
    X(:,k) = features;
    fprintf('image %d\n', i)
    catch
        fprintf('error image %d\n', i)
        N1 = N1 + 1;
        continue
    end
end

i = 0;
while(i < N2)
    i = i + 1;
    if file_set_2(i).isdir
        fprintf('dir image %d\n', i)
        N2 = N2 + 1;
        continue
    end
    pic_name = [ '../data/paintings/', styles_name{2},'/', file_set_2(i).name];
    try
        img = imread(pic_name); 
    catch
        fprintf('error image %d\n', i)
        N2 = N2 + 1;
        continue
    end
    try
    features = [extract_hog_feature(img); extract_feature(img)];
    k = k + 1;
    X(:,k) = features;
    fprintf('image %d\n', i)
    catch
        fprintf('error image %d\n', i)
        N2 = N2 + 1;
        continue
    end
end

[trained_model,~]=logitBin(X,tag);
test_size = 50;
t1 = test_size;
t2 = test_size;
k = 0;

TEST = zeros(dim,t1 + t2);
i = 0;
while(i < t1)
    i = i + 1;
    if file_set_1(N1 + i).isdir
        fprintf('dir image %d\n', i)
        t1 = t1 + 1;
        continue
    end
    pic_name = [ '../data/paintings/', styles_name{1},'/', file_set_1(N1 + i).name];
    try
        img = imread(pic_name); 
    catch
        fprintf('error image %d\n', i)
        t1 = t1 + 1;
        continue
    end
    try
    features = [extract_hog_feature(img); extract_feature(img)];
    k = k + 1;
    TEST(:,k) = features;
    fprintf('image %d\n', i)
    catch
         fprintf('error image %d\n', i)
        t1 = t1 + 1;
        continue
    end
end

i = 0;
while(i < t2)
    i = i + 1;
    if file_set_2(N2 + i).isdir
        fprintf('dir image %d\n', i)
        t2 = t2 + 1;
        continue
    end
    pic_name = [ '../data/paintings/', styles_name{2},'/', file_set_2(N2 + i).name];
    try
        img = imread(pic_name); 
    catch
        fprintf('error image %d\n', i)
        t2 = t2 + 1;
        continue
    end
    try
    features = [extract_hog_feature(img); extract_feature(img)];
    k = k + 1;
    TEST(:,k) = features;
    fprintf('image %d\n', i)
    catch
        fprintf('error image %d\n', i)
        t2 = t2 + 1;
        continue
    end
end

[y, ~] = logitBinPred(trained_model,TEST);
error = sum(y(1,1:test_size)) + test_size - sum(y(1,test_size + 1: test_size * 2))
accuracy = (test_size * 2 - error) / (test_size * 2);
 fprintf([styles_name{1} , '  ', styles_name{2}, '\n'])
 fprintf('accuracy %f % \n', accuracy *100);
end

