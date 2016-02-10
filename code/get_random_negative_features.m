% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);

%Calculate template dimensionality
D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;

num_feat_per_image = ceil(num_samples/num_images) + 3;

new_num_samples = num_feat_per_image * num_images;

fprintf('new num of samples = %d\n',new_num_samples);

yValue = 1;
for i = 1:num_images
    
    rawImg = imread(getfield(image_files(i),'name'));
    image = single(rawImg)/255;
    
    if size(image,3) > 1
        image = rgb2gray(rawImg);
    end
    
    sizeOfImg = size(image);
    
    height = sizeOfImg(1);
    width = sizeOfImg(2);
    fprintf('processing image %d \n',i);
    
    if i == 14
        fprintf('yay');
    end
    
    if min(width-feature_params.template_size,height-feature_params.template_size) < num_feat_per_image
        newNumFeatPerImage = min(width-feature_params.template_size,height-feature_params.template_size);
    else
        newNumFeatPerImage = num_feat_per_image;
    end
    randXArr = randperm(width-feature_params.template_size, newNumFeatPerImage);
    randYArr = randperm(height-feature_params.template_size, newNumFeatPerImage);
    
    for j = 1:newNumFeatPerImage
        
        segment = image(randYArr(j):randYArr(j)+feature_params.template_size, randXArr(j):randXArr(j)+feature_params.template_size);
        
        hog = vl_hog(single(segment), feature_params.hog_cell_size);
        
        sizeOfHog = size(hog);
        numOfMatrix = sizeOfHog(3);
        numOfRows = sizeOfHog(1);
        numOfCols = sizeOfHog(2);
    
    
        for z = 1:numOfMatrix
            for y = 1:numOfRows

                tempStartIndex = (z-1)*(numOfRows*numOfCols)+(y-1)*numOfCols + 1;
                tempEndIndex = (z-1)*(numOfRows*numOfCols)+(y-1)*numOfCols + numOfRows;

                temp_features_neg(yValue, tempStartIndex:tempEndIndex) = double(hog(y,:,z));
            end

        end
        yValue = yValue + 1;
    end
end

num_samples = min(yValue-1, num_samples);
randomIndex = randperm(yValue-1,num_samples);

   
features_neg(1:num_samples,:) = temp_features_neg(randomIndex,:);






