%--------------------------------------
% CSCI 59000 Biometrics - face recognition
% Author: Chu-An Tsai
% 04/09/2020
%--------------------------------------

clear,clc;

% List out all 9 people in the database
figure('Name' , 'People in the database');
for i = 1 : 9
    subplot(3,3,i)
    img = imread(strcat('yalefaces\',num2str(i),'0',num2str(1),'.gif'));
    imshow(img)
    title(['person #',num2str(i)])
end

% Convert each face image into a 1 by H*W vector and stack all 9 people
% Here I use 6 faces from each person
samples=[]; 
for i = 1 : 9    
    for j = 1 : 6        
        x = imread(strcat('yalefaces\',num2str(i),'0',num2str(j),'.gif'));                  
        y = x(1:160*120);         
        y = double(y);        
        samples = [samples;y];   
    end
end

samples_mean = mean(samples);
for i=1:54 
    samples_mean_metrix(i,:) = samples(i,:) - samples_mean;  
end 

% Prepare to do PCA
cov_mat = samples_mean_metrix * samples_mean_metrix';   
[v,d] = eig(cov_mat);
d1 = diag(d); 
[d2,index] = sort(d1);  
col_num = size(v,2); 

% Save the eigenvectors and eigenvalues in decending order
for i=1:col_num      
    eigenvectors_sorted(:,i) = v(:,index(col_num-i+1));    
    eigenvalues_sorted(i) = d1(index(col_num-i+1));  
end

% Get 80%
A = sum(eigenvalues_sorted);
B = 0;   
C = 0;     
while( B/A < 0.85)       
    C = C + 1;          
    B = sum(eigenvalues_sorted(1:C));     
end
x1 = 1:1:54;
for i = 1:1:54
    y1(i)=sum(eigenvalues_sorted(x1(1:i)) );
end

i=1; 
while (i<=C && eigenvalues_sorted(i)>0)      
    base(:,i) = eigenvalues_sorted(i)^(-1/2) * samples_mean_metrix' * eigenvectors_sorted(:,i);   
    i = i + 1; 
end

figure('Name' , 'Mean face')
imshow(mat2gray(reshape(samples_mean,160,120)));
title('Mean face')

figure('Name' , 'Eigenfaces')
for i=1:9
    faces_mean = reshape(base(:,i)',160,120);
    subplot(3,3,i)
    imshow(mat2gray(faces_mean));
end
%------------------------------------------------------------

% Import Traindatabase
train_data = dir('.\TrainDatabase\');
train_num = 0;

for i = 1:size(train_data,1)
    if not(strcmp(train_data(i).name,'.')|strcmp(train_data(i).name,'..'))
        train_num = train_num + 1; 
    end
end

Train_set = [];
for i = 1 : train_num
    file = strcat('.\TrainDatabase\',strcat(int2str(i),'.gif'));   
    img = imread(file);
    [H ,W] = size(img);
    space_projection = reshape(img',H*W,1);   
    Train_set = [Train_set space_projection];                   
end

% Compute the average face image
mean_face = mean(Train_set,2); 
num = size(Train_set,2);

% Calculate the differences and normalize it
sample_faces = [];  
for i = 1 : num
    space_projection = double(Train_set(:,i)) - mean_face; 
    sample_faces = [sample_faces space_projection]; 
end



% Find eigen-values and corresponding eigen-vectors and do dimensionality reduction
cov_mat = sample_faces'*sample_faces; 
[A,B] = eig(cov_mat); 

eigen_set = [];
for i = 1 : size(A,2) 
    if (B(i,i)>1)
        eigen_set = [eigen_set A(:,i)];
    end
end

eigen_faces = sample_faces * eigen_set; 

%{
figure()

for i=1:9
    faces_mean = reshape(eigen_faces(:,i),120,160);
    subplot(3,3,i)
    imshow(mat2gray(faces_mean'));
end

figure()
img = zeros(120,160);  
for i = 1 : 9  
    img(:) = eigen_faces(:,i)';  
    subplot(3,3,i);  
    imshow(img',[])  
end 
%}
image_projected = [];
num = size(eigen_faces,2);
for i = 1 : num
    space_projection = eigen_faces'*sample_faces(:,i);%????????????
    image_projected = [image_projected space_projection]; 
end

% List out all test images for users to select
figure('Name' , 'Faces in the Test database');
test_num = size(dir(fullfile('TestDatabase\')),1)-2;
for i = 1 : test_num
    subplot(ceil(test_num^(1/2))-2,ceil(test_num^(1/2))+2,i)
    img = imread(strcat('TestDatabase\',num2str(i),'.gif'));
    imshow(img)
    title(num2str(i))   
end

while 1
    user_input = input('Enter the number of the test image:\n');
    num_lines= 1;
    def = {'1'};
    image_test = strcat('.\TestDatabase\',num2str(user_input),'.gif');

    image_test = imread(image_test);
    space_projection = image_test(:,:,1);

    [H,W] = size(space_projection);
    test_face = reshape(space_projection',H*W,1);
    dif = double(test_face) - mean_face; 
    eigne_face_test = eigen_faces'*dif;


    dis_data = [];
    for i = 1 : num
        space_projection = (norm(eigne_face_test - image_projected(:,i)))^2;
        dis_data = [dis_data space_projection];
    end

    [ ~,index_match1] = min(dis_data);
    dis_data(index_match1) = max(dis_data);
    [ ~,index_match2] = min(dis_data);
    dis_data(index_match2) = max(dis_data);
    [ ~,index_match3] = min(dis_data);

    first_match = strcat(int2str(index_match1),'.gif');
    second_match = strcat(int2str(index_match2),'.gif');
    third_match = strcat(int2str(index_match3),'.gif');


    %-----------------------


    first = imread(strcat('.\TrainDatabase\',first_match));
    second = imread(strcat('.\TrainDatabase\',second_match));
    third = imread(strcat('.\TrainDatabase\',third_match));


    figure('Name' , 'Recognition result')
    subplot(1,4,1)
    imshow(image_test,'InitialMagnification','fit')                   
    title('Test Image');  
    subplot(1,4,2)
    imshow(first,'InitialMagnification','fit');
    ylabel(first)
    title({'1st match',[num2str(index_match1),'.gif']});
    subplot(1,4,3)
    imshow(second,'InitialMagnification','fit');
    title({'2nd match',[num2str(index_match2),'.gif']});
    subplot(1,4,4)
    imshow(third,'InitialMagnification','fit');
    title({'3rd match',[num2str(index_match3),'.gif']});
    
    if (input('Do you want to continue? (Y/N) \n','s') == 'N')
        break;
    end
end