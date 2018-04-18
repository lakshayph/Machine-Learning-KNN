function knn_classify (training_file, test_file,k )
training_data  = importdata(training_file);
test_data  = importdata(test_file);
classifier = str2double(k);
distance = zeros(size(training_data,1),2);
dummy_matrix = zeros(classifier,2);
final_matrix = zeros(size(test_data,1),1);

num_columns = size(training_data,2); %number of columns of training_data
num_rows = size(training_data,1); %number of rows of training_data
num_rows_test = size(test_data,1); %number of rows of test_data

training = zeros(size(training_data,1),num_columns - 1);
test = zeros(size(test_data,1),num_columns - 1);

classes_training = training_data(:,num_columns);
unique_classes = unique(classes_training);
num_unique_classes = numel(unique_classes);


    for i=1:num_columns-1
        training(:,i)=(training_data(:,i) - mean(training_data(:,i)))/std(training_data(:,i),1);
        test(:,i)=(test_data(:,i) - mean(training_data(:,i)))/std(training_data(:,i),1);
    end
    
    for i=1:num_rows_test
        class_val = zeros(num_unique_classes,1);
        for j=1:num_rows
            distance(j,1) = sqrt(sum((test(i,:) - training(j,:)).*(test(i,:) - training(j,:))));
        end
        distance(:,2) = classes_training(:);
        sorted_distance = sortrows(distance,1);
        for q=1:classifier
            dummy_matrix(q,:) = sorted_distance(q,:);
        end
        if(classifier>1)
            for q = 1:num_unique_classes
                for t = 1:classifier
                    if (dummy_matrix(t,2) == unique_classes(q,1))
                        if(unique_classes(1,1) == 0)
                            class_val(unique_classes(q,1)+1,1) = class_val(unique_classes(q,1)+1,1) + 1;
                        else
                            class_val(unique_classes(q,1),1) = class_val(unique_classes(q,1),1) + 1;
                        end
                    end
                end
            end
            max_val = max(class_val);
            idx = find(class_val(:,1) == max_val);
            if(unique_classes(1,1) == 0)
                idx = idx-1;
            end
            for p=1:size(idx,1)
                if(idx(p,1)== test_data(i,num_columns))
                   fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n', i, idx(p,1), test_data(i,num_columns), 1/size(idx,1));
                   final_matrix(i,1) = 1/size(idx,1);
                   break;
                elseif((p == size(idx,1)) && (idx(p,1) ~= test_data(i,num_columns)))
                    fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n', i, idx(p,1), test_data(i,num_columns), 0);
                    final_matrix(i,1) = 0;
                end
            end
        else
            if(dummy_matrix(1,2) == test_data(i,num_columns))
                fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n', i, dummy_matrix(1,2), test_data(i,num_columns), 1);
                final_matrix(i,1) = 1;
            else
                fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n', i, dummy_matrix(1,2), test_data(i,num_columns), 0);
                final_matrix(i,1) = 0;
            end
        end   
    end
    fprintf('classification accuracy=%6.4f\n', mean(final_matrix));
end