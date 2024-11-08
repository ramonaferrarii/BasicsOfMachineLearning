% Upload the dataset converted in a numerical matrix: 
matrix = [1 1 1 2 1 ; 
           1 2 2 1 1 ; 
           1 3 1 1 1 ; 
           1 1 2 2 1 ; 
           2 3 1 2 1 ; 
           2 2 2 2 1 ; 
           2 2 2 1 2 ; 
           2 3 2 2 1 ; 
           2 3 1 1 2 ; 
           3 1 1 2 2 ; 
           3 1 1 1 2 ; 
           3 3 1 2 2 ; 
           3 2 2 2 1 ; 
           3 3 2 1 1];

% Division of dataset in A (Train Set) and B (Test Set):
% Dimensions of "matrix" 
n = size(matrix, 1); 
m = size(matrix, 2);

% Generate random vector of row indexes without repetitions 
random_index = randperm(n);

% Extract the first 10 rows of matrix A 
A = matrix(random_index(1:10), :);

% Number of rows of matrices 
num_rows_matrix = size(matrix, 1);
num_rows_A = size(A, 1);

% Initialization of B
B = [];

% Scrolling each row of the matrix 
for i = 1:num_rows_matrix 
    current_row = matrix(i,:);

% Check if the current row is present in the A matrix 
present = false; 
for j = 1:num_rows_A 
    if all(current_row == A(j,:)) 
        present = true; 
        break; 
    end 
end

% If the current line is not in A, appends it to B 
if ~present B = [B; current_row]; 
end 
end

%disp("Matrix A:"); % disp(A); % disp("Matrix B:"); % disp(B);

% Check if the number of columns of A is equal to the number of columns of
% B
if size(A,2) ~= size(B,2) 
   disp("Pay attention: matrixes A and B have different dimensions")
end 


% Check if all the values are more or equal than 1 
for i = 1:size(A, 1)
    for j = 1:size(B, 2)
       if A(i,j) <= 0
           disp("Pay attention: there is a 0 in the dataset!");
           break;
       end 
    end
end


% Computation of P(x|ti) :
% Computation of number of yes/no 
last_column = A(:, end); 
count_1 = sum(last_column == 1); % yes 
count_2 = sum(last_column == 2); % no

%disp("Number of yes:") 
%disp(count_1) 
%disp("Number of no:") 
%disp(count_2)

% Creation of the matrix of probabilities linked to yes: 
prob_matrix_yes = zeros(height(A),width(A)-1); 
%disp(prob_matrix_yes)


% Definition of the parameter alpha of Laplace Smoothing
% (added here to have the possibility to change its value)
alpha = 1;

for i = 1:width(A)-1 
    a = unique(A(:,i)); % it gives the unique elements of the columns 
    countyes = zeros(size(a)); 
    for j=1:height(A) 
        for z = 1:size(a)
            if (A(j,i) == a(z) && A(j, width(A)) == 1) % pay attention to the check 
                countyes(z) = countyes(z) + 1; 
            end 
        end 
    end

countyes = (countyes + alpha) / (count_1 + alpha*numel(a));

% Put these probabilities in the matrix:
prob_matrix_yes(1:size(countyes),i)= countyes; 
end

%disp(prob_matrix_yes)

prob_matrix_yes = rimuoviZerosRows(prob_matrix_yes);

%disp("P(x|yes):")
%disp(prob_matrix_yes)

% Creation of the matrix of probabilities linked to no: 
prob_matrix_no = zeros(height(A),width(A)-1);

for i = 1:width(A)-1
    a = unique(A(:,i)); 
    countno = zeros(size(a)); 
    for j=1:height(A) 
        for z = 1:size(a) 
            if (A(j,i) == a(z) && A(j, width(A)) == 2) 
                countno(z) = countno(z) + 1; 
            end 
        end 
    end

countno = (countno + alpha) / (count_2 + alpha*numel(a));

prob_matrix_no(1:size(countno),i)= countno; 
end

prob_matrix_no = rimuoviZerosRows(prob_matrix_no); 

%disp("P(x|no):") 
%disp(prob_matrix_no)

% Computation of P(ti) and P(tj) on A

prior_yes = count_1 / num_rows_A; % priori probability of "yes" 
prior_no = count_2 / num_rows_A; % priori probability of "no"

% Computation of unique values of A 
% It goes for columns 
UniqueValuesColumns = arrayfun(@(x) unique(A(:,x)), 1:size(A, 2), 'UniformOutput', false);

% Computation of the max number in a column
maxNumUniqueValue = max(cellfun(@(x) numel(x), UniqueValuesColumns));

% Initialization of a matrix with the correct dimensions 
UniqueMatrix = zeros(maxNumUniqueValue, size(A, 2));

% Creation of the matrix of unique values 
for column = 1:size(A, 2) 
    UniqueMatrix(1:numel(UniqueValuesColumns{column}), column) = UniqueValuesColumns{column}; 
end

%disp(UniqueMatrix);

% Naive Bayes Classifier : 
% Initialization of matrix of results 
results = [];

% Classification of each row of B
for row = 1:size(B, 1)
    % Update these probabilities every loop 
    prob_yes = prior_yes;
    prob_no = prior_no;

% Calculate conditional probabilities for each attribute
    for attribute = 1:size(B, 2)-1
   
        %  Get the attribute value in the current row of B
           value_attribute = B(row, attribute);
        
        % Find the corresponding index in the matriceUnici
        index_attribute = find(UniqueMatrix(:, attribute) == value_attribute);
        
        % Computation of conditional probabilities and priori probabilities
        prob_yes = prob_yes * prob_matrix_yes(index_attribute, attribute);
        prob_no = prob_no * prob_matrix_no(index_attribute, attribute);
    end
    
    % Classification:
    if prob_yes >= prob_no
        results = [results; B(row, :), 1];
    else
        results = [results; B(row, :), 2];
    end
end

% See the results: 
disp("In the last column we have the classification of the Naive Bayes Classifier: ");
disp(results);

% Computation of Error Rate 
errors = sum(results(:, end) ~= B(:, end)); error_rate = errors / size(B, 1) * 100;

disp("Error Rate:"); disp(error_rate);

% Function to remove rows of zeros 
function matrix = rimuoviZerosRows(matrix) 
matrix(sum(matrix == 0, 2) == size(matrix, 2), :) = []; 
end 

