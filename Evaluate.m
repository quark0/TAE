% code modified based on
% http://ranger.uta.edu/~heng/Drosophila/utility/f_cal_stat_info.m

function [stat_info, stat, class_stat] = Evaluate(ground_truth_matrix, predicted_output)

% function [stat_info, class_stat_info] = f_cal_stat_info(ground_truth_matrix, nFold_indices, one_fold_test, predicted_output)
%
% 	stat_info:
% 		(1) macro precision; (2) macro recall; (3) macro accuracy; (4) macro F1
% 		(5) micro precision; (6) micro recall; (7) micro accuracy; (8) micro F1
% 		(9) Hamming loss
%
% 	stat:
%	 	first column: TP
% 		second column: FP
% 		third column: TN
% 		fourth column: FN
%
% 	class_stat:
%	 	first column: TP
% 		second column: FP
% 		third column: TN
% 		fourth column: FN
% 		fifth column: Precision
% 		sixth colum: Recall
% 		seventh column: Accuracy
% 		eighth column: F1

% tp all
stat(1) = nnz((ground_truth_matrix == 1) .* (predicted_output == 1));
% fp all
stat(2) = nnz((ground_truth_matrix == 0) .* (predicted_output == 1));
% tn all
stat(3) = nnz((ground_truth_matrix == 0) .* (predicted_output == 0));
% fn all
stat(4) = nnz((ground_truth_matrix == 1) .* (predicted_output == 0));

% Precision: TP / (TP + FP)
stat_info(1) = stat(1) / (stat(1) + stat(2)+eps);
% Recall: TP / (TP + FN)
stat_info(2) = stat(1) / (stat(1) + stat(4)+eps);
% Accuracy: (TP + TN) / (TP + FP + TN + FN)
stat_info(3) = (stat(1) + stat(3)) / (sum(stat)+eps);
% F1 : 2 * Precision * Recall / (Precision + Recall)
stat_info(4) = 2 * stat_info(1) * stat_info(2) / (stat_info(1) + stat_info(2)+eps);

class_stat = zeros(size(ground_truth_matrix, 2), 8);
for idx_func = 1 : size(ground_truth_matrix, 2)
	% tp
	class_stat(idx_func, 1) = nnz((ground_truth_matrix(:, idx_func) == 1) .* (predicted_output(:, idx_func) == 1));
	% fp
	class_stat(idx_func, 2) = nnz((ground_truth_matrix(:, idx_func) == 0) .* (predicted_output(:, idx_func) == 1));
	% tn
	class_stat(idx_func, 3) = nnz((ground_truth_matrix(:, idx_func) == 0) .* (predicted_output(:, idx_func) == 0));
	% fn
	class_stat(idx_func, 4) = nnz((ground_truth_matrix(:, idx_func) == 1) .* (predicted_output(:, idx_func) == 0));
end
class_stat(:, 5) = class_stat(:, 1) ./ (class_stat(:, 1) + class_stat(:, 2)+eps);
class_stat(:, 6) = class_stat(:, 1) ./ (class_stat(:, 1) + class_stat(:, 4)+eps);
class_stat(:, 7) = (class_stat(:, 1) + class_stat(:, 3)) ./ (sum(class_stat(:, 1 : 4), 2)+eps);
class_stat(:, 8) = 2 * class_stat(:, 5) .* class_stat(:, 6) ./ (class_stat(:, 5) + class_stat(:, 6)+eps);

stat_info(5 : 8) = mean(class_stat(:, 5 : 8));

end
