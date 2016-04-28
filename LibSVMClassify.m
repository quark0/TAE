function stat_info = LibSVMClassify(X_train, Y_train, X_test, Y_test)

    addpath('liblinear-2.1/matlab');

    svm_opts = '-s 2 -q -c 1e5';

    % training
    num_categories = size(Y_train,2);
    for c = 1:num_categories
        model{c} = train(full(Y_train(:,c)), sparse(X_train), svm_opts);
    end

    % predicting
    Y_pred = zeros(size(X_test,1),num_categories);
    for c = 1:num_categories
        Y_pred(:,c) = predict(full(Y_test(:,c)), sparse(X_test), model{c}, '-q');
    end

    [stat_info, ~, ~] = Evaluate(Y_test, Y_pred);

end

