function [P, Q] = GraphAutoencoder(G, opt)

    [m,n] = size(G);

    if opt.eco == true
        [I,J,V] = find(G);
        E = [I,J,V]';

        ell = @(o) sum((E(3,:)-o).^2);
        ell_grad = @(o) 2*(o-E(3,:));

        fprop = @forward_eco;
        bprop = @backward_eco;

        input = E;
    else
        ell = @(O) sum(sum((G-O).^2));
        ell_grad = @(O) 2*(O-G);

        fprop = @forward;
        bprop = @backward;
        input = G; 
    end

    sigma_g = activation(opt.nonlinear_g);
    sigma_h = activation(opt.nonlinear_h);
    sigma_g_grad = activation_grad(opt.nonlinear_g);
    sigma_h_grad = activation_grad(opt.nonlinear_h);

    P = rand(m, opt.d)/sqrt(m);
    Q = rand(n, opt.d)/sqrt(n);

    ZP = zeros(size(P));
    ZQ = zeros(size(Q));
    VP = zeros(size(P));
    VQ = zeros(size(Q));

    for i = 1:opt.maxIter

        [Zg, Ag, Zh, Ah, loss] = fprop(input, ell, sigma_g, sigma_h, P, Q);
        [dl_dP, dl_dQ] = bprop(input, ell_grad, sigma_g_grad, sigma_h_grad, P, Q, Zg, Ag, Zh, Ah);

        if opt.gradient_check
            [dl_dP_est, dl_dQ_est] = gradient_check(input, ell, sigma_g, sigma_h, P, Q, fprop);
            fprintf('grad=%e, grad_est=%e\n', dl_dP(1,1), dl_dP_est);
            fprintf('grad=%e, grad_est=%e\n', dl_dQ(1,1), dl_dQ_est);
            pause;
        end

        % adagrad
        ZP = ZP + dl_dP.^2;
        ZQ = ZQ + dl_dQ.^2;
        % momentum
        VP = opt.momentum*VP + opt.eta0*dl_dP./sqrt(ZP);
        VQ = opt.momentum*VQ + opt.eta0*dl_dQ./sqrt(ZQ);

        P = P - VP;
        Q = Q - VQ;

        fprintf('iter=%d, l=%e\n', i, loss);
    end

    %% filling up the full matrix
    %[Ah, Ag, Zh, Ah, ~] = forward(G, @(O) sum(sum((G-O).^2)), sigma_g, sigma_h, P, Q);

end

function func = activation(type)
    switch type
        case 'sigmoid'
            func = @(x) 1./(1.+exp(-x));
        case 'tanh'
            func = @(x) tanh(x);
        case 'relu'
            func = @(x) max(x, 0);
        case 'identity'
            func = @(x) x;
    end
end

function func = activation_grad(type)
    switch type
        case 'sigmoid'
            sigmoid = activation(type);
            func = @(x) sigmoid(x).*(1.-sigmoid(x));
        case 'tanh'
            func = @(x) sech(x).^2;
        case 'relu'
            func = @(x) x>0;
        case 'identity'
            func = @(x) 1;
    end
end

function [dl_dP_11, dl_dQ_11] = gradient_check(input, ell, sigma_g, sigma_h, P, Q, fprop);
    eps = 1e-8;
    Eg = zeros(size(P));
    Eg(1,1) = eps;

    % gradient check P
    [~,~,~,~,loss_p] = fprop(input, ell, sigma_g, sigma_h, P+Eg, Q);
    [~,~,~,~,loss_n] = fprop(input, ell, sigma_g, sigma_h, P-Eg, Q);
    dl_dP_11 = (loss_p-loss_n)/(2*eps);

    % gradient check Q
    Eh = zeros(size(Q));
    Eh(1,1) = eps;
    [~,~,~,~,loss_p] = fprop(input, ell, sigma_g, sigma_h, P, Q+Eh);
    [~,~,~,~,loss_n] = fprop(input, ell, sigma_g, sigma_h, P, Q-Eh);
    dl_dQ_11 = (loss_p-loss_n)/(2*eps);
end

function [Zg, Ag, Zh, Ah, loss] = forward(G, ell, sigma_g, sigma_h, P, Q)
    Ah = diag(P'*G*Q);
    Zh = sigma_h(Ah);
    Ag = P*diag(Zh)*Q';
    Zg = sigma_g(Ag);
    loss = ell(Zg);
end

function [dl_dP, dl_dQ] = backward(G, ell_grad, sigma_g_grad, sigma_h_grad, P, Q, Zg, Ag, Zh, Ah)
    dl_dAg = ell_grad(Zg).*sigma_g_grad(Ag);
    T = (P'*dl_dAg*Q).*diag(sigma_h_grad(Ah));
    dl_dP = dl_dAg*Q*diag(Zh) + G*Q*T;
    dl_dQ = dl_dAg'*P*diag(Zh) + G'*P*T';
end

% X - a d*m matrix, [I,J,V] - an m*n sparse matrix
function Y = dense_times_sparse(X, I, J, V, n)
% computational bottleneck

    %{
        d = size(X,1);
        Y = zeros(d,n); % output size
        for k = 1:d
            Y(k,:) = accumarray(J', V'.*X(k,I)', [n,1])';
        end
    %}

    % NOTE: sparse linear algebra is not supported by GPU 
    Y = X*sparse(I,J,V,size(X,2),n);

end

function [Zg, Ag, Zh, Ah, loss] = forward_eco(E, ell, sigma_g, sigma_h, P, Q)
    Ah = E(3,:)*(P(E(1,:),:).*Q(E(2,:),:));
    Zh = sigma_h(Ah);
    Ag = Zh*(P(E(1,:),:).*Q(E(2,:),:))';
    Zg = sigma_g(Ag);
    loss = ell(Zg);
end

function [dl_dP, dl_dQ] = backward_eco(E, ell_grad, sigma_g_grad, sigma_h_grad, P, Q, Zg, Ag, Zh, Ah)
    dl_dAg = ell_grad(Zg).*sigma_g_grad(Ag);

    m = size(P,1);
    n = size(Q,1);
    Pt_dl_dAg = dense_times_sparse(P', E(1,:), E(2,:), dl_dAg, n);
    Qt_dl_dAgt = dense_times_sparse(Q', E(2,:), E(1,:), dl_dAg, m);

    T = sum(Pt_dl_dAg'.*Q).*sigma_h_grad(Ah);

    dl_dP = bsxfun(@times, Qt_dl_dAgt', Zh) + dense_times_sparse(bsxfun(@times, T', Q'), E(2,:), E(1,:), E(3,:), m)';
    dl_dQ = bsxfun(@times, Pt_dl_dAg', Zh) + dense_times_sparse(bsxfun(@times, T', P'), E(1,:), E(2,:), E(3,:), n)';
end

function E = dropout(E, p)
    s = size(E,2);
    t = ceil(s*p);
    E(3,randsample(s,t)) = 0;
end

